"""Locked five-endpoint free-generation ADM FID on two GPUs, without training.

All endpoints share spatial-raw13173 generated1D and E117 routes. This isolates
2D interventions, NOT separate complete control/spatial1D pipelines. No teacher
images, cached1D, ground-truth codes, or source images enter generation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULT_ROOT = Path('/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/maskgit_optimization_sweep_20260910')
CHECKPOINT = RESULT_ROOT.parent / 'spatial_prefix_conditioning_20260910/spatial_bs128_from12938_plus2000_seed0_v1/latest.safetensors'
CONTROL = RESULT_ROOT / 'control_parent_plus235_seed0_v1/latest.safetensors'
SPATIAL_SHA = '58d28b69ae08392f1364ef8537c9fbb313e32d648356080ac77325ea35a102e1'
CONTROL_SHA = '84d2c8164d4747a4cd171675059943b7326c5b2d310c433398ffc3026a3d542b'
MOT_SHA = '86c8f9da5e61261ab93066c73d7719203e8c00b69f05b805c5937e6b7319b446'
ROUTER_SHA = 'a5b84689d2b29f579d2442da7594ac093292b6386760867a0668ca02f82e6156'
NAMES = ('base', 'control8', 'spatial8', 'spatial_margin4', 'spatial_margin4_blend050')
MOT = Path('/home/heyefei/lichenge/MoT/weights/sophiaa_root_latest/latest.pt')
ROUTER = Path('/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/dynamic/e116/direct_reconstruction_spatial_pilot_v1/latest.pt')
EVALUATOR = Path('/home/heyefei/lichenge/DetailFlow_change/evaluations/c2i/evaluator.py')
REFERENCE = EVALUATOR.parent / 'VIRTUAL_imagenet256_labeled.npz'
GRAPH = ROOT / 'classify_image_graph_def.pb'
ONE_D = dict(guidance_scale=4.5, guidance_decay='linear', randomize_temperature=9.5,
             softmax_temperature_annealing=False, num_sample_steps=8)
TWO_D = dict(num_steps=8, cfg_scale=4.5, randomize_temperature=1.,
             guidance_decay='constant', cfg_formula='standard',
             softmax_temperature_annealing=False)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def sorted_routes(decision):
    """Same ascending-coordinate contract as the existing E117 sampler."""
    import torch
    raw = decision['selected_indices_padded'].long()
    selected_valid = decision['selected_indices_valid'].bool()
    index = torch.zeros((len(raw), 128), dtype=torch.long, device=raw.device)
    valid = torch.zeros_like(index, dtype=torch.bool)
    for row in range(len(raw)):
        chosen = raw[row, selected_valid[row]].sort().values
        k = len(chosen)
        if k not in (64, 128) or chosen.unique().numel() != k:
            raise ValueError('duplicate or invalid E117 route count')
        if bool(((chosen < 0) | (chosen >= 256)).any()):
            raise ValueError('route out of range')
        index[row, :k], valid[row, :k] = chosen, True
    mask = torch.zeros((len(raw), 256), dtype=torch.long, device=raw.device)
    mask.scatter_add_(1, index, valid.long())
    if not torch.equal(mask.bool(), decision['selected_mask'].reshape(len(raw), 256).bool()):
        raise ValueError('sorted routes changed the E117 mask')
    return index, valid


def check_budget(args, started):
    if time.monotonic() - started > args.max_seconds:
        raise TimeoutError('bounded evaluation time exhausted')
    if (args.output / 'STOP_REQUEST').exists():
        raise InterruptedError('evaluation STOP_REQUEST')


def generation_worker(rank, gpu, workers, args, messages):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['USE_TF'] = '0'
    sys.path.insert(0, str(ROOT))
    began = time.monotonic()
    try:
        import numpy as np
        import torch
        from PIL import Image
        from safetensors import safe_open
        from experiments.maskgit_optimization_sweep_20260910.common import enable_h20
        enable_h20()
        from experiments.spatial_prefix_conditioning_20260910.model import create_model
        from h20.model import OfficialSamplingView
        sys.path.insert(0, '/home/heyefei/lichenge/1d-tokenizer')
        from modeling.maskgit import ImageBert
        sys.path.insert(0, str(ROOT.parent / 'dynamic'))
        from e117_ar_adapter import E117ARDecisionAdapter, load_e117
        from mot_mixture_decoder import MoTMixtureDecoder
        from e117_sparse_decoder import decode_e117_sparse_codes
        from experiments.maskgit_optimization_sweep_20260910.sampling_followup import (
            BASE, MARGIN4, generate, blend_features, decode, VARIANTS)

        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if torch.cuda.device_count() != 1:
            raise RuntimeError('one visible GPU required per generator')
        cores = {}
        for name, checkpoint in [('spatial', args.checkpoint), ('control', CONTROL)]:
            model = create_model(name, attention_implementation='sdpa')
            with safe_open(str(checkpoint), framework='pt', device='cpu') as state:
                prefix = args.state + '/'
                model.load_state_dict({k[len(prefix):]: state.get_tensor(k)
                                       for k in state.keys() if k.startswith(prefix)}, strict=True)
            cores[name] = model.cuda().eval().requires_grad_(False)
        decoder = MoTMixtureDecoder(MOT, 'cuda', state_key='model_ema')
        saved = torch.load(MOT, map_location='cpu', weights_only=True, mmap=True)['model_ema']
        loaded = decoder.model.state_dict()
        for key, value in saved.items():
            if key not in loaded or not torch.equal(value, loaded[key].cpu()):
                raise ValueError('MoT EMA tensor mismatch: ' + key)
        audit = dict(mot_ema_tensors_exact=len(saved), state=args.state, baseline_replay={})
        del saved, loaded
        student, metadata = load_e117(ROUTER, torch.device('cuda'), use_ema=True)
        if metadata['loaded_model_state'] != 'model_ema':
            raise RuntimeError('expected E117 EMA')
        del metadata
        adapter = E117ARDecisionAdapter(decoder.model, student).cuda().eval()
        view = OfficialSamplingView(cores['spatial']).eval()

        def uint8(image):
            if not torch.isfinite(image).all() or tuple(image.shape[1:]) != (3, 256, 256):
                raise ValueError('invalid decoded pixels')
            return ((image.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

        with torch.inference_mode():
            for start in range(rank * args.batch, args.n, workers * args.batch):
                check_budget(args, began)
                end = min(start + args.batch, args.n)
                ids = np.arange(start, end)
                labels = torch.as_tensor(ids % 1000, device='cuda')
                started = time.monotonic()
                torch.manual_seed(args.seed + 2 * start)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z1 = ImageBert.generate(view, labels, **ONE_D)
                if z1.shape != (end-start, 32) or not ((z1 >= 0) & (z1 < 4096)).all():
                    raise ValueError('invalid generated1D codes')
                decision = adapter(z1, return_1d_features=True)
                flags = ['source_image_used', 'f_2d_used', 'old_router_forwards',
                         'probe_reconstructions', 'batch_statistics_used', 'extra_route_fields_used']
                if any(decision[k] for k in flags):
                    raise RuntimeError('Router information boundary violated')
                index, valid = sorted_routes(decision)
                counts = valid.sum(-1)
                f1d = decision['f_1d']
                def provider(query):
                    if not torch.equal(query, z1):
                        raise ValueError('spatial features bound to wrong1D')
                    return f1d
                cores['spatial'].set_feature_provider(provider)
                images = {'base': uint8(decision['x_base'])}
                codes, sampling_seconds, decoding_seconds = {}, {}, {}
                for name, core, cfg in [('control8', cores['control'], BASE),
                                        ('spatial8', cores['spatial'], BASE),
                                        ('spatial_margin4', cores['spatial'], MARGIN4)]:
                    torch.manual_seed(args.seed + 2 * start + 1)
                    torch.cuda.synchronize()
                    tick = time.monotonic()
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        z2 = generate(core, z1, index, valid, labels, cfg)
                    after_rng = torch.cuda.get_rng_state()
                    torch.cuda.synchronize()
                    sampling_seconds[name] = time.monotonic() - tick
                    if start == 0 and cfg == BASE:
                        torch.manual_seed(args.seed + 1)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            original = core.generate_2d(z1, index, valid, labels, **TWO_D)
                        if not torch.equal(original, z2) or not torch.equal(after_rng, torch.cuda.get_rng_state()):
                            raise ValueError('original sampler token/RNG replay mismatch: ' + name)
                        audit['baseline_replay'][name] = True
                        torch.cuda.set_rng_state(after_rng)
                    tick = time.monotonic()
                    result = decode_e117_sparse_codes(decoder.model, f1d, z2, index, valid)
                    images[name] = uint8(result['image'])
                    torch.cuda.synchronize()
                    decoding_seconds[name] = time.monotonic() - tick
                    codes[name] = z2.cpu().numpy().astype(np.uint16)
                    if name == 'spatial_margin4':
                        tick = time.monotonic()
                        blended = blend_features(f1d, result['f_mix'], .5)
                        image_blend = decoder.model.llamagen_vq.decoder(blended)
                        images['spatial_margin4_blend050'] = uint8(image_blend)
                        torch.cuda.synchronize()
                        decoding_seconds['spatial_margin4_blend050'] = time.monotonic() - tick
                        if start == 0:
                            reference_cfg = next(v for v in VARIANTS if v.name == 'margin4_blend050')
                            reference = decode(decoder.model, f1d, z2, index, valid, reference_cfg)
                            if not torch.equal(reference, image_blend):
                                raise ValueError('shared decode optimization changed float pixels')
                            selection = decision['selected_mask'].reshape(len(ids), 1, 16, 16).bool().expand_as(f1d)
                            if not torch.equal(blended[~selection], f1d[~selection]):
                                raise ValueError('blend modified unselected cells')
                            audit.update(blend_pixels_exact=True, unrouted_features_exact=True,
                                         router_boundary_flags={k: decision[k] for k in flags})
                    del result
                if tuple(images) != NAMES:
                    raise ValueError('endpoint coverage mismatch')
                np.savez_compressed(args.output / f'codes_{start:05d}.npz', ids=ids, labels=ids % 1000,
                                    z1d=z1.cpu().numpy().astype(np.uint16), **codes,
                                    index=index.cpu().numpy().astype(np.uint16), valid=valid.cpu().numpy())
                if start == 0:
                    for name, array in images.items():
                        preview = np.concatenate([np.concatenate(list(array[i:i+4]), axis=1)
                                                  for i in (0, 4)], axis=0)
                        Image.fromarray(preview).save(args.output / f'{name}_first8.png')
                    save_json(args.output / 'generator_audit.json', audit)
                record = dict(rank=rank, start=start, count=len(ids), seconds=time.monotonic()-started,
                              sampling_seconds=sampling_seconds, decoding_seconds=decoding_seconds,
                              peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
                save_json(args.output / f'generator_{rank}.json', dict(status='running', pid=os.getpid(), **record))
                messages.put(('batch', ids, images, counts.cpu().numpy(), record))
                del z1, z2, decision, index, valid, images, f1d, blended, image_blend
        save_json(args.output / f'generator_{rank}.json', dict(status='complete', pid=os.getpid(), rank=rank))
        messages.put(('done', rank))
    except BaseException:
        error = traceback.format_exc()
        save_json(args.output / f'generator_{rank}_failure.json', dict(error=error, pid=os.getpid()))
        messages.put(('error', rank, error))
        raise


def run(args):
    import numpy as np
    if len(set(args.gpus)) != 2 or len(args.gpus) != 2:
        raise ValueError('exactly two distinct GPUs: one generator and one ADM scorer')
    if args.n < 8 or not 8 <= args.batch <= 32 or not 1 <= args.feature_batch <= 32:
        raise ValueError('invalid count/batch')
    if not args.smoke and args.n != 50000:
        raise ValueError('full FID run requires exactly 50000 samples')
    args.output = args.output.resolve()
    if args.output == RESULT_ROOT.resolve() or not args.output.is_relative_to(RESULT_ROOT.resolve()):
        raise ValueError('output must be a new child of the approved experiment result root')
    if args.smoke and args.n > 64:
        raise ValueError('smoke count capped at64')
    if not 60 <= args.max_seconds <= 7200:
        raise ValueError('explicit evaluation time cap60..7200 seconds required')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    assets = {str(p): digest(p) for p in (args.checkpoint, CONTROL, MOT, ROUTER, EVALUATOR, REFERENCE, GRAPH)}
    for path, expected in [(args.checkpoint, SPATIAL_SHA), (CONTROL, CONTROL_SHA), (MOT, MOT_SHA), (ROUTER, ROUTER_SHA)]:
        if assets[str(path)] != expected:
            raise ValueError('pinned immutable source mismatch: ' + str(path))
    for path in (args.checkpoint, CONTROL):
        meta = json.loads(path.with_suffix('.json').read_text())
        if meta['step'] != 13173 or meta['sha256'] != assets[str(path)]:
            raise ValueError('checkpoint metadata mismatch')
    sources = [Path(__file__), HERE/'common.py', HERE/'sampling.py', HERE/'sampling_followup.py',
               ROOT/'experiments/spatial_prefix_conditioning_20260910/model.py',
               ROOT/'delivery_h20_20260910/h20/model.py', ROOT/'delivery_h20_20260910/h20/base_model.py',
               ROOT/'e117_sparse_decoder.py', ROOT/'mot_mixture_decoder.py',
               ROOT.parent/'dynamic/e117_ar_adapter.py', ROOT.parent/'dynamic/e84_ar_adapter.py',
               ROOT.parent/'dynamic/one_d_deep_spatial_residual_e113.py',
               Path('/home/heyefei/lichenge/1d-tokenizer/modeling/maskgit.py')]
    source_hashes = {str(p): digest(p) for p in sources}
    if not args.smoke:
        if args.smoke_evidence is None:
            raise ValueError('full50k requires a passed same-code32-image smoke')
        smoke = json.loads((args.smoke_evidence / 'summary.json').read_text())
        smoke_manifest = json.loads((args.smoke_evidence / 'manifest.json').read_text())
        audit = json.loads((args.smoke_evidence / 'generator_audit.json').read_text())
        if (smoke['status'] != 'complete' or not smoke['smoke'] or smoke['n'] < 32
                or not smoke['complete_coverage'] or not smoke['gpu_features_verified']
                or smoke_manifest['source_hashes'] != source_hashes
                or smoke_manifest['assets'] != assets or smoke_manifest['batch'] != args.batch
                or smoke_manifest['state'] != args.state or smoke_manifest['seed'] != args.seed
                or audit['baseline_replay'] != {'control8': True, 'spatial8': True}
                or not audit['blend_pixels_exact'] or not audit['unrouted_features_exact']):
            raise ValueError('smoke evidence does not match the frozen50k protocol')
    manifest = dict(format='maskgit_optimization_five_paired_adm_v1', n=args.n, seed=args.seed,
                    state=args.state, batch=args.batch, feature_batch=args.feature_batch, gpus=args.gpus,
                    stage1=ONE_D, stage2=TWO_D, assets=assets, script_sha256=digest(__file__),
                    labels='global_sample_id % 1000', paired_endpoints=list(NAMES),
                    stage2_variants=dict(control8='original8', spatial8='original8',
                        spatial_margin4='margin_confidence_4steps_CFG4.5',
                        spatial_margin4_blend050='same_margin4_tokens_alpha0.5'),
                    stage1_source='spatial_raw13173', shared_1d_across_models=True,
                    comparison_scope='conditional2D interventions, not separate full1D pipelines',
                    source_hashes=source_hashes, max_seconds=args.max_seconds,
                    uint8_protocol='MoT round((clamp[-1,1]+1)*127.5)',
                    generated_prefix=True, real_images_used=False, generator_precision='bf16',
                    router_decoder_precision='fp32', feature_precision='fp32', tf32=False,
                    pid=os.getpid(), smoke=args.smoke, started_unix=time.time())
    save_json(args.output / 'manifest.json', manifest)
    save_json(args.output / 'progress.json', dict(status='loading', completed=0, n=args.n, pid=os.getpid()))
    ctx = mp.get_context('spawn')
    messages = ctx.Queue(maxsize=4)
    processes = [ctx.Process(target=generation_worker, args=(rank, gpu, len(args.gpus)-1, args, messages))
                 for rank, gpu in enumerate(args.gpus[:-1])]
    run_wandb = None
    try:
        for process in processes:
            process.start()
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus[-1])
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import wandb
        run_wandb = wandb.init(project='motar-maskgit', group='maskgit-optimization-free-fid',
                               name=args.output.name, mode='online', dir=str(args.output), config=manifest,
                               settings=wandb.Settings(disable_code=True, disable_git=True, save_code=False, console='off'))
        run_wandb.define_metric('images')
        run_wandb.define_metric('eval/*', step_metric='images')
        save_json(args.output / 'wandb_run.json', dict(id=run_wandb.id, url=run_wandb.url, mode='online'))
        spec = importlib.util.spec_from_file_location('registered_adm_eval50k', EVALUATOR)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.INCEPTION_V3_PATH = str(GRAPH)
        tf = module.tf
        tf.disable_eager_execution()
        if not tf.config.list_physical_devices('GPU'):
            raise RuntimeError('ADM GPU unavailable; refusing silent CPU fallback')
        tf.config.experimental.enable_tensor_float_32_execution(False)
        config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=8,
                                inter_op_parallelism_threads=2)
        config.gpu_options.allow_growth = True
        features = {(name, kind): np.lib.format.open_memmap(args.output/f'{name}_{kind}.npy',
                    mode='w+', dtype=np.float32, shape=(args.n, dim))
                    for name in NAMES for kind, dim in (('pool', 2048), ('spatial', 2023))}
        written = np.zeros(args.n, dtype=bool)
        route_counts = np.zeros(args.n, dtype=np.int64)
        timing = []
        done = set()
        traced_gpu = False
        with tf.Session(config=config) as session:
            evaluator = module.Evaluator(session, batch_size=args.feature_batch)
            while int(written.sum()) < args.n:
                check_budget(args, started)
                try:
                    message = messages.get(timeout=10)
                except queue.Empty:
                    failed = [(p.pid, p.exitcode) for p in processes if p.exitcode not in (None, 0)]
                    if failed or all(not p.is_alive() for p in processes):
                        raise RuntimeError(f'generators terminated before coverage: {failed}')
                    continue
                if message[0] == 'error':
                    raise RuntimeError(message[2])
                if message[0] == 'done':
                    done.add(message[1])
                    continue
                _, ids, images_by_name, counts, record = message
                if tuple(images_by_name) != NAMES:
                    raise ValueError('generator endpoints do not match locked protocol')
                if not np.isin(counts, [64, 128]).all():
                    raise ValueError('invalid route count packet')
                if ids.ndim != 1 or len(set(ids.tolist())) != len(ids) or (ids < 0).any() or (ids >= args.n).any() or written[ids].any():
                    raise ValueError('duplicate/out-of-bounds sample IDs')
                if not np.array_equal(ids, np.arange(record['start'], record['start']+len(ids))):
                    raise ValueError('incorrect global ID range')
                for name, images in images_by_name.items():
                    if images.shape != (len(ids), 256, 256, 3) or images.dtype != np.uint8:
                        raise ValueError('invalid image array')
                    for offset in range(0, len(ids), args.feature_batch):
                        end = min(offset + args.feature_batch, len(ids))
                        kwargs = {}
                        if not traced_gpu:
                            trace = tf.RunMetadata()
                            kwargs = dict(options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=trace)
                        values = session.run([evaluator.pool_features, evaluator.spatial_features],
                                  {evaluator.image_input: images[offset:end].astype(np.float32)}, **kwargs)
                        if not traced_gpu:
                            devices = [d.device for d in trace.step_stats.dev_stats
                                       if 'GPU' in d.device.upper() and any('conv' in n.node_name.lower() for n in d.node_stats)]
                            if not devices:
                                raise RuntimeError('Inception convolutions did not execute on GPU')
                            traced_gpu = True
                            save_json(args.output/'adm_gpu_trace.json', dict(convolution_devices=devices, tensorflow=tf.__version__))
                        for kind, value in zip(('pool', 'spatial'), values):
                            value = value.reshape(end-offset, -1)
                            if value.shape[1] != features[name, kind].shape[1] or not np.isfinite(value).all():
                                raise ValueError('invalid ADM features')
                            features[name, kind][ids[offset:end]] = value
                route_counts[ids] = counts
                written[ids] = True
                timing.append(record)
                completed = int(written.sum())
                for array in features.values():
                    array.flush()
                np.save(args.output/'written.npy', written)
                progress = dict(status='running', completed=completed, n=args.n,
                                seconds=time.monotonic()-started, mean_k=float(route_counts[written].mean()),
                                pid=os.getpid(), gpu_features_verified=traced_gpu)
                save_json(args.output/'progress.json', progress)
                print(json.dumps(progress), flush=True)
                run_wandb.log({'images':completed, 'eval/seconds':progress['seconds'], 'eval/mean_k':progress['mean_k']})
            for process in processes:
                process.join(timeout=5)
                if process.exitcode != 0:
                    raise RuntimeError(f'generator {process.pid} did not complete: {process.exitcode}')
            np.save(args.output/'route_counts.npy', route_counts)
            save_json(args.output/'timings.json', timing)
            metrics = {}
            if not args.smoke:
                save_json(args.output/'progress.json', dict(status='scoring', completed=args.n, n=args.n, pid=os.getpid()))
                with np.load(REFERENCE) as reference:
                    for name in NAMES:
                        check_budget(args, started)
                        metrics[name] = {}
                        for kind, metric, mu, sigma in [('pool','fid','mu','sigma'),('spatial','sfid','mu_s','sigma_s')]:
                            stats = evaluator.compute_statistics(features[name, kind])
                            value = float(stats.frechet_distance(module.FIDStatistics(reference[mu], reference[sigma])))
                            if not np.isfinite(value):
                                raise ValueError('nonfinite FID')
                            metrics[name][metric] = value
                            run_wandb.summary[f'{name}/{metric}'] = value
                        metrics[name]['inception_score'] = evaluator.compute_inception_score(features[name, 'pool'], split_size=5000)
                        save_json(args.output/'partial_metrics.json', metrics)
                        print(name, metrics[name], flush=True)
            if assets != {path: digest(path) for path in assets} or source_hashes != {path: digest(path) for path in source_hashes}:
                raise RuntimeError('input assets or evaluation script changed during run')
            summary = dict(status='complete', n=args.n, smoke=args.smoke, metrics=metrics,
                           checkpoint_sha256=assets[str(args.checkpoint)], control_sha256=assets[str(CONTROL)],
                           state=args.state, endpoints=list(NAMES), source_hashes_unchanged=True,
                           full_class_balance=bool(np.array_equal(np.bincount(np.arange(args.n)%1000, minlength=1000),
                                                                 np.full(1000,50))) if not args.smoke else None,
                           complete_coverage=bool(written.all()), generated_prefix=True,
                           gpu_features_verified=traced_gpu, runtime_seconds=time.monotonic()-started,
                           mean_k=float(route_counts.mean()), protocol=manifest['format'],
                           feature_hashes={f'{name}_{kind}':digest(args.output/f'{name}_{kind}.npy') for name,kind in features})
            save_json(args.output/'summary.json', summary)
            save_json(args.output/'progress.json', dict(status='complete', completed=args.n, n=args.n, pid=os.getpid()))
            run_wandb.summary['outcome'] = 'complete'
            run_wandb.finish()
            print(json.dumps(summary), flush=True)
    except BaseException:
        save_json(args.output/'failure.json', dict(error=traceback.format_exc(), pid=os.getpid()))
        if run_wandb is not None:
            run_wandb.finish(exit_code=1)
        raise
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--state', choices=('raw',), default='raw')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', type=lambda text: [int(x) for x in text.split(',')], default=[5,6])
    parser.add_argument('--n', type=int, default=50000)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--feature-batch', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261112)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--max-seconds', type=int, default=7200)
    parser.add_argument('--smoke-evidence', type=Path)
    run(parser.parse_args())
