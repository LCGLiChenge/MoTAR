"""Eight-H20 continuation of dense proxy MaskGIT; one rolling latest only."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('USE_TF', '0')
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from bert2d.paths import ASSETS, RESULT_ROOT, atomic_json, sha256, output_path
from bert2d.runtime import RankBucketSampler, OffsetSampler, save_latest
from bert2d.resume import load_resume, restore_rng, resumed_stream
from bert2d.features import PortableBaseFeatures
from h20.data import E117SparseCodeDataset, collate_e117_sparse
from h20_joint.assets import FrozenAssets
from experiments.feature_to_token_20260916.grid_maskgit import GridMaskGIT, GridObjective
from experiments.feature_to_token_20260916.probe import nearest
from experiments.feature_to_token_20260916.continue_grid_four import make_optimizer


def migration_rng(checkpoint_hash, rank, device):
    seed = int.from_bytes(hashlib.sha256(f'dense-grid-h20:{checkpoint_hash}:{rank}'.encode()).digest()[:8], 'big') % (2**63-1)
    return seed, torch.Generator().manual_seed(seed).get_state(), torch.Generator(device=device).manual_seed(seed).get_state()


def run(args):
    rank, world, local = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    assert world == 8
    source, out = args.resume.resolve(), output_path(args.output)
    meta = json.loads((source / 'latest.json').read_text())
    old, start = meta['config'], meta['step']
    assert old == json.loads((source / 'config.json').read_text())
    assert 6000 <= start < args.target <= 20000
    assert args.target in (6002, 8000, 12000, 16000, 20000)
    assert old['world'] in (4, 8) and old['accumulation'] == 1
    assert old['micro'] * old['world'] == 448
    for k, v in dict(style='dense_proxy_grid_maskgit_v1', global_batch=448, seed=0,
                     lr_new=1e-4, lr_pretrained=1e-5, warmup_updates=100).items():
        assert old[k] == v, k
    assert old['converter'] == {'mode': 'nearest'} and not old['unselected_gt_used']
    assert meta['round_trip_exact'] and sha256(source / 'latest.pt') == meta['sha256']
    for name, value in old['source_hashes'].items():
        assert sha256(Path(__file__).with_name(name)) == value, name
    if source == out:
        assert old['world'] == 8 and old['recompute_layers'] == 0
        assert old['total_updates'] == 20000 and Path(old['assets_root']).resolve() == ASSETS.resolve()
        config = old
    else:
        assert start == 6000 and old['world'] == 4 and not out.exists()
        config = dict(old, assets_root=str(ASSETS.resolve()), world=8, micro=56,
                      accumulation=1, total_updates=20000, recompute_layers=0,
                      resume_step=start, resume_from=str(source), resume_sha256=meta['sha256'],
                      additional_updates=14000, sampler='unchanged global448 K-buckets split into8x56',
                      execution_migration=dict(old_world=4, new_world=8, old_recompute=14,
                          new_recompute=0, bitwise_replay=False, global_batch_unchanged=True,
                          rng='retain ranks0..3; independent checkpoint-hash seeds for ranks4..7'),
                      checkpoint_policy='rolling latest only; paused eval at8k/12k/16k/20k')
        config['source_hashes'] = dict(old['source_hashes'], **{Path(__file__).name: sha256(Path(__file__))})
    torch.set_num_threads(4)
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    assert 'H20' in torch.cuda.get_device_name(device)
    torch.cuda.set_per_process_memory_fraction(.90, device)
    assert torch.cuda.mem_get_info(device)[0] >= 40 * 1024**3
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(minutes=15))
    if rank == 0 and source != out:
        out.mkdir(parents=True, exist_ok=False)
        atomic_json(out / 'config.json', config)
    dist.barrier()
    status = out / f'train_rank{rank}.json'
    begun = time.monotonic()
    atomic_json(status, dict(status='loading', pid=os.getpid(), step=start, target=args.target))
    core = GridMaskGIT(**old['model_config']).to(device).train()
    # H20 capacity removes activation recomputation, not any model layers.
    optimizer = make_optimizer(core, old)
    source_rank = rank % old['world']
    restored = load_resume(core, optimizer, source, meta, source_rank)
    seed = None
    if rank >= old['world']:
        seed, restored['rng_cpu'], restored['rng_cuda'] = migration_rng(meta['sha256'], rank, device)
    provider = PortableBaseFeatures(str(device), chunk=16)
    assert provider.audit['identities'] == old['feature_audit']['identities']
    temporary = FrozenAssets(ASSETS, 'cpu', chunk=1)
    with torch.no_grad():
        vq = temporary.shell.llamagen_vq
        emb = vq.quantize.get_codebook_entry(torch.arange(16384))
        book = vq.post_quant_conv(emb.T[None, :, None]).squeeze(0).squeeze(1).T.contiguous().to(device)
    del temporary
    wrapped = DistributedDataParallel(GridObjective(core), device_ids=[local])
    data = E117SparseCodeDataset(ASSETS / 'codes/train', ASSETS / 'routes/train', split='all')
    assert len(data) == 2562334
    sampler = OffsetSampler(RankBucketSampler(data, 56, world, rank, old['seed']))
    cursor = dict(meta['cursor'])  # old/new layouts each consume one global batch/update.
    loader = DataLoader(data, batch_sampler=sampler, collate_fn=collate_e117_sparse,
                        num_workers=2, pin_memory=True)
    stream = resumed_stream(loader, sampler, cursor)
    pending = next(stream)  # iterator creation must precede RNG restoration.
    ids = torch.stack([pending[0]['source_index'], pending[0]['augmentation']], 1)
    restore_rng(restored)
    audit = dict(**{k: v for k, v in restored.items() if not k.startswith('rng_')},
                 source_sha256=meta['sha256'], source_rng_rank=source_rank, migration_seed=seed,
                 rng_restored_exact=True, bitwise_replay=False, warmup_restarted=False,
                 mapped_cursor=cursor, first_pending_cursor=pending[1],
                 first_batch_source_aug_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                 device=torch.cuda.get_device_name(device), torch_version=torch.__version__,
                 cuda_version=torch.version.cuda, global_batch=448, micro=56, world=8)
    atomic_json(out / f'resume_audit_from{start}_rank{rank}.json', audit)
    ticks = []
    for step in range(start + 1, args.target + 1):
        tick = time.monotonic()
        if tick - begun > args.max_seconds:
            raise TimeoutError('authorized stage runtime exceeded')
        optimizer.zero_grad(set_to_none=True)
        cpu, cursor = pending if pending is not None else next(stream)
        pending = None
        batch = {k: v.to(device, non_blocking=True) for k, v in cpu.items()}
        features = provider(batch['z1d'])
        with torch.no_grad(): proxy = nearest(features, book)[0]
        indices, valid = batch['route_indices'], batch['route_valid']
        gt = torch.zeros(len(proxy), 256, dtype=torch.long, device=device)
        bi = torch.arange(len(proxy), device=device)[:, None].expand_as(indices)[valid]
        gt[bi, indices[valid]] = batch['z2d'][valid]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, row = wrapped(proxy, gt, batch['label'], indices, valid)
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(core.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        ticks.append(time.monotonic()-tick)
        if step <= start+2 or step % 20 == 0 or step == args.target:
            dist.all_reduce(row); row /= world
            record = dict(status='running', pid=os.getpid(), step=step, target=args.target, total=20000,
                loss=float(row[0]), masked_nll=float(row[1]), mask_ratio=float(row[2]),
                class_dropout=float(row[3]), grad_norm=float(norm), seconds=time.monotonic()-begun,
                seconds_per_step=sum(ticks[-20:])/len(ticks[-20:]),
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3,
                lr_new=optimizer.param_groups[0]['lr'], lr_pretrained=optimizer.param_groups[1]['lr'])
            atomic_json(status, record)
            if rank == 0:
                with (out / 'metrics.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
                print(json.dumps(record), flush=True)
        if step % 500 == 0 or step == args.target:
            rng = [None] * world
            dist.all_gather_object(rng, dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device)))
            if rank == 0: save_latest(out, core, None, optimizer, rng, dict(step=step, config=config, cursor=cursor))
            dist.barrier()
        del features, loss, row
    provider.assert_frozen()
    atomic_json(status, dict(status='complete', pid=os.getpid(), step=args.target,
                            seconds=time.monotonic()-begun, peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3))
    dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--target', type=int, required=True)
    p.add_argument('--max-seconds', type=int, default=14400)
    a = p.parse_args()
    try: run(a)
    except BaseException as exc:
        if a.output.exists(): atomic_json(a.output / f'failure_rank{os.getenv("RANK", "0")}.json', dict(error=repr(exc), pid=os.getpid()))
        raise
