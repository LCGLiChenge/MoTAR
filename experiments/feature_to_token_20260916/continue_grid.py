"""Stateful, same-recipe continuation: 2000->4000->6000 dense-grid MaskGIT."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
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
from bert2d.paths import ASSETS, atomic_json, sha256
from bert2d.features import PortableBaseFeatures
from bert2d.runtime import OffsetSampler, save_latest
from bert2d.resume import load_resume, restore_rng, resumed_stream
from bert_continue.data import FoldedSampler
from h20.data import E117SparseCodeDataset, collate_e117_sparse
from h20_joint.assets import FrozenAssets
from experiments.feature_to_token_20260916.grid_maskgit import GridMaskGIT, GridObjective
from experiments.feature_to_token_20260916.probe import nearest


def make_optimizer(core, config):
    fresh = set(config['init_audit']['fresh_parameters'])
    return torch.optim.AdamW([
        {'params': [p for n, p in core.named_parameters() if n in fresh], 'lr': 1e-4},
        {'params': [p for n, p in core.named_parameters() if n not in fresh], 'lr': 1e-5}],
        betas=(.9, .96), eps=1e-8, weight_decay=.03)


def run(args):
    rank, world, local = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    assert world == 2
    source = args.resume.resolve()
    meta = json.loads((source / 'latest.json').read_text())
    old = meta['config']; start = meta['step']
    assert old == json.loads((source / 'config.json').read_text())
    assert (start, args.target) in ((2000, 2002), (2000, 4000), (4000, 6000))
    expected = dict(style='dense_proxy_grid_maskgit_v1', global_batch=448, micro=112,
                    world=2, accumulation=2, seed=0, recompute_layers=14,
                    lr_new=1e-4, lr_pretrained=1e-5, warmup_updates=100)
    for key, value in expected.items():
        assert old[key] == value, key
    assert old['converter'] == {'mode': 'nearest'} and not old['unselected_gt_used']
    assert Path(old['assets_root']).resolve() == ASSETS.resolve()
    assert meta['round_trip_exact'] and sha256(source / 'latest.pt') == meta['sha256']
    for name, expected_hash in old['source_hashes'].items():
        assert sha256(Path(__file__).with_name(name)) == expected_hash, name
    data_root = Path('/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results').resolve()
    out = args.output.resolve()
    assert out != data_root and out.is_relative_to(data_root) and out != source
    torch.set_num_threads(4); torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.cuda.set_per_process_memory_fraction(.90, device)
    assert torch.cuda.mem_get_info(device)[0] >= 25 * 1024**3
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(minutes=10))
    if rank == 0: out.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    status = out / f'train_rank{rank}.json'; started = time.monotonic()
    atomic_json(status, dict(status='loading', pid=os.getpid(), step=start))
    core = GridMaskGIT(**old['model_config']).to(device).train()
    core.enable_recompute(old['recompute_layers'])
    optimizer = make_optimizer(core, old)
    restored = load_resume(core, optimizer, source, meta, rank)
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
    sampler = OffsetSampler(FoldedSampler(data, rank, old['seed'], micro=112, world=2))
    cursor = dict(meta['cursor'])
    assert cursor['next_microbatch_offset'] % 2 == 0
    loader = DataLoader(data, batch_sampler=sampler, collate_fn=collate_e117_sparse,
                        num_workers=4, pin_memory=True)
    stream = resumed_stream(loader, sampler, cursor)
    # Prime the replacement iterator BEFORE restoring RNG. Creating a DataLoader
    # iterator draws a worker seed from CPU RNG; the uninterrupted run already
    # has its iterator. Later epoch transitions retain the original RNG behavior.
    pending = next(stream)
    first_batch, next_cursor = pending
    first_ids = torch.stack([first_batch['source_index'], first_batch['augmentation']], 1)
    import hashlib
    first_hash = hashlib.sha256(first_ids.numpy().tobytes()).hexdigest()
    config = dict(old, total_updates=args.target, resume_from=str(source), resume_step=start,
                  resume_sha256=meta['sha256'], additional_updates=args.target-start)
    config['source_hashes'] = dict(old['source_hashes'], **{Path(__file__).name: sha256(Path(__file__))})
    if rank == 0: atomic_json(out / 'config.json', config)
    restore_rng(restored)
    atomic_json(out / f'resume_audit_rank{rank}.json', dict(
        **{k: v for k, v in restored.items() if not k.startswith('rng_')},
        source_sha256=meta['sha256'], first_batch_source_aug_sha256=first_hash,
        first_pending_cursor=next_cursor, rng_restored_exact=True, execution_layout_changed=False,
        iterator_primed_before_rng_restore=True, warmup_restarted=False))
    ticks = []
    for step in range(start + 1, args.target + 1):
        tick = time.monotonic()
        if tick - started > 5400: raise TimeoutError('90-minute stage bound')
        optimizer.zero_grad(set_to_none=True)
        stats = torch.zeros(4, device=device)
        optimizer.param_groups[0]['lr'] = 1e-4 * min(step / 100, 1.)
        optimizer.param_groups[1]['lr'] = 1e-5 * min(step / 100, 1.)
        for micro_step in range(2):
            if pending is not None:
                cpu, cursor = pending; pending = None
            else:
                cpu, cursor = next(stream)
            batch = {k: v.to(device, non_blocking=True) for k, v in cpu.items()}
            features = provider(batch['z1d'])
            with torch.no_grad(): proxy = nearest(features, book)[0]
            indices, valid = batch['route_indices'], batch['route_valid']
            gt = torch.zeros(len(proxy), 256, dtype=torch.long, device=device)
            bi = torch.arange(len(proxy), device=device)[:, None].expand_as(indices)[valid]
            gt[bi, indices[valid]] = batch['z2d'][valid]
            with (wrapped.no_sync() if micro_step == 0 else nullcontext()):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, row = wrapped(proxy, gt, batch['label'], indices, valid)
                assert torch.isfinite(loss)
                (loss / 2).backward()
            stats += row / 2
            del features, loss, row
        norm = torch.nn.utils.clip_grad_norm_(core.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); ticks.append(time.monotonic() - tick)
        if step <= start + 2 or step % 20 == 0 or step == args.target:
            dist.all_reduce(stats); stats /= world
            record = dict(status='running', pid=os.getpid(), step=step, total=args.target,
                          loss=float(stats[0]), masked_nll=float(stats[1]), mask_ratio=float(stats[2]),
                          class_dropout=float(stats[3]), grad_norm=float(norm),
                          seconds=time.monotonic()-started, seconds_per_step=sum(ticks[-20:])/len(ticks[-20:]),
                          peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3,
                          lr_new=optimizer.param_groups[0]['lr'], lr_pretrained=optimizer.param_groups[1]['lr'])
            atomic_json(status, record)
            if rank == 0:
                with (out / 'metrics.jsonl').open('a') as handle: handle.write(json.dumps(record)+'\n')
                print(json.dumps(record), flush=True)
        if step % 500 == 0 or step == args.target:
            all_rng = [None] * world
            dist.all_gather_object(all_rng, {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device)})
            if rank == 0:
                save_latest(out, core, None, optimizer, all_rng, dict(step=step, config=config, cursor=cursor))
            dist.barrier()
    provider.assert_frozen()
    atomic_json(status, dict(status='complete', pid=os.getpid(), step=args.target,
                            seconds=time.monotonic()-started,
                            peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3))
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', type=int, required=True)
    args = parser.parse_args()
    try: run(args)
    except BaseException as exc:
        if args.output.exists():
            atomic_json(args.output / f'failure_rank{os.getenv("RANK", "0")}.json', dict(error=repr(exc), pid=os.getpid()))
        raise
