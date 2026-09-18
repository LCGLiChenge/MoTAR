"""Cache paired no-xbase E117 routes and f1d-mapper proxy IDs for MaskGIT.

Run with torchrun. Ranks write disjoint flat ImageNet-train source/view IDs.
The cache is resumable and becomes trainable only after full coverage audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("USE_TF", "0")

from bert2d.paths import ASSETS, atomic_json, output_path, sha256
from h20_joint.assets import FrozenAssets
from experiments.feature_to_token_20260916.converter_codebook import CodebookFeatureTokenConverter
from experiments.feature_to_token_20260916.e117_no_xbase import install_no_xbase


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def arrays(out: Path, sources: int, views: int, create: bool):
    specs = dict(route_k=((sources, views), np.uint8),
                 route_indices=((sources, views, 128), np.uint8),
                 proxy_codes=((sources * views, 256), np.uint16),
                 written=((sources, views), np.bool_))
    result = {}
    for name, (shape, dtype) in specs.items():
        path = out / f"{name}.npy"
        if create:
            value = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
            value[:] = 0
            value.flush()
        else:
            value = np.load(path, mmap_mode="r+", allow_pickle=False)
            require(value.shape == shape and value.dtype == dtype, f"bad {name} cache array")
        result[name] = value
    return result


def main(args: argparse.Namespace) -> None:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    require(torch.cuda.is_available(), "CUDA is required")
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl")
    try:
        torch.set_num_threads(4)
        out = output_path(args.output)
        root = args.assets_root.resolve()
        packed = root / "codes/train"
        meta = json.loads((packed / "meta.json").read_text())
        require(meta.get("completed") and int(meta.get("format_version", -1)) == 1,
                "packed training codes are incomplete")
        z1 = np.load(packed / "titok_codes.npy", mmap_mode="r", allow_pickle=False)
        z2 = np.load(packed / "llamagen_codes.npy", mmap_mode="r", allow_pickle=False)
        require(z1.shape == (1_281_167, 2, 32) and z2.shape == (1_281_167, 2, 256),
                "unexpected full ImageNet train cache shape")
        sources = len(z1) if args.limit_sources == 0 else min(args.limit_sources, len(z1))
        require(sources > 0 and args.batch > 0 and args.flush_every > 0, "positive cache size/batch/flush required")
        views = z1.shape[1]
        source_ids = np.arange(sources, dtype=np.int64)
        ids_sha = hashlib.sha256(source_ids.astype("<i8").tobytes()).hexdigest()
        router_sha = sha256(args.router_proxy_checkpoint)
        mapper_sha = sha256(args.mapper_checkpoint)
        mapper_state = torch.load(args.mapper_checkpoint, map_location="cpu", weights_only=True)
        require(mapper_state.get("format") == "feature_proxy_token_converter_codebook_v1",
                "expected the trained 8D-codebook mapper")
        route_meta = dict(format="e117_sparse_route_cache_v1", completed=False,
                          num_samples=sources, num_aug=views, source_start=0,
                          source_stop=len(z1), source_selection="sequential", source_seed=0,
                          source_indices_sha256=ids_sha, packed_root=str(packed),
                          e117_checkpoint=str(root / "router/e117.pt"),
                          e117_checkpoint_sha256=sha256(root / "router/e117.pt"),
                          router_mode="no-xbase", router_proxy_sha256=router_sha,
                          mapper_sha256=mapper_sha, k_candidates=[64, 128], max_k=128)
        proxy_config = dict(format="f1d_mapper_proxy_v1", shape=[sources * views, 256],
                            dtype="uint16", flat_layout="source_index*num_aug+augmentation",
                            mapper_checkpoint_sha256=mapper_sha,
                            router_proxy_sha256=router_sha,
                            packed_meta_sha256=sha256(packed / "meta.json"))
        if rank == 0:
            if out.exists():
                require(args.resume, "existing cache requires --resume")
                require(json.loads((out / "config.json").read_text()) == proxy_config,
                        "resume proxy config mismatch")
                old = json.loads((out / "meta.json").read_text())
                require(all(old.get(k) == v for k, v in route_meta.items() if k != "completed"),
                        "resume route metadata mismatch")
                require(np.array_equal(np.load(out / "source_indices.npy"), source_ids),
                        "resume source order mismatch")
                arrays(out, sources, views, create=False)
            else:
                out.mkdir(parents=True)
                arrays(out, sources, views, create=True)
                np.save(out / "source_indices.npy", source_ids)
                atomic_json(out / "meta.json", route_meta)
                atomic_json(out / "config.json", proxy_config)
        if world > 1:
            dist.barrier()
        cache = arrays(out, sources, views, create=False)
        pending = np.flatnonzero(~np.asarray(cache["written"]).reshape(-1))
        local_ids = pending[pending % world == rank]
        device = torch.device("cuda", local)
        assets = FrozenAssets(root, device=device, chunk=min(args.batch, 4))
        router_audit = install_no_xbase(assets, args.router_proxy_checkpoint)
        mapper = CodebookFeatureTokenConverter(**mapper_state["model_config"]).to(device)
        native = assets.shell.llamagen_vq.quantize.get_codebook_entry(
            torch.arange(16384, device=device))
        mapper.initialize_codebook(native, assets.shell.llamagen_vq.post_quant_conv)
        mapper.load_state_dict(mapper_state["model"], strict=True)
        mapper.eval().requires_grad_(False)
        require(mapper_state["train_config"]["frozen_audit"]["identities"] == assets.identities,
                "mapper was trained with different frozen assets")
        start = time.monotonic()
        uncommitted: list[np.ndarray] = []
        with torch.inference_mode():
            for offset in range(0, len(local_ids), args.batch):
                ids = local_ids[offset:offset + args.batch]
                rows, aug = np.divmod(ids, views)
                codes = torch.from_numpy(np.array(z1[rows, aug], dtype=np.int64)).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    bundle = assets.bundle(codes)
                    proxy = torch.cat([mapper.tokens(part) for part in bundle["base"].split(4)])
                indices = bundle["index"].cpu().numpy()
                valid = bundle["valid"].cpu().numpy()
                counts = valid.sum(axis=1)
                require(np.all((counts == 64) | (counts == 128)), "invalid Router budget")
                require(proxy.shape == (len(ids), 256), "invalid mapper output shape")
                require(bool(((proxy >= 0) & (proxy < 16384)).all()), "invalid proxy ID")
                cache["proxy_codes"][ids] = proxy.cpu().numpy().astype(np.uint16)
                for i, (row, view) in enumerate(zip(rows, aug)):
                    k = int(counts[i])
                    chosen = indices[i, :k]
                    require(np.all(chosen[:-1] < chosen[1:]) and 0 <= chosen[0]
                            and chosen[-1] < 256, "invalid sorted route")
                    cache["route_k"][row, view] = k
                    cache["route_indices"][row, view, :] = 0
                    cache["route_indices"][row, view, :k] = chosen.astype(np.uint8)
                uncommitted.append(ids)
                if ((offset // args.batch + 1) % args.flush_every == 0
                    or offset + len(ids) == len(local_ids)):
                    for name in ("route_k", "route_indices", "proxy_codes"):
                        cache[name].flush()
                    cache["written"].reshape(-1)[np.concatenate(uncommitted)] = True
                    cache["written"].flush()
                    uncommitted.clear()
                if offset == 0 or (offset // args.batch + 1) % args.log_every == 0:
                    atomic_json(out / f"status_rank{rank}.json",
                                dict(status="running", completed=min(offset + len(ids), len(local_ids)),
                                     total=len(local_ids), seconds=time.monotonic() - start,
                                     peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3))
        atomic_json(out / f"status_rank{rank}.json",
                    dict(status="complete", completed=len(local_ids), total=len(local_ids),
                         seconds=time.monotonic() - start))
        if world > 1:
            dist.barrier()
        if rank == 0:
            require(bool(np.asarray(cache["written"]).all()), "cache coverage incomplete")
            counts = np.asarray(cache["route_k"])
            require(bool(np.isin(counts, (64, 128)).all()), "invalid route K in completed cache")
            require(bool((np.asarray(cache["proxy_codes"]) < 16384).all()), "invalid proxy in completed cache")
            route_meta["completed"] = True
            route_meta["writer_world_size"] = world
            route_meta["summary"] = dict(count=int(counts.size), mean_k=float(counts.mean()),
                                         k64=int((counts == 64).sum()), k128=int((counts == 128).sum()))
            atomic_json(out / "meta.json", route_meta)
            summary = dict(status="complete", complete_coverage=True,
                           total=int(sources * views),
                           proxy_codes_sha256=sha256(out / "proxy_codes.npy"),
                           router=router_audit, mapper_sha256=mapper_sha,
                           route_summary=route_meta["summary"])
            atomic_json(out / "summary.json", summary)
            print(json.dumps(summary), flush=True)
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets-root", type=Path, default=ASSETS)
    p.add_argument("--router-proxy-checkpoint", type=Path, required=True)
    p.add_argument("--mapper-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit-sources", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--flush-every", type=int, default=64)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    main(p.parse_args())
