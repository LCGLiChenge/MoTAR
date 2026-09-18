"""Collect no-xbase E117 budget scores on disjoint train or generated cohorts.

Only the 32-token 1D generator and frozen Router are run. The score includes
E117's exact float64 tie adjustment, so a fitted threshold is deployable.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for vendor in (ROOT / "third_party/E117", ROOT / "delivery_h20_20260910/third_party/E117"):
    if (vendor / "e117_ar_adapter.py").exists():
        sys.path.insert(0, str(vendor))
        break
os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch
import torch.distributed as dist

from bert2d.paths import ASSETS, ONE_D, atomic_json, output_path, sha256
from bert2d.eval import ImageBert, OfficialSamplingView
from h20.assets import official_model
from h20_joint.assets import FrozenAssets
from e117_ar_adapter import K_THRESHOLD, tie_adjusted_score_e117
from experiments.feature_to_token_20260916.e117_no_xbase import install_no_xbase
from experiments.feature_to_token_20260916.halton_parallel_eval import worker_batches


def score_codes(assets: FrozenAssets, codes: torch.Tensor) -> np.ndarray:
    scores = []
    for part in codes.split(4):
        quantized = assets.adapter._quantized_from_codes(part)
        prefix, f1d, placeholder = assets.adapter._decode_1d_bundle(quantized)
        details = assets.adapter.student.forward_details(part, prefix, f1d, placeholder)
        adjusted = tie_adjusted_score_e117(details["k_score"], details["parent_grid"])
        scores.append(adjusted.cpu().numpy())
    return np.concatenate(scores)


def main(args: argparse.Namespace) -> None:
    rank, world, local = (int(os.environ[name]) for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if world != 8 or args.n < 8 or args.batch < 1:
        raise ValueError("expected eight GPUs and a nonempty cohort")
    torch.cuda.set_device(local)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl")
    try:
        out = output_path(args.output)
        if rank == 0:
            out.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        device = torch.device("cuda", local)
        assets = FrozenAssets(ASSETS, device=device, chunk=4)
        audit = install_no_xbase(assets, args.router_proxy_checkpoint)
        one_d_audit = None
        if args.mode == "packed_train":
            packed = ASSETS / "codes/train"
            meta = json.loads((packed / "meta.json").read_text())
            codes = np.load(packed / "titok_codes.npy", mmap_mode="r", allow_pickle=False)
            if not meta.get("completed") or codes.shape != (1281167, 2, 32):
                raise RuntimeError("full ImageNet train cache is incomplete")
            rng = np.random.default_rng(args.seed)
            chosen = rng.choice(len(codes), size=args.n, replace=False)
            local_ids = chosen[rank::world]
            batches = [local_ids[i:i+args.batch] for i in range(0, len(local_ids), args.batch)]
        elif args.mode == "generated":
            if args.n not in (5000, 50000) or args.batch != 8:
                raise ValueError("generated cohort uses the FID eight-image batch plan")
            one_d, one_d_audit = official_model(ASSETS)
            one_d = one_d.to(device).eval().requires_grad_(False)
            view = OfficialSamplingView(one_d).eval()
            batches = worker_batches(args.n, world, rank)
        else:
            raise ValueError("unknown mode")
        values = []
        completed = 0
        start = time.monotonic()
        with torch.inference_mode():
            for batch_index, ids in enumerate(batches):
                if args.mode == "packed_train":
                    z1 = torch.from_numpy(np.array(codes[ids, 0], dtype=np.int64)).to(device)
                else:
                    labels = torch.as_tensor(ids % 1000, device=device)
                    torch.manual_seed(args.seed + 2 * int(ids[0]))
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        z1 = ImageBert.generate(view, condition=labels, **ONE_D)
                values.append(score_codes(assets, z1))
                completed += len(ids)
                if batch_index % 100 == 0 or batch_index + 1 == len(batches):
                    atomic_json(out / f"status_rank{rank}.json", dict(
                        status="running", completed=completed,
                        total=sum(len(batch) for batch in batches), seconds=time.monotonic()-start))
        score = np.concatenate(values).astype(np.float64, copy=False)
        np.save(out / f"score_rank{rank}.npy", score)
        atomic_json(out / f"status_rank{rank}.json", dict(
            status="complete", completed=completed, total=completed,
            seconds=time.monotonic()-start))
        dist.barrier()
        if rank == 0:
            all_scores = np.concatenate([np.load(out / f"score_rank{i}.npy", allow_pickle=False)
                                         for i in range(world)])
            if len(all_scores) != args.n or not np.isfinite(all_scores).all():
                raise RuntimeError("incomplete or nonfinite Router score collection")
            ordered = np.sort(all_scores)
            def threshold_for_mean(target: float) -> float:
                fraction = (target - 64.) / 64.
                high = round(fraction * args.n)
                if not 0 < high < args.n:
                    raise ValueError("target budget outside 64..128")
                lower, upper = ordered[-high-1:-high+1]
                if lower == upper:
                    raise RuntimeError("Router score tie prevents exact budget target")
                return float((lower + upper) * .5)
            summary = dict(status="complete", mode=args.mode, n=args.n,
                           seed=args.seed, workers=world, original_threshold=K_THRESHOLD,
                           original_k128=int((all_scores > K_THRESHOLD).sum()),
                           original_mean_2d=float(64 + 64 * (all_scores > K_THRESHOLD).mean()),
                           thresholds={str(target): threshold_for_mean(target)
                                       for target in (88, 96, 97, 98)},
                           router_proxy_sha256=audit["proxy_sha256"],
                           one_d_audit=one_d_audit,
                           source_sha256=sha256(Path(__file__)))
            atomic_json(out / "summary.json", summary)
            print(json.dumps(summary), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("packed_train", "generated"), required=True)
    parser.add_argument("--router-proxy-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort-size", dest="n", type=int, default=50000)
    parser.add_argument("--cohort-seed", dest="seed", type=int, default=20260918)
    parser.add_argument("--batch", type=int, default=64)
    main(parser.parse_args())
