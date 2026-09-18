"""Count deployed calibrated Router selections on a generated 1D cohort."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch
import torch.distributed as dist

from bert2d.paths import ASSETS, ONE_D, atomic_json, output_path, sha256
from bert2d.eval import ImageBert, OfficialSamplingView
from h20.assets import official_model
from h20_joint.assets import FrozenAssets
from experiments.feature_to_token_20260916.e117_no_xbase import install_no_xbase
from experiments.feature_to_token_20260916.halton_parallel_eval import worker_batches


def main(args: argparse.Namespace) -> None:
    rank, world, local = (int(os.environ[key]) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if world not in (4, 8) or args.n not in (5000, 50000):
        raise ValueError("expected four or eight workers and a 5k or 50k generated cohort")
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
        assets = FrozenAssets(ASSETS, device=device, chunk=4, keep_encoder=True)
        audit = install_no_xbase(assets, args.router_proxy_checkpoint)
        if "budget_threshold" not in audit:
            raise RuntimeError("expected a calibrated no-xbase Router")
        one_d, one_d_audit = official_model(ASSETS)
        one_d = one_d.to(device).eval().requires_grad_(False)
        view = OfficialSamplingView(one_d).eval()
        batches = worker_batches(args.n, world, rank)
        expected = None
        if args.expected_score_root is not None:
            meta = json.loads((args.expected_score_root / "summary.json").read_text())
            if meta["n"] != args.n or meta["seed"] != args.seed or meta["mode"] != "generated":
                raise RuntimeError("expected score cohort differs")
            expected = np.load(args.expected_score_root / f"score_rank{rank}.npy", allow_pickle=False)
        done = k64 = k128 = 0
        start = time.monotonic()
        with torch.inference_mode():
            for index, ids in enumerate(batches):
                labels = torch.as_tensor(ids % 1000, device=device)
                torch.manual_seed(args.seed + 2 * int(ids[0]))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z1 = ImageBert.generate(view, condition=labels, **ONE_D)
                bundle = assets.bundle(z1)
                counts = bundle["valid"].sum(1).cpu().numpy()
                if not np.isin(counts, (64, 128)).all():
                    raise RuntimeError("invalid deployed Router K")
                if expected is not None:
                    decisions = expected[done:done + len(ids)] > audit["budget_threshold"]
                    if not np.array_equal(decisions, counts == 128):
                        raise RuntimeError("deployed K differs from the calibrated score threshold")
                done += len(ids)
                k64 += int((counts == 64).sum())
                k128 += int((counts == 128).sum())
                if index % 100 == 0 or index + 1 == len(batches):
                    atomic_json(out / f"status_rank{rank}.json", dict(
                        status="running", completed=done,
                        total=sum(len(batch) for batch in batches),
                        k64=k64, k128=k128, seconds=time.monotonic() - start))
        if expected is not None and done != len(expected):
            raise RuntimeError("score cohort coverage differs")
        totals = torch.tensor([done, k64, k128, int(expected is not None)],
                              device=device, dtype=torch.int64)
        dist.all_reduce(totals)
        total, all64, all128, verified = map(int, totals.cpu().tolist())
        if total != args.n or all64 + all128 != args.n:
            raise RuntimeError("incomplete token count")
        atomic_json(out / f"status_rank{rank}.json", dict(
            status="complete", completed=done, total=done, k64=k64, k128=k128,
            seconds=time.monotonic() - start))
        if rank == 0:
            mean_2d = (64 * all64 + 128 * all128) / total
            atomic_json(out / "summary.json", dict(
                status="complete", n=total, seed=args.seed,
                k64=all64, k128=all128, mean_1d=32,
                mean_generated_2d=mean_2d, mean_generated_total=32 + mean_2d,
                expected_score_ranks_verified=verified,
                router_proxy_sha256=audit["proxy_sha256"],
                budget_threshold=audit["budget_threshold"],
                one_d_audit=one_d_audit, source_sha256=sha256(Path(__file__))))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-proxy-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-score-root", type=Path)
    parser.add_argument("--cohort-size", dest="n", type=int, default=50000)
    parser.add_argument("--cohort-seed", dest="seed", type=int, default=20260914)
    main(parser.parse_args())
