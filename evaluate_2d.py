#!/usr/bin/env python3
"""Fixed-sample, fixed-order teacher-forced 2D diagnostics for MoTAR."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors.torch import load_file

from motar.checkpoint import validate_checkpoint as validate_baseline_checkpoint
from motar.checkpoint_disjoint import validate_checkpoint as validate_disjoint_checkpoint
from motar.model import TiTokLlamaGenUnifiedAR


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    return rank, world_size, torch.device("cuda", local_rank)


def fixed_order(sample_id: int, block_size: int, seed: int) -> np.ndarray:
    """Return an order independent of batch size and distributed world size."""

    rng = np.random.default_rng(np.random.SeedSequence([seed, sample_id]))
    return rng.permutation(block_size).astype(np.int64, copy=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/h200_8gpu_150epoch_disjoint.yaml",
    )
    parser.add_argument(
        "--checkpoint-kind",
        choices=("baseline", "disjoint"),
        required=True,
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--packed-code-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-samples", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--augmentation", type=int, default=0)
    parser.add_argument("--order-seed", type=int, default=20260828)
    parser.add_argument("--ar-dtype", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    rank, world_size, device = init_distributed()
    config = OmegaConf.load(args.config)
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    validators = {
        "baseline": validate_baseline_checkpoint,
        "disjoint": validate_disjoint_checkpoint,
    }
    metadata = validators[args.checkpoint_kind](checkpoint_dir)

    model = TiTokLlamaGenUnifiedAR(
        **OmegaConf.to_container(config.model, resolve=True)
    )
    state = load_file(str(checkpoint_dir / "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)
    dtype = torch.bfloat16 if args.ar_dtype == "bf16" else torch.float32
    model = model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    model.grad_checkpointing = False

    root = Path(args.packed_code_root).expanduser().resolve()
    z1d_all = np.load(root / "titok_codes.npy", mmap_mode="r")
    z2d_all = np.load(root / "llamagen_codes.npy", mmap_mode="r")
    labels_all = np.load(root / "labels.npy", mmap_mode="r")
    count = min(int(args.num_samples), int(z1d_all.shape[0]))
    if not 0 <= args.augmentation < z1d_all.shape[1]:
        raise ValueError("augmentation index is out of range")
    if z2d_all.shape[0] != z1d_all.shape[0] or labels_all.shape[0] != z1d_all.shape[0]:
        raise ValueError("packed arrays have inconsistent sample counts")

    positions = int(config.model.block_size)
    step_ce_sum = torch.zeros(positions, dtype=torch.float64, device=device)
    step_top1_sum = torch.zeros_like(step_ce_sum)
    step_top5_sum = torch.zeros_like(step_ce_sum)
    spatial_ce_sum = torch.zeros_like(step_ce_sum)
    spatial_top1_sum = torch.zeros_like(step_ce_sum)
    spatial_top5_sum = torch.zeros_like(step_ce_sum)
    local_count = 0
    local_ids = np.arange(rank, count, world_size, dtype=np.int64)

    for start in range(0, len(local_ids), args.batch_size):
        indices = local_ids[start : start + args.batch_size]
        z1d = torch.from_numpy(
            np.array(z1d_all[indices, args.augmentation], dtype=np.int64, copy=True)
        ).to(device)
        z2d = torch.from_numpy(
            np.array(z2d_all[indices, args.augmentation], dtype=np.int64, copy=True)
        ).to(device)
        labels = torch.from_numpy(
            np.array(labels_all[indices], dtype=np.int64, copy=True)
        ).to(device)
        orders_np = np.stack(
            [fixed_order(int(sample_id), positions, args.order_seed) for sample_id in indices]
        )
        orders = torch.from_numpy(orders_np).to(device)
        targets = torch.gather(z2d, 1, orders)

        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16
        ):
            logits = model(
                z1d=z1d,
                z2d=z2d,
                cond_idx=labels,
                token_order=orders,
                targets_1d=z1d,
                targets_2d=z2d,
            )["logits_2d"]

        ce = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
        top1 = logits.argmax(dim=-1).eq(targets)
        top5 = logits.topk(5, dim=-1).indices.eq(targets.unsqueeze(-1)).any(dim=-1)
        step_ce_sum += ce.double().sum(dim=0)
        step_top1_sum += top1.double().sum(dim=0)
        step_top5_sum += top5.double().sum(dim=0)

        spatial_ce_sum.scatter_add_(0, orders.reshape(-1), ce.double().reshape(-1))
        spatial_top1_sum.scatter_add_(0, orders.reshape(-1), top1.double().reshape(-1))
        spatial_top5_sum.scatter_add_(0, orders.reshape(-1), top5.double().reshape(-1))
        local_count += len(indices)

    count_tensor = torch.tensor(local_count, dtype=torch.int64, device=device)
    if world_size > 1:
        for tensor in (
            step_ce_sum,
            step_top1_sum,
            step_top5_sum,
            spatial_ce_sum,
            spatial_top1_sum,
            spatial_top5_sum,
            count_tensor,
        ):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        total = int(count_tensor.item())
        if total != count:
            raise RuntimeError(f"distributed sample count mismatch: {total} != {count}")
        step_ce = (step_ce_sum / total).cpu().tolist()
        step_top1 = (step_top1_sum / total).cpu().tolist()
        step_top5 = (step_top5_sum / total).cpu().tolist()
        spatial_ce = (spatial_ce_sum / total).cpu().tolist()
        spatial_top1 = (spatial_top1_sum / total).cpu().tolist()
        spatial_top5 = (spatial_top5_sum / total).cpu().tolist()
        mean_ce = sum(step_ce) / positions
        ranges = tuple((start, start + 32) for start in range(0, positions, 32))
        report = {
            "checkpoint_kind": args.checkpoint_kind,
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_metadata": metadata,
            "packed_code_root": str(root),
            "selection": f"fixed sample indices [0,{count})",
            "augmentation": args.augmentation,
            "order": "per-sample deterministic NumPy permutation",
            "order_seed": args.order_seed,
            "num_samples": total,
            "num_tokens": total * positions,
            "world_size": world_size,
            "batch_size_per_gpu": args.batch_size,
            "ar_dtype": args.ar_dtype,
            "overall": {
                "cross_entropy": mean_ce,
                "perplexity": math.exp(min(mean_ce, 50.0)),
                "top1_accuracy": sum(step_top1) / positions,
                "top5_accuracy": sum(step_top5) / positions,
            },
            "generation_step_groups": {
                f"steps_{begin:03d}_{end - 1:03d}": {
                    "cross_entropy": sum(step_ce[begin:end]) / (end - begin),
                    "top1_accuracy": sum(step_top1[begin:end]) / (end - begin),
                    "top5_accuracy": sum(step_top5[begin:end]) / (end - begin),
                }
                for begin, end in ranges
            },
            "generation_steps": [
                {
                    "step": index,
                    "cross_entropy": step_ce[index],
                    "top1_accuracy": step_top1[index],
                    "top5_accuracy": step_top5[index],
                }
                for index in range(positions)
            ],
            "spatial_positions": [
                {
                    "position": index,
                    "row": index // 16,
                    "column": index % 16,
                    "cross_entropy": spatial_ce[index],
                    "top1_accuracy": spatial_top1[index],
                    "top5_accuracy": spatial_top5[index],
                }
                for index in range(positions)
            ],
        }
        output = Path(args.output_json).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
