#!/usr/bin/env python3
"""Probe one H200 micro-batch with the exact two-branch MaskGIT train step."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from e117_sparse_maskgit import (  # noqa: E402
    E117SparseUnifiedMaskGIT,
    masked_token_metrics,
    sample_arccos_mask,
)
from train_e117_sparse_maskgit import (  # noqa: E402
    masked_inputs_1d,
    masked_inputs_2d,
    optimizer_for,
    update_ema_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e117_maskgit_h200_8gpu.yaml")
    parser.add_argument("--candidate", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--memory-fraction", type=float, default=0.90)
    parser.add_argument("--allow-non-h200", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("batch probe requires exactly one visible CUDA GPU")
    if args.candidate < 1 or args.world_size < 1:
        raise ValueError("candidate and world size must be positive")
    if not 0.1 < args.memory_fraction < 1.0:
        raise ValueError("memory fraction must be between 0.1 and 1.0")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    if "H200" not in gpu_name.upper() and not args.allow_non_h200:
        raise RuntimeError(f"expected an H200, found {gpu_name!r}")

    config = OmegaConf.load(args.config)
    batch = int(args.candidate)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        model = E117SparseUnifiedMaskGIT(
            **OmegaConf.to_container(config.model, resolve=True)
        ).to(device)
        model.train()
        # Formal training keeps a full fp32 EMA in addition to model/optimizer.
        ema_model = copy.deepcopy(model).to(device).eval().requires_grad_(False)
        optimizer = optimizer_for(model, config.optimizer)

        z1d = torch.randint(0, int(config.model.titok_vocab_size), (batch, 32), device=device)
        z2d = torch.randint(
            0,
            int(config.model.llamagen_vocab_size),
            (batch, int(config.model.max_sparse_tokens)),
            device=device,
        )
        labels = torch.randint(0, int(config.model.num_classes), (batch,), device=device)
        route_indices = torch.arange(
            int(config.model.max_sparse_tokens), device=device
        )[None].expand(batch, -1)
        route_valid = torch.ones_like(route_indices, dtype=torch.bool)
        mask_1d, _ = sample_arccos_mask(torch.ones_like(z1d, dtype=torch.bool))
        mask_2d, _ = sample_arccos_mask(route_valid)
        input_1d = masked_inputs_1d(model, z1d, mask_1d)
        input_2d = masked_inputs_2d(model, z2d, mask_2d, route_valid)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits_1d = model(stage="1d", input_tokens=input_1d, labels=labels)
            logits_2d = model(
                stage="2d",
                completed_1d=z1d,
                input_tokens=input_2d,
                route_indices=route_indices,
                route_valid=route_valid,
                labels=labels,
            )
            metrics_1d = masked_token_metrics(
                logits_1d,
                z1d,
                mask_1d,
                torch.ones_like(z1d, dtype=torch.bool),
                config.training.label_smoothing,
                config.training.unmasked_loss_weight,
                config.training.loss_normalization,
            )
            metrics_2d = masked_token_metrics(
                logits_2d,
                z2d,
                mask_2d,
                route_valid,
                config.training.label_smoothing,
                config.training.unmasked_loss_weight,
                config.training.loss_normalization,
            )
            loss = (
                float(config.training.loss_1d_weight) * metrics_1d["loss"]
                + float(config.training.loss_2d_weight) * metrics_2d["loss"]
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.optimizer.max_grad_norm))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_ema_model(ema_model, model, float(config.training.ema_decay))
        torch.cuda.synchronize(device)
    except torch.OutOfMemoryError as error:
        print(
            json.dumps(
                {
                    "candidate": batch,
                    "gpu": gpu_name,
                    "status": "oom",
                    "error": str(error).splitlines()[0],
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2

    total = int(torch.cuda.get_device_properties(device).total_memory)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    fraction = peak_reserved / total
    status = "pass" if fraction <= args.memory_fraction else "over_margin"
    print(
        json.dumps(
            {
                "candidate": batch,
                "global_batch_size": batch * int(args.world_size),
                "gpu": gpu_name,
                "total_bytes": total,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "peak_reserved_fraction": fraction,
                "memory_fraction_limit": float(args.memory_fraction),
                "status": status,
            },
            sort_keys=True,
        )
    )
    return 0 if status == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
