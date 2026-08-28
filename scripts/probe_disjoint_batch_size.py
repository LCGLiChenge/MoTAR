#!/usr/bin/env python3
"""Probe one per-GPU micro batch with a real forward/backward/AdamW step."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motar.model import TiTokLlamaGenUnifiedAR
from motar.sidecar import TiTok1DDisjointSidecar, detached_sidecar_inputs
from train_disjoint import build_optimizer, build_sidecar_optimizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/h200_8gpu_150epoch_disjoint.yaml")
    parser.add_argument("--candidate", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--memory-fraction", type=float, default=0.90)
    parser.add_argument("--allow-non-h200", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Batch probe requires exactly one visible CUDA GPU")
    if args.candidate < 1:
        raise ValueError("--candidate must be positive")
    if not 0.1 < args.memory_fraction < 1.0:
        raise ValueError("--memory-fraction must be between 0.1 and 1.0")

    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(device)
    if "H200" not in name.upper() and not args.allow_non_h200:
        raise RuntimeError(f"Expected an H200 for production probing, found {name!r}")

    config = OmegaConf.load(args.config)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        model = TiTokLlamaGenUnifiedAR(
            **OmegaConf.to_container(config.model, resolve=True)
        )
        sidecar = TiTok1DDisjointSidecar(
            model,
            depth=int(config.sidecar.depth),
            drop_path=float(config.sidecar.drop_path),
        )
        model = model.to(device)
        sidecar = sidecar.to(device)
        optimizer, _roles, _base_lr, _new_lr = build_optimizer(
            model,
            config,
            args.candidate * args.world_size,
        )
        sidecar_optimizer, _sidecar_lr = build_sidecar_optimizer(
            sidecar,
            config,
            args.candidate * args.world_size,
        )
        z1d = torch.randint(
            0,
            int(config.model.titok_vocab_size),
            (args.candidate, int(config.model.titok_num_tokens)),
            device=device,
        )
        z2d = torch.randint(
            0,
            int(config.model.vocab_size),
            (args.candidate, int(config.model.block_size)),
            device=device,
        )
        labels = torch.randint(
            0,
            int(config.model.num_classes),
            (args.candidate,),
            device=device,
        )
        model.train()
        sidecar.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                z1d=z1d,
                z2d=z2d,
                cond_idx=labels,
                targets_1d=z1d,
                targets_2d=z2d,
            )
            prefix_h, prefix_freqs = detached_sidecar_inputs(model, z1d, labels)
            sidecar_logits = sidecar(prefix_h, prefix_freqs)
            sidecar_loss = F.cross_entropy(
                (output["logits_1d"].detach() + sidecar_logits).reshape(
                    -1, int(config.model.titok_vocab_size)
                ),
                z1d.reshape(-1),
            )
        output["loss"].backward()
        (float(config.sidecar.aux_weight) * sidecar_loss).backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config.optimizer.max_grad_norm)
        )
        torch.nn.utils.clip_grad_norm_(
            sidecar.parameters(), float(config.optimizer.max_grad_norm)
        )
        optimizer.step()
        sidecar_optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        sidecar_optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
    except torch.OutOfMemoryError as exc:
        print(
            json.dumps(
                {
                    "candidate": args.candidate,
                    "gpu": name,
                    "status": "oom",
                    "error": str(exc).splitlines()[0],
                }
            ),
            file=sys.stderr,
        )
        return 2

    total = int(torch.cuda.get_device_properties(device).total_memory)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    fraction = peak_reserved / total
    status = "pass" if fraction <= args.memory_fraction else "over_margin"
    payload = {
        "candidate": args.candidate,
        "global_batch_size": args.candidate * args.world_size,
        "gpu": name,
        "total_bytes": total,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_reserved_fraction": fraction,
        "memory_fraction_limit": args.memory_fraction,
        "status": status,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if status == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
