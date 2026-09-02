#!/usr/bin/env python3
"""Validate the latest-only E117 MaskGIT checkpoint contract."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("--expected-world-size", type=int, default=8)
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    latest = output / "latest.pt"
    if not latest.is_file():
        raise FileNotFoundError(latest)
    siblings = sorted(path.name for path in output.glob("*.pt") if path.name != "latest.pt")
    if siblings:
        raise RuntimeError(f"unexpected versioned checkpoints: {siblings}")
    checkpoint = torch.load(latest, map_location="cpu", weights_only=False, mmap=True)
    required = {
        "format", "model", "model_ema", "optimizer", "scheduler", "global_step",
        "epoch", "config", "config_sha256", "resume_capable", "rng_states",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise KeyError(f"latest checkpoint is missing fields: {missing}")
    if checkpoint["format"] != "e117_sparse_unified_maskgit_v3":
        raise ValueError(f"unexpected checkpoint format: {checkpoint['format']!r}")
    if not checkpoint["resume_capable"]:
        raise ValueError("latest checkpoint is not resume-capable")
    if len(checkpoint["rng_states"]) != args.expected_world_size:
        raise ValueError(
            f"checkpoint world size {len(checkpoint['rng_states'])} "
            f"!= {args.expected_world_size}"
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "path": str(latest),
                "bytes": latest.stat().st_size,
                "global_step": int(checkpoint["global_step"]),
                "completed_data_epochs": int(checkpoint["epoch"]),
                "format": checkpoint["format"],
                "resume_capable": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
