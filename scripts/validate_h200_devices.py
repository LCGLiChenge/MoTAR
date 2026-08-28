#!/usr/bin/env python3
"""Fail unless exactly the requested number of visible NVIDIA H200 GPUs exists."""

from __future__ import annotations

import argparse
import json

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--require-bf16", action="store_true")
    args = parser.parse_args()

    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(index) for index in range(count)]
    if count != args.expected_gpus:
        raise RuntimeError(f"Expected {args.expected_gpus} visible GPUs, found {count}")
    bad = [name for name in names if "H200" not in name.upper()]
    if bad:
        raise RuntimeError(f"Expected only NVIDIA H200 GPUs, found: {bad}")
    if args.require_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Visible H200 does not report bf16 support")
    print(
        json.dumps(
            {
                "status": "ok",
                "gpu_count": count,
                "gpu_names": names,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
