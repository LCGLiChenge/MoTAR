#!/usr/bin/env python3
"""Fail-fast validation for the 8xH200 MoTAR handoff."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motar.checkpoint import validate_checkpoint
from motar.data import validate_packed_dataset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-code-root", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--allow-non-h200", action="store_true")
    parser.add_argument("--verify-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpu_count = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(index) for index in range(gpu_count)]
    if gpu_count != args.expected_gpus:
        raise RuntimeError(f"Expected {args.expected_gpus} visible GPUs, found {gpu_count}")
    if not args.allow_non_h200:
        bad = [name for name in gpu_names if "H200" not in name.upper()]
        if bad:
            raise RuntimeError(f"Expected only H200 GPUs, found non-H200 devices: {bad}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Visible CUDA device does not report bf16 support")

    dataset = validate_packed_dataset(args.packed_code_root)
    checkpoint = None
    if args.checkpoint:
        checkpoint = validate_checkpoint(args.checkpoint)

    manifest_result = None
    if args.verify_manifest:
        root = Path(args.packed_code_root).expanduser().resolve()
        manifest = root / "manifest.sha256"
        if not manifest.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest}")
        manifest_result = {}
        for raw_line in manifest.read_text().splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            expected, name = raw_line.split(maxsplit=1)
            name = name.lstrip("*")
            actual = sha256_file(root / name)
            if actual != expected:
                raise RuntimeError(f"SHA256 mismatch for {name}: {actual} != {expected}")
            manifest_result[name] = actual

    packages = {}
    for name in ("torch", "accelerate", "omegaconf", "safetensors", "numpy", "einops"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"

    report = {
        "status": "ok",
        "python": platform.python_version(),
        "torch_cuda": torch.version.cuda,
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "bf16_supported": True,
        "dataset": dataset,
        "checkpoint": checkpoint,
        "manifest": manifest_result,
        "packages": packages,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
