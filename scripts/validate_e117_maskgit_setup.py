#!/usr/bin/env python3
"""Fail-fast validation for the formal 8-H200 E117 MaskGIT run."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motar.data import validate_packed_dataset  # noqa: E402

E117_SHA256 = "a5b84689d2b29f579d2442da7594ac093292b6386760867a0668ca02f82e6156"


def validate_route_cache(route_root, packed_report, *, spot_check_rows=64):
    root = Path(route_root).expanduser().resolve()
    required = ("route_k.npy", "route_indices.npy", "source_indices.npy", "written.npy", "meta.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"route cache is missing files under {root}: {missing}")
    metadata = json.loads((root / "meta.json").read_text())
    if metadata.get("format") != "e117_sparse_route_cache_v1":
        raise ValueError(f"unsupported route-cache format: {metadata.get('format')!r}")
    if not metadata.get("completed", False):
        raise RuntimeError(f"route cache is not complete: {root}")
    if metadata.get("e117_checkpoint_sha256") != E117_SHA256:
        raise ValueError(
            "route cache was not produced by the registered E117 checkpoint: "
            f"{metadata.get('e117_checkpoint_sha256')!r}"
        )

    route_k = np.load(root / "route_k.npy", mmap_mode="r")
    route_indices = np.load(root / "route_indices.npy", mmap_mode="r")
    source_indices = np.load(root / "source_indices.npy", mmap_mode="r")
    written = np.load(root / "written.npy", mmap_mode="r")
    if route_k.ndim != 2 or route_indices.shape != (*route_k.shape, 128):
        raise ValueError(
            f"route shapes must be [N,A] and [N,A,128], got {route_k.shape}, "
            f"{route_indices.shape}"
        )
    if written.shape != route_k.shape:
        raise ValueError(f"written shape {written.shape} != route_k shape {route_k.shape}")
    if source_indices.shape != (route_k.shape[0],):
        raise ValueError("source_indices does not match route rows")
    if int(metadata.get("num_samples", -1)) != int(route_k.shape[0]):
        raise ValueError("route metadata num_samples does not match arrays")
    if int(metadata.get("num_aug", -1)) != int(route_k.shape[1]):
        raise ValueError("route metadata num_aug does not match arrays")
    if int(route_k.shape[1]) != int(packed_report["num_augmentations"]):
        raise ValueError("route and packed-code augmentation counts differ")
    if int(np.count_nonzero(written)) != int(written.size):
        raise RuntimeError("route cache contains unwritten entries")
    unique_k = sorted(int(value) for value in np.unique(route_k))
    if unique_k != [64, 128]:
        raise ValueError(f"route K values must be exactly [64,128], got {unique_k}")
    sources = np.asarray(source_indices)
    if sources.size != np.unique(sources).size:
        raise ValueError("route source indices are not unique")
    if sources.min() < 0 or sources.max() >= int(packed_report["num_samples"]):
        raise ValueError("route source index is outside the packed-code dataset")

    row_count = min(max(1, int(spot_check_rows)), int(route_k.shape[0]))
    rows = np.linspace(0, route_k.shape[0] - 1, row_count, dtype=np.int64)
    for row in rows:
        for augmentation in range(route_k.shape[1]):
            k = int(route_k[row, augmentation])
            selected = np.asarray(route_indices[row, augmentation, :k], dtype=np.int16)
            if selected.min() < 0 or selected.max() >= 256:
                raise ValueError(f"route coordinate out of range at row={row}, aug={augmentation}")
            if np.any(np.diff(selected) <= 0):
                raise ValueError(
                    f"route coordinates are not unique/sorted at row={row}, aug={augmentation}"
                )
    return {
        "root": str(root),
        "shape_k": list(route_k.shape),
        "shape_indices": list(route_indices.shape),
        "k_values": unique_k,
        "mean_k": float(np.asarray(route_k, dtype=np.float64).mean()),
        "e117_checkpoint_sha256": E117_SHA256,
        "completed": True,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-code-root", required=True)
    parser.add_argument("--route-cache", required=True)
    parser.add_argument("--eval-packed-code-root", required=True)
    parser.add_argument("--eval-route-cache", required=True)
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--allow-non-h200", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(index) for index in range(count)]
    if count != args.expected_gpus:
        raise RuntimeError(f"expected {args.expected_gpus} visible GPUs, found {count}")
    if not args.allow_non_h200:
        bad = [name for name in names if "H200" not in name.upper()]
        if bad:
            raise RuntimeError(f"expected only NVIDIA H200 GPUs, found: {bad}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("visible CUDA devices do not report bf16 support")

    train_codes = validate_packed_dataset(args.packed_code_root)
    val_codes = validate_packed_dataset(args.eval_packed_code_root)
    train_routes = validate_route_cache(args.route_cache, train_codes)
    val_routes = validate_route_cache(args.eval_route_cache, val_codes)
    if int(train_codes["num_samples"]) != 1_281_167:
        raise ValueError("formal train packed cache must contain 1,281,167 ImageNet sources")
    if int(val_codes["num_samples"]) != 50_000:
        raise ValueError("formal validation packed cache must contain 50,000 ImageNet sources")

    packages = {}
    for name in ("torch", "accelerate", "numpy", "omegaconf", "wandb", "einops"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    print(
        json.dumps(
            {
                "status": "ok",
                "python": platform.python_version(),
                "torch_cuda": torch.version.cuda,
                "gpu_count": count,
                "gpu_names": names,
                "bf16_supported": True,
                "train_codes": train_codes,
                "train_routes": train_routes,
                "val_codes": val_codes,
                "val_routes": val_routes,
                "packages": packages,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
