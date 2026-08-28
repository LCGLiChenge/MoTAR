"""Packed TiTok/MoT code dataset and validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

REQUIRED_FILES = (
    "titok_codes.npy",
    "llamagen_codes.npy",
    "labels.npy",
    "written.npy",
    "meta.json",
)


def _load_arrays(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Packed dataset is missing files under {root}: {missing}")
    titok = np.load(root / "titok_codes.npy", mmap_mode="r")
    llamagen = np.load(root / "llamagen_codes.npy", mmap_mode="r")
    labels = np.load(root / "labels.npy", mmap_mode="r")
    written = np.load(root / "written.npy", mmap_mode="r")
    return titok, llamagen, labels, written


def validate_packed_dataset(
    packed_root: str | Path,
    *,
    scan_written: bool = True,
    spot_check_rows: int = 64,
) -> dict[str, Any]:
    """Validate the representation contract without loading full code arrays."""

    root = Path(packed_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Packed dataset directory not found: {root}")
    titok, llamagen, labels, written = _load_arrays(root)
    metadata = json.loads((root / "meta.json").read_text())

    if titok.ndim != 3 or titok.shape[-1] != 32:
        raise ValueError(f"titok_codes.npy must be [N,A,32], got {titok.shape}")
    if llamagen.ndim != 3 or llamagen.shape[-1] != 256:
        raise ValueError(f"llamagen_codes.npy must be [N,A,256], got {llamagen.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels.npy must be [N], got {labels.shape}")
    sample_count = int(titok.shape[0])
    if sample_count < 1:
        raise ValueError("Packed dataset is empty")
    if int(llamagen.shape[0]) != sample_count or int(labels.shape[0]) != sample_count:
        raise ValueError(
            "Packed arrays have mismatched sample counts: "
            f"titok={titok.shape}, llamagen={llamagen.shape}, labels={labels.shape}"
        )
    if titok.shape[1] < 1 or llamagen.shape[1] < 1:
        raise ValueError("Packed arrays must contain at least one augmentation")
    if written.shape != (sample_count,):
        raise ValueError(f"written.npy must be [{sample_count}], got {written.shape}")
    if not metadata.get("completed", False):
        raise RuntimeError(f"meta.json does not report completed=true: {metadata}")
    if int(metadata.get("num_samples", -1)) != sample_count:
        raise ValueError(
            f"meta.json num_samples={metadata.get('num_samples')} != arrays={sample_count}"
        )
    if scan_written:
        written_count = int(np.count_nonzero(written))
        if written_count != sample_count:
            raise RuntimeError(
                f"Packed extraction incomplete: {written_count:,}/{sample_count:,} rows"
            )

    check_count = min(max(int(spot_check_rows), 1), sample_count)
    indices = np.linspace(0, sample_count - 1, num=check_count, dtype=np.int64)
    titok_probe = np.asarray(titok[indices])
    llamagen_probe = np.asarray(llamagen[indices])
    label_probe = np.asarray(labels[indices])
    titok_min, titok_max = int(titok_probe.min()), int(titok_probe.max())
    llamagen_min, llamagen_max = int(llamagen_probe.min()), int(llamagen_probe.max())
    label_min, label_max = int(label_probe.min()), int(label_probe.max())
    if titok_min < 0 or titok_max >= 4096:
        raise ValueError(f"TiTok spot-check range is invalid: {titok_min}..{titok_max}")
    if llamagen_min < 0 or llamagen_max >= 16384:
        raise ValueError(
            f"MoT/LlamaGen spot-check range is invalid: {llamagen_min}..{llamagen_max}"
        )
    if label_min < 0 or label_max >= 1000:
        raise ValueError(f"ImageNet label spot-check range is invalid: {label_min}..{label_max}")

    return {
        "root": str(root),
        "num_samples": sample_count,
        "num_augmentations": min(int(titok.shape[1]), int(llamagen.shape[1])),
        "titok_shape": list(titok.shape),
        "llamagen_shape": list(llamagen.shape),
        "labels_shape": list(labels.shape),
        "dtypes": {
            "titok": str(titok.dtype),
            "llamagen": str(llamagen.dtype),
            "labels": str(labels.dtype),
            "written": str(written.dtype),
        },
        "spot_check": {
            "rows": check_count,
            "titok_range": [titok_min, titok_max],
            "llamagen_range": [llamagen_min, llamagen_max],
            "label_range": [label_min, label_max],
        },
        "completed": True,
    }


class PackedJointCodeDataset(Dataset):
    """Memory-mapped paired TiTok-L32 and MoT/LlamaGen VQ-16 codes."""

    def __init__(self, packed_root: str | Path, *, validate: bool = True):
        self.root = Path(packed_root).expanduser().resolve()
        if validate:
            validate_packed_dataset(self.root)
        self.titok, self.llamagen, self.labels, _ = _load_arrays(self.root)
        self.num_augmentations = min(self.titok.shape[1], self.llamagen.shape[1])

    def __len__(self) -> int:
        return int(self.titok.shape[0])

    def __getitem__(self, index: int):
        augmentation = int(torch.randint(0, self.num_augmentations, (1,)).item())
        z1d = torch.from_numpy(
            np.array(self.titok[index, augmentation], dtype=np.int64, copy=True)
        )
        z2d = torch.from_numpy(
            np.array(self.llamagen[index, augmentation], dtype=np.int64, copy=True)
        )
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return z1d, z2d, label, torch.tensor(index, dtype=torch.long)
