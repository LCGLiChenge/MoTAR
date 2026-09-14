from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "delivery_h20_20260910"
ASSETS = ROOT / "h20_local_assets"
HERE = Path(__file__).resolve().parent
RESULT_ROOT = Path(os.environ.get(
    "MOTAR_MASKGIT_RESULT_ROOT",
    "/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/maskgit_optimization_sweep_20260910",
))

def enable_h20() -> None:
    os.environ.setdefault("USE_TF", "0")
    release = str(RELEASE)
    if release not in sys.path:
        sys.path.insert(0, release)

def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()

def digest(path: str | Path) -> str:
    return sha256(path)

def atomic_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)

def output_path(path: str | Path) -> Path:
    path = Path(path).resolve()
    root = RESULT_ROOT.resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("output must be a specific child of " + str(root))
    return path

enable_h20()
