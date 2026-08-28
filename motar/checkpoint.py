"""Latest-only checkpoint save, promotion, validation, and resume helpers."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

LATEST_NAME = "latest"
STAGED_NAME = ".latest-next"
PREVIOUS_NAME = ".latest-previous"
REQUIRED_STATE_FILES = ("model.safetensors", "optimizer.bin")


def _managed_path(checkpoint_root: Path, name: str) -> Path:
    root = checkpoint_root.resolve()
    path = (root / name).resolve()
    if path.parent != root or name not in {LATEST_NAME, STAGED_NAME, PREVIOUS_NAME}:
        raise ValueError(f"Refusing unmanaged checkpoint path: {path}")
    return path


def _remove_managed_tree(checkpoint_root: Path, name: str) -> None:
    path = _managed_path(checkpoint_root, name)
    if path.exists():
        if not path.is_dir():
            raise RuntimeError(f"Managed checkpoint path is not a directory: {path}")
        shutil.rmtree(path)


def checkpoint_metadata(checkpoint_dir: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_dir)
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Checkpoint metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if int(metadata.get("format_version", -1)) != 1:
        raise ValueError(f"Unsupported checkpoint metadata: {metadata}")
    return metadata


def validate_checkpoint(checkpoint_dir: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_dir)
    if not path.is_dir():
        raise NotADirectoryError(f"Checkpoint directory not found: {path}")
    missing = [name for name in REQUIRED_STATE_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint is missing state files: {missing}")
    empty = [name for name in REQUIRED_STATE_FILES if (path / name).stat().st_size <= 0]
    if empty:
        raise RuntimeError(f"Checkpoint contains empty state files: {empty}")
    metadata = checkpoint_metadata(path)
    expected_ranks = int(metadata["world_size"])
    random_states = sorted(path.glob("random_states_*.pkl"))
    if random_states and len(random_states) != expected_ranks:
        raise RuntimeError(
            f"Checkpoint random-state count={len(random_states)} != world_size={expected_ranks}"
        )
    return metadata


def recover_checkpoint_tree(checkpoint_root: str | Path) -> Path | None:
    """Recover a save interrupted during directory rotation.

    At a stable point only checkpoints/latest remains. Temporary directories are
    confined to this checkpoint root and are never versioned epoch checkpoints.
    """

    root = Path(checkpoint_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    latest = _managed_path(root, LATEST_NAME)
    staged = _managed_path(root, STAGED_NAME)
    previous = _managed_path(root, PREVIOUS_NAME)

    if latest.is_dir():
        _remove_managed_tree(root, STAGED_NAME)
        _remove_managed_tree(root, PREVIOUS_NAME)
        validate_checkpoint(latest)
        return latest

    if previous.is_dir():
        previous.rename(latest)
        _remove_managed_tree(root, STAGED_NAME)
        validate_checkpoint(latest)
        return latest

    if staged.is_dir():
        validate_checkpoint(staged)
        staged.rename(latest)
        return latest

    return None


def promote_staged_checkpoint(checkpoint_root: str | Path) -> Path:
    root = Path(checkpoint_root).expanduser().resolve()
    latest = _managed_path(root, LATEST_NAME)
    staged = _managed_path(root, STAGED_NAME)
    previous = _managed_path(root, PREVIOUS_NAME)
    validate_checkpoint(staged)

    _remove_managed_tree(root, PREVIOUS_NAME)
    if latest.exists():
        if not latest.is_dir():
            raise RuntimeError(f"latest is not a directory: {latest}")
        latest.rename(previous)
    staged.rename(latest)
    _remove_managed_tree(root, PREVIOUS_NAME)
    return latest


def save_latest_checkpoint(
    accelerator,
    checkpoint_root: str | Path,
    metadata: dict[str, Any],
) -> Path:
    """Collectively save state and replace the single stable latest checkpoint."""

    root = Path(checkpoint_root).expanduser().resolve()
    staged = _managed_path(root, STAGED_NAME)
    if accelerator.is_main_process:
        root.mkdir(parents=True, exist_ok=True)
        _remove_managed_tree(root, STAGED_NAME)
    accelerator.wait_for_everyone()

    accelerator.save_state(str(staged))
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        payload = dict(metadata)
        payload["format_version"] = 1
        metadata_tmp = staged / ".metadata.json.tmp"
        metadata_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(metadata_tmp, staged / "metadata.json")
        validate_checkpoint(staged)
        latest = promote_staged_checkpoint(root)
    else:
        latest = root / LATEST_NAME
    accelerator.wait_for_everyone()
    return latest


def load_model_optimizer_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    """Load model/optimizer on CPU before Accelerator prepares them.

    Only native epoch-boundary ``latest`` states with metadata are accepted. The
    scheduler is rebuilt from the recorded absolute H200 epoch progress.
    """

    path = Path(checkpoint_dir).expanduser().resolve()
    metadata = validate_checkpoint(path)

    state = load_file(str(path / "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Bad model resume: missing={missing}, unexpected={unexpected}")
    del state

    optimizer_state = torch.load(
        path / "optimizer.bin",
        map_location="cpu",
        weights_only=False,
    )
    optimizer.load_state_dict(optimizer_state)
    del optimizer_state
    return metadata
