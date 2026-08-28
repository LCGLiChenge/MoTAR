#!/usr/bin/env python3
"""Download and verify the exact tokenizer assets used for packed-code extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download

REPOSITORIES = {
    "titok": {
        "url": "https://github.com/bytedance/1d-tokenizer.git",
        "commit": "942a96fbdd873780179d1b78d5462911528bf8c8",
    },
    "llamagen": {
        "url": "https://github.com/FoundationVision/LlamaGen.git",
        "commit": "ce98ec41803a74a90ce68c40ababa9eaeffeb4ec",
    },
}

CHECKPOINTS = {
    "mot": {
        "repo_id": "sophiaa/MoT-1-checkpoints",
        "revision": "0ed66fb6f5f3edc79205fab87c39139772caab4d",
        "filename": "latest.pt",
        "size": 6_400_628_829,
        "sha256": "86c8f9da5e61261ab93066c73d7719203e8c00b69f05b805c5937e6b7319b446",
    },
    "titok": {
        "repo_id": "fun-research/TiTok",
        "revision": "ab646ed225080a3acb7c78440a574d7f67f16fa7",
        "filename": "tokenizer_titok_l32.bin",
        "size": 2_564_477_610,
        "sha256": "b8f0bf61e9ee1791d8b76fa723bdcb2c85a039a7d027e597f685db492935c31f",
    },
}


def run(command):
    subprocess.run(command, check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_repository(repo_root: Path, name: str, spec: dict[str, str]) -> Path:
    target = repo_root / name
    if target.exists():
        if not (target / ".git").is_dir():
            raise RuntimeError(f"Existing dependency is not a Git repository: {target}")
        current = subprocess.check_output(
            ["git", "-C", str(target), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if current != spec["commit"]:
            raise RuntimeError(
                f"{target} is at {current}; expected {spec['commit']}. "
                "Use a new ASSET_ROOT instead of overwriting it."
            )
        return target

    run(["git", "clone", spec["url"], str(target)])
    run(["git", "-C", str(target), "checkout", "--detach", spec["commit"]])
    return target


def ensure_checkpoint(
    checkpoint_root: Path,
    name: str,
    spec: dict[str, str | int],
    *,
    local_files_only: bool,
) -> Path:
    local_dir = checkpoint_root / name
    path = Path(
        hf_hub_download(
            repo_id=str(spec["repo_id"]),
            filename=str(spec["filename"]),
            revision=str(spec["revision"]),
            local_dir=local_dir,
            local_files_only=local_files_only,
        )
    ).resolve()
    size = path.stat().st_size
    if size != int(spec["size"]):
        raise RuntimeError(f"Wrong size for {path}: {size} != {spec['size']}")
    digest = sha256_file(path)
    if digest != spec["sha256"]:
        raise RuntimeError(f"SHA256 mismatch for {path}: {digest} != {spec['sha256']}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.asset_root).expanduser().resolve()
    repo_root = root / "repos"
    checkpoint_root = root / "checkpoints"
    repo_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    repos = {
        name: ensure_repository(repo_root, name, spec)
        for name, spec in REPOSITORIES.items()
    }
    checkpoints = {
        name: ensure_checkpoint(
            checkpoint_root,
            name,
            spec,
            local_files_only=args.local_files_only,
        )
        for name, spec in CHECKPOINTS.items()
    }
    manifest = {
        "format_version": 1,
        "repositories": {
            name: {"path": str(repos[name]), **spec}
            for name, spec in REPOSITORIES.items()
        },
        "checkpoints": {
            name: {"path": str(checkpoints[name]), **spec}
            for name, spec in CHECKPOINTS.items()
        },
    }
    manifest_path = root / "extraction_assets.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
