"""Download and verify a manifest-complete RGB proxy-cache handoff."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def main(args) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix = args.path_in_repo.rstrip("/")
    manifest_source = Path(hf_hub_download(
        repo_id=args.repo_id,
        filename=f"{prefix}/partial_manifest.json",
        revision=args.revision,
        repo_type="model",
        cache_dir=str(args.cache_dir),
    ))
    manifest = json.loads(manifest_source.read_text())
    if manifest.get("status") != "complete_partial_snapshot" or int(manifest.get("world_size", -1)) != 8:
        raise RuntimeError("remote handoff manifest is incomplete or incompatible")
    if int(manifest["completed"]) != sum(map(int, manifest["rank_completed"])):
        raise RuntimeError("remote manifest completed count differs")
    atomic_json(output / "download_status.json", {
        "status": "downloading",
        "repo_id": args.repo_id,
        "revision": args.revision,
        "completed": manifest["completed"],
        "total": manifest["total"],
    })
    for name, expected in manifest["files"].items():
        source = Path(hf_hub_download(
            repo_id=args.repo_id,
            filename=f"{prefix}/{name}",
            revision=args.revision,
            repo_type="model",
            cache_dir=str(args.cache_dir),
        ))
        destination = output / name
        if destination.exists():
            if destination.stat().st_size == int(expected["bytes"]) and sha256(destination) == expected["sha256"]:
                continue
            raise RuntimeError(f"refusing to overwrite mismatched destination: {destination}")
        temporary = destination.with_suffix(destination.suffix + ".download")
        if temporary.exists():
            raise RuntimeError(f"stale incomplete download requires inspection: {temporary}")
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=16 * 1024**2)
        if temporary.stat().st_size != int(expected["bytes"]) or sha256(temporary) != expected["sha256"]:
            raise RuntimeError(f"download verification failed: {name}")
        temporary.replace(destination)
    local_manifest = output / "partial_manifest.json"
    shutil.copy2(manifest_source, local_manifest)
    result = {
        "status": "complete",
        "repo_id": args.repo_id,
        "revision": args.revision,
        "completed": manifest["completed"],
        "total": manifest["total"],
        "all_sizes_and_sha256_verified": True,
    }
    atomic_json(output / "download_status.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="Chloeeeeeeee123/MoT-1")
    parser.add_argument("--path-in-repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    main(parser.parse_args())
