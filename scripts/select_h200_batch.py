#!/usr/bin/env python3
"""Select the largest passing H200 micro batch in isolated subprocesses."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/h200_8gpu_150epoch.yaml")
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--memory-fraction", type=float, default=None)
    parser.add_argument("--allow-non-h200", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = OmegaConf.load(args.config)
    if args.candidates:
        candidates = [int(value) for value in args.candidates.split(",") if value]
    else:
        candidates = [int(value) for value in config.h200.batch_candidates]
    candidates = sorted(set(candidates), reverse=True)
    memory_fraction = float(
        args.memory_fraction
        if args.memory_fraction is not None
        else config.h200.memory_fraction
    )
    if not candidates:
        raise ValueError("No batch candidates supplied")

    for candidate in candidates:
        command = [
            sys.executable,
            "scripts/probe_batch_size.py",
            "--config",
            args.config,
            "--candidate",
            str(candidate),
            "--world-size",
            str(config.h200.required_gpu_count),
            "--memory-fraction",
            str(memory_fraction),
        ]
        if args.allow_non_h200:
            command.append("--allow-non-h200")
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            cwd=REPO_ROOT,
        )
        for stream in (completed.stdout, completed.stderr):
            if stream.strip():
                print(stream.rstrip(), file=sys.stderr)
        if completed.returncode == 0:
            print(candidate)
            return 0

    print(
        json.dumps(
            {
                "status": "no_candidate_passed",
                "candidates": candidates,
                "memory_fraction": memory_fraction,
            }
        ),
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
