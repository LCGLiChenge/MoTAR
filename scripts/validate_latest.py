#!/usr/bin/env python3
"""Validate and print the single stable latest checkpoint metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motar.checkpoint import validate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir")
    args = parser.parse_args()
    print(json.dumps(validate_checkpoint(args.checkpoint_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
