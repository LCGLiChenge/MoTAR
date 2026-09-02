#!/usr/bin/env python3
"""Require a configured W&B credential unless offline mode is explicit."""

import argparse
import json

import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="online")
    args = parser.parse_args()
    mode = args.mode.lower()
    if mode == "offline":
        print(json.dumps({"status": "ok", "mode": "offline", "authenticated": False}))
        return
    if mode not in {"online", "run"}:
        raise ValueError("logging mode must be online or explicitly offline")
    if not bool(getattr(wandb.api, "api_key", None)):
        raise RuntimeError("W&B is not authenticated; run 'wandb login' before training")
    print(json.dumps({"status": "ok", "mode": "online", "authenticated": True}))


if __name__ == "__main__":
    main()
