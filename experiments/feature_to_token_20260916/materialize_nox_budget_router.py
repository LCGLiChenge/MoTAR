"""Save a versioned no-xbase Router with a train-fitted K threshold.

The spatial model and feature-proxy tensors remain bitwise unchanged. A
disjoint generated calibration cohort must meet the requested budget interval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bert2d.paths import atomic_json, output_path, sha256


def main(args: argparse.Namespace) -> None:
    train = json.loads((args.train_scores / "summary.json").read_text())
    calibration = json.loads((args.generated_scores / "summary.json").read_text())
    if (train["status"] != "complete" or train["mode"] != "packed_train"
            or train["n"] != 50000 or calibration["status"] != "complete"
            or calibration["mode"] != "generated" or calibration["n"] != 5000):
        raise RuntimeError("expected complete disjoint train and generated cohorts")
    digest = sha256(args.base_checkpoint)
    if train["router_proxy_sha256"] != digest or calibration["router_proxy_sha256"] != digest:
        raise RuntimeError("Router score cohorts were collected from different weights")
    threshold = float(train["thresholds"]["88"])
    scores = np.concatenate([np.load(args.generated_scores / f"score_rank{i}.npy",
                                      allow_pickle=False) for i in range(8)])
    if len(scores) != 5000 or not np.isfinite(scores).all():
        raise RuntimeError("generated calibration scores are incomplete")
    k128 = int((scores > threshold).sum())
    mean_2d = 64 + 64 * k128 / len(scores)
    if not args.min_generated_2d <= mean_2d <= args.max_generated_2d:
        raise RuntimeError(f"generated calibration mean {mean_2d} is outside approved interval")
    state = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
    if state.get("format") != "e117_no_xbase_feature_proxy_v1" or "k_threshold" in state:
        raise RuntimeError("expected the uncalibrated no-xbase Router checkpoint")
    state["k_threshold"] = threshold
    state["budget_calibration"] = dict(
        type="train_score_quantile_v1", target_train_mean_2d=88,
        training_cohort=50000, train_summary_sha256=sha256(args.train_scores / "summary.json"),
        generated_calibration_cohort=5000,
        generated_calibration_summary_sha256=sha256(args.generated_scores / "summary.json"),
        generated_calibration_mean_2d=mean_2d,
        generated_calibration_k128=k128,
        original_router_sha256=digest)
    out = output_path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.save(state, out / "latest.pt")
    reloaded = torch.load(out / "latest.pt", map_location="cpu", weights_only=True)
    if reloaded["k_threshold"] != threshold or set(reloaded["model"]) != set(state["model"]):
        raise RuntimeError("calibrated Router checkpoint round-trip failed")
    if not all(torch.equal(reloaded["model"][key], state["model"][key]) for key in state["model"]):
        raise RuntimeError("feature-proxy tensors changed during calibration")
    atomic_json(out / "summary.json", dict(
        status="complete", checkpoint_sha256=sha256(out / "latest.pt"),
        base_checkpoint_sha256=digest, threshold=threshold,
        target_train_mean_2d=88, generated_calibration_mean_2d=mean_2d,
        generated_calibration_k128=k128, feature_proxy_weights_unchanged=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--train-scores", type=Path, required=True)
    parser.add_argument("--generated-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-generated-2d", type=float, default=96.)
    parser.add_argument("--max-generated-2d", type=float, default=98.)
    main(parser.parse_args())
