"""Merge disjoint reconstruction shards and compute exact same-cohort 50k rFID."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from bert2d.paths import atomic_json
from experiments.feature_to_token_20260916.eval_no_xbase_router_recon import ARMS, paired_rfid


def run(root: Path, n: int, shards: int) -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("select one GPU to compute the FID statistics")
    if shards < 2 or n < 2:
        raise ValueError("invalid merge shape")
    root = root.resolve()
    if (root / "summary.json").exists():
        raise FileExistsError(root / "summary.json")
    summaries = []
    manifests = []
    indices = []
    features = {arm: [] for arm in ARMS}
    for rank in range(shards):
        directory = root / f"rank{rank}"
        summary = json.loads((directory / "summary.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        local_indices = np.load(directory / "cohort_indices.npy", allow_pickle=False)
        if (summary["status"] != "complete" or summary["smoke"] or summary["n"] != n
                or summary["shard_rank"] != rank or summary["shard_count"] != shards
                or manifest["n"] != n or manifest["shard_rank"] != rank
                or manifest["shard_count"] != shards or len(local_indices) != summary["local_n"]):
            raise ValueError(f"invalid shard {rank} identity or coverage")
        for arm in ARMS:
            rows = np.load(directory / f"{arm}_features.npy", mmap_mode="r", allow_pickle=False)
            if rows.shape != (len(local_indices), 2048) or rows.dtype != np.float32:
                raise ValueError(f"invalid {arm} feature shape in shard {rank}")
            if not np.isfinite(rows).all():
                raise ValueError(f"nonfinite {arm} feature in shard {rank}")
            features[arm].append(np.asarray(rows))
        summaries.append(summary)
        manifests.append(manifest)
        indices.append(local_indices)
    all_indices = np.concatenate(indices)
    if len(all_indices) != n or not np.array_equal(np.sort(all_indices), np.arange(n)):
        raise ValueError("shards do not cover each validation image exactly once")
    for manifest in manifests[1:]:
        for key in ("protocol", "n", "seed", "batch_size", "val_cache_sha256",
                    "frozen_assets", "original_router", "proxy_router", "decoder",
                    "generator_used", "fid_weight_sha256", "metric", "tf32", "source_sha256"):
            if manifest[key] != manifests[0][key]:
                raise ValueError(f"shard protocol mismatch: {key}")
    merged = {arm: np.concatenate(rows, axis=0) for arm, rows in features.items()}
    metrics = {}
    for arm in ARMS[1:]:
        means = np.array([item["metrics"][arm]["psnr"] for item in summaries], dtype=np.float64)
        stds = np.array([item["metrics"][arm]["psnr_std"] for item in summaries], dtype=np.float64)
        weights = np.array([item["local_n"] for item in summaries], dtype=np.float64) / n
        mean = float((weights * means).sum())
        variance = float((weights * (stds * stds + means * means)).sum() - mean * mean)
        metrics[arm] = dict(psnr=mean, psnr_std=max(0., variance)**0.5,
                            rfid=paired_rfid(merged[arm], merged["real"]))
    totals = {key: sum(item["route_totals"][key] for item in summaries)
              for key in summaries[0]["route_totals"]}
    if totals["rows"] != n:
        raise ValueError("route metrics do not cover all images")
    routes = dict(rows=n, k_original_mean=totals["k_original_sum"]/n,
        k_nox_mean=totals["k_nox_sum"]/n, k_accuracy=totals["k_equal"]/n,
        pooled_iou=totals["intersection"]/totals["union"],
        mean_image_iou=totals["image_iou_sum"]/n,
        matched_k_pooled_iou=totals["matched_intersection"]/totals["matched_union"])
    result = dict(status="complete", n=n, shards=shards, metrics=metrics, routes=routes,
        paired_rfid_delta_nox_minus_e117=metrics["no_xbase"]["rfid"]-metrics["e117"]["rfid"],
        paired_rfid_delta_matched_k_minus_e117=metrics["no_xbase_matched_k"]["rfid"]-metrics["e117"]["rfid"],
        paired_psnr_delta_nox_minus_e117=metrics["no_xbase"]["psnr"]-metrics["e117"]["psnr"],
        paired_psnr_delta_matched_k_minus_e117=metrics["no_xbase_matched_k"]["psnr"]-metrics["e117"]["psnr"],
        protocol=manifests[0]["protocol"], cohort="all 50k ImageNet-val images; real 1D/2D codes",
        interpretation="same-image reconstruction rFID, not generation FID")
    atomic_json(root / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--shards", type=int, default=4)
    print(json.dumps(run(**vars(parser.parse_args()))), flush=True)
