"""Paired ImageNet-val reconstruction audit of E117 versus no-x_base E117.

Both arms consume the same cached *real* 1D/2D codes and frozen MoT EMA
decoder. Only the Router decision changes; no MaskGIT model is used.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
DELIVERY = ROOT / "delivery_h20_20260910"
for path in (ROOT, DELIVERY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.datasets import ImageFolder

from bert2d.paths import atomic_json, sha256
from h20_joint.assets import FrozenAssets
from experiments.feature_to_token_20260916.e117_no_xbase import install_no_xbase
from scripts.extract_mot_titok_llamagen_packed import PairedTransform
from e117_ar_adapter import E117ARDecisionAdapter
from e84_ar_adapter import E84ARDecisionAdapter


PACKED = Path("/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/data/imagenet-val-titok_l32-mot199440ema-none-256_packed")
FID_WEIGHT = Path("/home/heyefei/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth")
RESULT_ROOT = Path("/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results")
ARMS = ("real", "e117", "no_xbase", "no_xbase_matched_k")


def cohort(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n >= 1000:
        if n % 1000 or n > 50000:
            raise ValueError("full-class cohort size must be a multiple of 1000")
        classes = np.arange(1000)
        per_class = n // 1000
    else:
        classes = rng.permutation(1000)[:n]
        per_class = 1
    chosen = []
    for label in classes:
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < per_class:
            raise ValueError(f"class {label} has fewer than {per_class} examples")
        chosen.extend(rng.choice(candidates, per_class, replace=False).tolist())
    result = np.array(sorted(chosen), dtype=np.int64)
    if len(result) != n or len(np.unique(result)) != n:
        raise RuntimeError("invalid fixed reconstruction cohort")
    return result


def paired_rfid(fake: np.ndarray, real: np.ndarray) -> float:
    """Float64 GPU FID on features from the same image cohort."""
    x = torch.as_tensor(fake, device="cuda", dtype=torch.float64)
    y = torch.as_tensor(real, device="cuda", dtype=torch.float64)
    mx, my = x.mean(0), y.mean(0)
    x, y = x - mx, y - my
    cx = x.T @ x / (len(x) - 1)
    cy = y.T @ y / (len(y) - 1)
    eigen, vec = torch.linalg.eigh(cy)
    sqrt_y = (vec * eigen.clamp_min(0).sqrt()[None]) @ vec.T
    middle = sqrt_y @ cx @ sqrt_y
    cross = torch.linalg.eigvalsh((middle + middle.T) * 0.5).clamp_min(0).sqrt().sum()
    return float(((mx - my).square().sum() + torch.trace(cx) + torch.trace(cy) - 2 * cross).clamp_min(0))


def selected_mask(bundle: dict[str, torch.Tensor]) -> torch.Tensor:
    index, valid = bundle["index"], bundle["valid"]
    result = torch.zeros((len(index), 256), dtype=torch.bool, device=index.device)
    batch = torch.arange(len(index), device=index.device)[:, None].expand_as(index)
    result[batch[valid], index[valid]] = True
    return result


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("use exactly one visible GPU for this paired audit")
    if args.batch_size < 1 or args.n < 2:
        raise ValueError("invalid evaluation size")
    if not 1 <= args.shard_count <= 4 or not 0 <= args.shard_rank < args.shard_count:
        raise ValueError("invalid shard assignment")
    output = args.output.resolve()
    if not output.is_relative_to(RESULT_ROOT.resolve()) or output.exists():
        raise ValueError("output must be a new named directory under the AR results disk")
    output.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    metadata = json.loads((PACKED / "meta.json").read_text())
    if not metadata.get("completed") or metadata.get("num_samples") != 50000 or metadata.get("mot_step") != 199440:
        raise ValueError("expected complete MoT199440 ImageNet-val cache")
    dataset = ImageFolder(metadata["data_path"])
    labels = np.load(PACKED / "labels.npy", mmap_mode="r")
    if len(dataset) != 50000 or not np.array_equal(dataset.targets, labels):
        raise ValueError("cached codes and val image ordering differ")
    if not np.load(PACKED / "written.npy", mmap_mode="r").all():
        raise ValueError("val cache incomplete")
    chosen = cohort(labels, args.n, args.seed)[args.shard_rank::args.shard_count]
    local_n = len(chosen)
    codes1 = np.load(PACKED / "titok_codes.npy", mmap_mode="r")
    codes2 = np.load(PACKED / "llamagen_codes.npy", mmap_mode="r")
    np.save(output / "cohort_indices.npy", chosen)
    assets = FrozenAssets(args.assets_root, device="cuda", chunk=4)
    original = assets.adapter
    candidate = E117ARDecisionAdapter(assets.shell, copy.deepcopy(original.student)).cuda().eval()
    proxy_audit = install_no_xbase(SimpleNamespace(adapter=candidate, device=torch.device("cuda")),
                                   args.router_proxy_checkpoint)
    feature_metric = FrechetInceptionDistance(feature=2048, normalize=False,
        feature_extractor_weights_path=str(FID_WEIGHT)).cuda().eval().requires_grad_(False)
    if next(feature_metric.inception.parameters()).device.type != "cuda":
        raise RuntimeError("FID feature extractor is not on GPU")
    manifest = dict(protocol="paired_e117_no_xbase_reconstruction_v1", n=args.n,
        local_n=local_n, shard_rank=args.shard_rank, shard_count=args.shard_count,
        seed=args.seed, batch_size=args.batch_size, cohort="equal images per ImageNet class",
        cohort_sha256=sha256(output / "cohort_indices.npy"),
        val_cache_sha256=sha256(PACKED / "meta.json"),
        frozen_assets=assets.audit, original_router="E117 EMA", proxy_router=proxy_audit,
        source_sha256=sha256(Path(__file__)),
        decoder="MoT199440 EMA; selected GT 2D codes replace 1D features directly",
        generator_used=False, fid_weight_sha256=sha256(FID_WEIGHT),
        metric="same-cohort torchmetrics Inception 2048 rFID", tf32=False)
    atomic_json(output / "manifest.json", manifest)
    transform = PairedTransform(256)
    def load_image(index: int) -> torch.Tensor:
        with Image.open(dataset.samples[int(index)][0]) as image:
            return transform(image.convert("RGB"))[0]
    features: dict[str, list[np.ndarray]] = {key: [] for key in ARMS}
    psnr: dict[str, list[np.ndarray]] = {key: [] for key in ARMS[1:]}
    route_rows: list[dict[str, float]] = []
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0, local_n, args.batch_size):
            indices = chosen[offset:offset + args.batch_size]
            real = torch.stack(list(pool.map(load_image, indices))).cuda(non_blocking=True)
            z1 = torch.from_numpy(np.array(codes1[indices, 0], dtype=np.int64)).cuda()
            z2 = torch.from_numpy(np.array(codes2[indices, 0], dtype=np.int64)).cuda()
            assets.adapter = original
            baseline = assets.bundle(z1)
            assets.adapter = candidate
            nox = assets.bundle(z1)
            assets.adapter = original
            if offset == 0:
                torch.testing.assert_close(baseline["base"], nox["base"], atol=0, rtol=0)
            masks = {"e117": selected_mask(baseline), "no_xbase": selected_mask(nox)}
            intersect = (masks["e117"] & masks["no_xbase"]).sum(1).float()
            union = (masks["e117"] | masks["no_xbase"]).sum(1).clamp_min(1).float()
            k_original = baseline["valid"].sum(1)
            k_nox = nox["valid"].sum(1)
            _, matched_index, matched_valid = E84ARDecisionAdapter._selection_from_scores(
                nox["scores"], k_original)
            matched_index = matched_index.masked_fill(~matched_valid, 256).sort(1).values.masked_fill(~matched_valid, -1)
            matched = dict(base=nox["base"], index=matched_index, valid=matched_valid)
            masks["no_xbase_matched_k"] = selected_mask(matched)
            route_rows.append(dict(rows=len(indices), k_original_sum=int(k_original.sum()),
                k_nox_sum=int(k_nox.sum()), k_equal=int((k_original == k_nox).sum()),
                intersection=int(intersect.sum()), union=int(union.sum()),
                image_iou_sum=float((intersect / union).sum()),
                matched_intersection=int((masks["e117"] & masks["no_xbase_matched_k"]).sum()),
                matched_union=int((masks["e117"] | masks["no_xbase_matched_k"]).sum())))
            images = {"real": real}
            for arm, bundle in (("e117", baseline), ("no_xbase", nox), ("no_xbase_matched_k", matched)):
                selected = z2.gather(1, bundle["index"].clamp_min(0))
                mixed = assets.mixed(bundle["base"], selected, bundle["index"], bundle["valid"])
                rendered = torch.cat([assets.render(part) for part in mixed.split(4)])
                images[arm] = rendered
                mse = (rendered - real).square().flatten(1).mean(1)
                psnr[arm].append((-10 * (mse + 1e-8).log10()).cpu().numpy())
            for arm, image in images.items():
                encoded = (image.clamp(0, 1) * 255).round().to(torch.uint8)
                vector = feature_metric.inception(encoded)
                if vector.shape != (len(indices), 2048) or not torch.isfinite(vector).all():
                    raise RuntimeError(f"invalid FID features for {arm}")
                features[arm].append(vector.cpu().numpy())
            if offset == 0 or (offset + len(indices)) % args.log_every < args.batch_size or offset + len(indices) == local_n:
                progress = dict(status="running", completed=offset + len(indices), total=local_n,
                    elapsed_seconds=time.monotonic() - started,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
                atomic_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    merged = {arm: np.concatenate(parts) for arm, parts in features.items()}
    if any(value.shape != (local_n, 2048) for value in merged.values()):
        raise RuntimeError("feature coverage mismatch")
    if not args.smoke:
        for arm, value in merged.items():
            np.save(output / f"{arm}_features.npy", value)
    metrics = {}
    for arm in ARMS[1:]:
        per_image = np.concatenate(psnr[arm])
        metrics[arm] = dict(psnr=float(per_image.mean(dtype=np.float64)),
            psnr_std=float(per_image.std(dtype=np.float64)),
            rfid=(paired_rfid(merged[arm], merged["real"]) if not args.smoke and args.shard_count == 1 else None))
    rows = sum(int(item["rows"]) for item in route_rows)
    routes = dict(rows=rows,
        k_original_mean=sum(item["k_original_sum"] for item in route_rows)/rows,
        k_nox_mean=sum(item["k_nox_sum"] for item in route_rows)/rows,
        k_accuracy=sum(item["k_equal"] for item in route_rows)/rows,
        pooled_iou=sum(item["intersection"] for item in route_rows)/sum(item["union"] for item in route_rows),
        mean_image_iou=sum(item["image_iou_sum"] for item in route_rows)/rows,
        matched_k_pooled_iou=sum(item["matched_intersection"] for item in route_rows)/sum(item["matched_union"] for item in route_rows))
    route_totals = {key: sum(item[key] for item in route_rows) for key in route_rows[0]}
    summary = dict(status="complete", n=args.n, local_n=local_n,
        shard_rank=args.shard_rank, shard_count=args.shard_count, smoke=args.smoke, metrics=metrics, route_totals=route_totals,
        routes=routes, paired_psnr_delta_nox_minus_e117=metrics["no_xbase"]["psnr"]-metrics["e117"]["psnr"],
        paired_rfid_delta_nox_minus_e117=(None if args.smoke or args.shard_count > 1 else metrics["no_xbase"]["rfid"]-metrics["e117"]["rfid"]),
        paired_rfid_delta_matched_k_minus_e117=(None if args.smoke or args.shard_count > 1 else metrics["no_xbase_matched_k"]["rfid"]-metrics["e117"]["rfid"]),
        elapsed_seconds=time.monotonic()-started,
        interpretation="paired reconstruction of real codes; not generation FID")
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "progress.json", dict(status="complete", completed=local_n, total=local_n))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=ROOT / "h20_local_assets")
    parser.add_argument("--router-proxy-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--log-every", type=int, default=400)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    run(parser.parse_args())
