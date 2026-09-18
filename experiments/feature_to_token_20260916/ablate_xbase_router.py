"""Pilot: replace E117's RGB->16x16 branch with a direct f_1d->16x16 proxy.

The frozen E117 weights and decision rule are unchanged. Training sees the
teacher's decoded RGB only to distill its xbase_encoder feature. Inference of
the proxy arm never calls the RGB decoder.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DELIVERY = ROOT / "delivery_h20_20260910"
for path in (ROOT, DELIVERY, DELIVERY / "third_party/E117"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from h20.data import PackedOneDCodeDataset
from h20_joint.assets import FrozenAssets
from e117_ar_adapter import selected_tokens_e117, spatial_score_e117
from e84_ar_adapter import E84ARDecisionAdapter


class FeatureProxy(nn.Module):
    """Predict the E87 xbase_encoder output directly from the 16x16 latent."""

    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 1),
        )
        self.source: torch.Tensor | None = None

    def forward(self, unused_rgb: torch.Tensor) -> torch.Tensor:
        if self.source is None:
            raise RuntimeError("set the 1D feature before the Router forward")
        if unused_rgb.shape != (len(self.source), 3, 256, 256):
            raise ValueError("invalid RGB placeholder shape")
        return self.net(self.source.float())


def route(details: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    counts = selected_tokens_e117(details)
    scores = spatial_score_e117(details, counts)
    mask = E84ARDecisionAdapter._selection_from_scores(scores, counts)[0]
    return counts, mask


def proxy_forward(student: nn.Module, proxy: FeatureProxy, ids: torch.Tensor,
                  prefix: torch.Tensor, f1d: torch.Tensor) -> dict[str, torch.Tensor]:
    proxy.source = f1d
    placeholder = torch.empty((len(ids), 3, 256, 256), device=ids.device,
                              dtype=f1d.dtype)
    try:
        return student.forward_details(ids, prefix, f1d, placeholder)
    finally:
        proxy.source = None


@torch.no_grad()
def evaluate(assets: FrozenAssets, student: nn.Module, proxy: FeatureProxy,
             loader: DataLoader, batches: int) -> dict[str, float]:
    proxy.eval()
    totals = dict(rows=0, k_correct=0, overlap=0, union=0, feature_sse=0.,
                  feature_elements=0)
    for i, batch in enumerate(loader):
        if i >= batches:
            break
        ids = batch["z1d"].cuda(non_blocking=True)
        q = assets.adapter._quantized_from_codes(ids)
        prefix, f1d, rgb = assets.adapter._decode_1d_bundle(q)
        truth = assets.adapter.student.forward_details(ids, prefix, f1d, rgb)
        target = assets.adapter.student.parent.base.xbase_encoder(rgb)
        estimate = proxy.net(f1d.float())
        pred = proxy_forward(student, proxy, ids, prefix, f1d)
        kt, mt = route(truth)
        kp, mp = route(pred)
        totals["rows"] += len(ids)
        totals["k_correct"] += int((kt == kp).sum())
        totals["overlap"] += int((mt & mp).sum())
        totals["union"] += int((mt | mp).sum())
        totals["feature_sse"] += float((estimate.float()-target.float()).square().sum())
        totals["feature_elements"] += target.numel()
    if totals["rows"] == 0:
        raise RuntimeError("empty held-out evaluation")
    return dict(rows=totals["rows"], k_accuracy=totals["k_correct"]/totals["rows"],
                pooled_route_iou=totals["overlap"]/max(1, totals["union"]),
                feature_mse=totals["feature_sse"]/totals["feature_elements"])


@torch.no_grad()
def benchmark(assets: FrozenAssets, student: nn.Module, proxy: FeatureProxy,
              ids: torch.Tensor, repeats: int) -> dict[str, float]:
    q = assets.adapter._quantized_from_codes(ids)
    latent = assets.shell.latent_decoder
    for _ in range(5):
        prefix = latent.forward_backbone(q)
        f1d = latent.lg_latent_head(prefix)
        rgb = assets.shell.llamagen_vq.decoder(f1d)
        assets.adapter.student.forward_details(ids, prefix, f1d, rgb)
        proxy_forward(student, proxy, ids, prefix, f1d)
    torch.cuda.synchronize()
    def timed(fn):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record(); torch.cuda.synchronize()
        return start.elapsed_time(end)/repeats
    prefix = latent.forward_backbone(q)
    f1d = latent.lg_latent_head(prefix)
    rgb = assets.shell.llamagen_vq.decoder(f1d)
    return dict(batch=len(ids), rgb_decoder_ms=timed(lambda: assets.shell.llamagen_vq.decoder(f1d)),
                original_router_ms=timed(lambda: assets.adapter.student.forward_details(ids,prefix,f1d,rgb)),
                proxy_router_ms=timed(lambda: proxy_forward(student,proxy,ids,prefix,f1d)),
                repeats=repeats)


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("select exactly one GPU for this pilot")
    if args.steps < 0 or args.batch_size < 1 or args.eval_batches < 1:
        raise ValueError("invalid experiment size")
    output = args.output.resolve()
    allowed = Path("/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results").resolve()
    if not output.is_relative_to(allowed) or output.exists():
        raise ValueError("output must be a new named results directory")
    output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    assets = FrozenAssets(args.assets_root, device="cuda", chunk=args.batch_size)
    teacher = assets.adapter.student
    student = copy.deepcopy(teacher).eval().requires_grad_(False)
    base = student.parent.base
    proxy = FeatureProxy(base.latent_channels, base.hidden_dim).cuda()
    base.xbase_encoder = proxy
    train = PackedOneDCodeDataset(args.assets_root / "codes/train", "train")
    heldout = PackedOneDCodeDataset(args.assets_root / "codes/train", "eval")
    if set(train.source_indices) & set(heldout.source_indices):
        raise RuntimeError("train/eval source leakage")
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    eval_loader = DataLoader(heldout, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    config = vars(args).copy()
    config.update(assets_root=str(args.assets_root), output=str(output),
                  train_sources=len(train.source_indices),
                  eval_sources=len(heldout.source_indices),
                  proxy_parameters=sum(p.numel() for p in proxy.parameters()),
                  split_seed=20260901, eval_fraction=0.002,
                  teacher="E117 EMA; all weights frozen",
                  objective="MSE of E87 xbase_encoder 16x16 features")
    (output / "config.json").write_text(json.dumps(config, indent=2))
    before = evaluate(assets, student, proxy, eval_loader, args.eval_batches)
    (output / "before.json").write_text(json.dumps(before, indent=2))
    print("before", json.dumps(before), flush=True)
    optimizer = torch.optim.AdamW(proxy.parameters(), lr=args.lr, weight_decay=0.01)
    stream = iter(train_loader)
    started = time.monotonic()
    for step in range(1, args.steps+1):
        try:
            batch = next(stream)
        except StopIteration:
            stream = iter(train_loader)
            batch = next(stream)
        ids = batch["z1d"].cuda(non_blocking=True)
        with torch.no_grad():
            q = assets.adapter._quantized_from_codes(ids)
            _, f1d, rgb = assets.adapter._decode_1d_bundle(q)
            target = teacher.parent.base.xbase_encoder(rgb).detach()
        proxy.train()
        estimate = proxy.net(f1d.float())
        loss = F.mse_loss(estimate.float(), target.float())
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite proxy loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(proxy.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = dict(step=step, feature_mse=float(loss.detach()),
                       elapsed_seconds=time.monotonic()-started,
                       peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
            (output / "progress.json").write_text(json.dumps(row, indent=2))
            print(json.dumps(row), flush=True)
    after = evaluate(assets, student, proxy, eval_loader, args.eval_batches)
    ids = next(iter(eval_loader))["z1d"].cuda()[:min(args.batch_size, 8)]
    timing = benchmark(assets, student, proxy, ids, args.benchmark_repeats)
    summary = dict(before=before, after=after, timing=timing,
                   elapsed_seconds=time.monotonic()-started)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    if args.steps and not args.smoke:
        torch.save(dict(format="e117_no_xbase_feature_proxy_v1", model=proxy.state_dict(),
                        config=config, heldout=after), output / "latest.pt")
    print("summary", json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=ROOT / "h20_local_assets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--benchmark-repeats", type=int, default=30)
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args())
