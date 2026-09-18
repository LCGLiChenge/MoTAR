"""Optional E117 deployment ablation: route from 1D features without base RGB."""
from __future__ import annotations

import importlib
import math
from pathlib import Path
from types import MethodType

import torch
from torch import nn

from bert2d.paths import sha256


class FeatureProxy(nn.Module):
    """Predict E117's RGB-branch feature directly from the 16x16 1D latent."""

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


def install_no_xbase(assets, checkpoint: str | Path) -> dict[str, object]:
    """Swap only E117's RGB feature extractor and its RGB-producing decode."""
    path = Path(checkpoint).resolve()
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != "e117_no_xbase_feature_proxy_v1":
        raise ValueError("wrong no-x_base Router proxy checkpoint format")
    base = assets.adapter.student.parent.base
    proxy = FeatureProxy(base.latent_channels, base.hidden_dim)
    proxy.load_state_dict(state["model"], strict=True)
    proxy = proxy.to(assets.device).eval().requires_grad_(False)
    base.xbase_encoder = proxy

    def decode_without_rgb(adapter, z_quantized):
        latent = adapter.tokenizer_model.latent_decoder
        if getattr(latent, "head_mode", None) != "feature":
            raise ValueError("E117 needs feature-mode latent decoder")
        prefix = latent.forward_backbone(z_quantized)
        f1d = latent.lg_latent_head(prefix)
        proxy.source = f1d
        placeholder = torch.empty((len(f1d), 3, 256, 256), device=f1d.device,
                                  dtype=f1d.dtype)
        return prefix, f1d, placeholder

    assets.adapter._decode_1d_bundle = MethodType(decode_without_rgb, assets.adapter)
    audit = dict(mode="e117-no-x_base", proxy_checkpoint=str(path),
                 proxy_sha256=sha256(path),
                 proxy_parameters=sum(p.numel() for p in proxy.parameters()),
                 teacher_router="E117 EMA")
    if "k_threshold" in state:
        threshold = float(state["k_threshold"])
        if not math.isfinite(threshold):
            raise ValueError("nonfinite calibrated Router threshold")
        e117 = importlib.import_module(assets.adapter.__class__.__module__)

        @torch.inference_mode()
        def forward_with_budget(adapter, code_ids, return_1d_features=False):
            ids = code_ids.to(adapter._device(), dtype=torch.long)
            quantized = adapter._quantized_from_codes(ids)
            prefix, f1d, xbase = adapter._decode_1d_bundle(quantized)
            details = adapter.student.forward_details(ids, prefix, f1d, xbase)
            counts = e117.selected_tokens_e117(details, threshold=threshold)
            scores = e117.spatial_score_e117(details, counts)
            mask, indices, valid = e117.E84ARDecisionAdapter._selection_from_scores(scores, counts)
            result = dict(grid_scores=scores, k_score=details["k_score"],
                          token_counts=counts, selected_mask=mask,
                          selected_indices_padded=indices, selected_indices_valid=valid,
                          threshold=threshold, gamma=e117.SELECTED_GAMMA,
                          student_forwards=1, old_router_forwards=0,
                          source_image_used=False, f_2d_used=False,
                          probe_reconstructions=0, batch_statistics_used=False,
                          extra_route_fields_used=False)
            if return_1d_features:
                result.update(z_1d=quantized, f_prefix=prefix, f_1d=f1d, x_base=xbase)
            return result

        assets.adapter.forward = MethodType(forward_with_budget, assets.adapter)
        audit.update(budget_threshold=threshold,
                     budget_calibration=state.get("budget_calibration"))
    return audit
