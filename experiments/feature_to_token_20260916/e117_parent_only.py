"""Deployment-compatible E117 ablation retaining only its frozen parent Router."""
from __future__ import annotations

import torch
from torch import nn


class E117ParentOnly(nn.Module):
    """Return E117 details with its learned spatial residual exactly removed."""

    def __init__(self, parent: nn.Module) -> None:
        super().__init__()
        self.parent = parent.eval().requires_grad_(False)

    @torch.inference_mode()
    def forward_details(self, code_ids, f_prefix, f_1d, x_base):
        base_hidden, base_tokens = self.parent.base.hidden_features(f_1d, x_base)
        base_grid = self.parent.base.grid_head(base_hidden)
        k_normalized = self.parent.base.k_head(
            self.parent.base._k_features(base_tokens, base_grid)
        ).squeeze(1).float()
        k_score = self.parent.base.margin_mean + self.parent.base.margin_scale * k_normalized
        parent_grid = (
            base_grid
            + self.parent.prefix_residual(f_prefix.detach()).to(base_grid.dtype)
            + self.parent.code_residual(base_tokens.detach(), code_ids.detach()).to(base_grid.dtype)
        )
        zeros = parent_grid.new_zeros((len(code_ids), 3, 16, 16))
        return {
            "grid": parent_grid,
            "k_score": k_score,
            "parent_grid": parent_grid,
            "grid_delta": torch.zeros_like(parent_grid),
            "boundary_logits": zeros,
        }


def install_parent_only(assets) -> dict[str, int | str]:
    source = assets.adapter.student
    parent = E117ParentOnly(source.parent).to(assets.device).eval().requires_grad_(False)
    parameters = sum(value.numel() for value in parent.parameters())
    if parameters != 3_108_318:
        raise RuntimeError(f"E117 parent parameter contract changed: {parameters}")
    assets.adapter.student = parent
    del source
    torch.cuda.empty_cache()
    return {"mode": "e117-parent-only", "parameters": parameters}
