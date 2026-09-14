"""Unified MaskGIT wrapper with a full-canvas sparse 2D stage.

This is the conservative unified variant suggested by the successful
2D-only experiments:

* the 1D branch keeps the official TiTok MaskGIT/ImageBERT initialization;
* the 2D branch keeps the Halton/LlamaGen 16x16 MaskGIT initialization;
* the 2D branch runs on a full 16x16 canvas internally and returns logits only
  for Router-selected sparse positions.

The class intentionally does not concatenate 1D tokens and sparse 2D tokens
into one sequence.  It is "unified" at the objective/checkpoint/sampling
pipeline level while preserving modality-specific geometry.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from experiments.maskgit_optimization_sweep_20260910.common import ASSETS
from experiments.maskgit_optimization_sweep_20260910.portable_base_features import PortableBaseFeatures
from experiments.maskgit_optimization_sweep_20260910.halton_sparse2d_adapter import (
    HALTON_BASE_CKPT,
    HaltonFullContextSparse2DAdapter,
    load_halton_base_state,
)


class FullContextUnifiedMaskGIT(nn.Module):
    """Factorized unified 1D+2D MaskGIT.

    ``one_d`` must provide the TiTok-compatible methods/properties used by
    ``h20.model.TiTokSparseImageBert``.  ``two_d`` must implement the selected
    sparse 2D contract of :class:`HaltonFullContextSparse2DAdapter`.
    """

    def __init__(self, one_d: nn.Module, two_d: HaltonFullContextSparse2DAdapter) -> None:
        super().__init__()
        self.one_d = one_d
        self.two_d = two_d
        self.titok_vocab_size = int(one_d.titok_vocab_size)
        self.titok_num_tokens = int(one_d.titok_num_tokens)
        self.llamagen_vocab_size = int(two_d.llamagen_vocab_size)
        self.grid_size = int(two_d.grid_size)
        self.grid_tokens = int(two_d.grid_tokens)
        self.max_sparse_tokens = int(two_d.max_sparse_tokens)
        self.mask_token_1d = int(one_d.mask_token_1d)
        self.mask_token_2d = int(two_d.mask_token_2d)
        self.pad_token_2d = int(two_d.pad_token_2d)

    def forward_1d(
        self,
        input_tokens: torch.Tensor,
        labels: torch.Tensor,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.one_d.forward_1d(input_tokens, labels, force_drop_ids)

    def forward_2d(
        self,
        completed_1d: torch.Tensor,
        input_tokens: torch.Tensor,
        route_indices: torch.Tensor,
        route_valid: torch.Tensor,
        labels: torch.Tensor,
        force_drop_ids: torch.Tensor | None = None,
        base_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.two_d.forward_2d(
            completed_1d,
            input_tokens,
            route_indices,
            route_valid,
            labels,
            force_drop_ids,
            base_features=base_features,
        )


    def set_feature_provider(self, provider) -> None:
        self.two_d.set_feature_provider(provider)

    def _cfg_logits(self, cond: torch.Tensor, uncond: torch.Tensor, scale: float, mode: str = "standard") -> torch.Tensor:
        return self.two_d._cfg_logits(cond, uncond, scale, mode)

    def forward(self, stage: str, **kwargs) -> torch.Tensor:
        if stage == "1d":
            return self.forward_1d(**kwargs)
        if stage == "2d":
            return self.forward_2d(**kwargs)
        raise ValueError(f"unknown stage: {stage!r}")


class OneDConditionedHaltonFullContextSparse2DAdapter(HaltonFullContextSparse2DAdapter):
    """Halton full-context sparse 2D branch with a zero-ended 1D-context path.

    This keeps the successful 2D-only computation graph intact at
    initialization: the new 1D hidden-state residual is exactly zero before
    fine-tuning.  Training can then learn how much information should flow from
    the TiTok MaskGIT branch into the 2D Halton/LlamaGen branch without
    corrupting the pretrained 2D generator on step zero.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.one_d_context_norm = nn.LayerNorm(self.dim, elementwise_affine=False)
        self.one_d_context_attn = nn.MultiheadAttention(self.dim, self.transformer.layers[0].attn.heads, dropout=0.0, batch_first=True)
        self.one_d_context_out = nn.Linear(self.dim, self.dim, bias=False)
        nn.init.zeros_(self.one_d_context_out.weight)

    def one_d_context_delta(self, hidden_grid: torch.Tensor, one_d_context: torch.Tensor | None) -> torch.Tensor:
        if one_d_context is None:
            return hidden_grid.new_zeros(hidden_grid.shape)
        if one_d_context.ndim != 3 or one_d_context.shape[0] != hidden_grid.shape[0] or one_d_context.shape[2] != self.dim:
            raise ValueError("one_d_context must be [B,L,dim]")
        if not bool(torch.isfinite(one_d_context).all()):
            raise ValueError("nonfinite 1D context")
        query = self.one_d_context_norm(hidden_grid)
        memory = one_d_context.to(query.dtype)
        delta = self.one_d_context_attn(query, memory, memory, need_weights=False)[0]
        return self.one_d_context_out(delta)

    def forward_2d(
        self,
        completed_1d: torch.Tensor,
        input_tokens: torch.Tensor,
        route_indices: torch.Tensor,
        route_valid: torch.Tensor,
        labels: torch.Tensor,
        force_drop_ids: torch.Tensor | None = None,
        base_features: torch.Tensor | None = None,
        one_d_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate(completed_1d, input_tokens, route_indices, route_valid, labels)
        if base_features is None:
            if self._feature_provider is None:
                raise RuntimeError("Halton full-context sparse adapter requires a feature provider")
            base_features = self._feature_provider(completed_1d)
        batch, k = input_tokens.shape
        if force_drop_ids is None:
            force_drop_ids = torch.zeros_like(labels)
        y = torch.where(force_drop_ids.bool(), torch.full_like(labels, 1000), labels)
        cond = self.cls_emb(y)

        full_indices = torch.arange(self.grid_tokens, device=input_tokens.device).expand(batch, -1)
        full_tokens = torch.full((batch, self.grid_tokens), self.mask_token_2d, dtype=input_tokens.dtype, device=input_tokens.device)
        safe_tokens = input_tokens.clamp(0, self.mask_token_2d)
        safe_indices = route_indices.clamp(0, self.grid_tokens - 1)
        scatter_values = torch.where(route_valid, safe_tokens, torch.full_like(safe_tokens, self.mask_token_2d))
        full_tokens.scatter_(1, safe_indices, scatter_values)

        hidden_grid = self.tok_emb(full_tokens) + self.pos_emb(full_indices)
        hidden_grid = hidden_grid + self.feature_delta_full(base_features, hidden_grid).to(hidden_grid.dtype)
        hidden_grid = hidden_grid + self.one_d_context_delta(hidden_grid, one_d_context).to(hidden_grid.dtype)
        reg = self.reg_tokens.weight[None].expand(batch, -1, -1)
        hidden = torch.cat((hidden_grid, reg), dim=1)
        valid = torch.ones((batch, self.grid_tokens + 1), dtype=torch.bool, device=input_tokens.device)
        for block in self.transformer.layers:
            hidden = block(hidden, cond, valid)
        grid_hidden = self.last_norm(hidden[:, : self.grid_tokens], cond)
        selected_hidden = grid_hidden.gather(1, safe_indices[..., None].expand(-1, -1, self.dim))
        logits = self.head(selected_hidden).float()[..., : self.codebook_size]
        if logits.shape != (batch, k, self.codebook_size):
            raise RuntimeError("full-context adapter violated sparse-output contract")
        return logits


class UnifiedHaltonMixMaskGIT(FullContextUnifiedMaskGIT):
    """True unified 1D+2D MaskGIT with modality-specific expert structure.

    The model keeps one checkpoint, one optimizer namespace and one generation
    contract, but does not force TiTok-1D and LlamaGen-2D tokens through the
    same transformer blocks.  Instead, the verified TiTok MaskGIT branch
    generates/encodes the 32-token global representation and the verified
    Halton full-context branch generates selected sparse 2D tokens, with a
    trainable zero-ended residual from 1D hidden states into the 2D grid.
    """

    def __init__(
        self,
        one_d: nn.Module,
        two_d: OneDConditionedHaltonFullContextSparse2DAdapter,
        *,
        detach_1d_context: bool = False,
    ) -> None:
        super().__init__(one_d, two_d)
        self.detach_1d_context = bool(detach_1d_context)

    def encode_1d_context(
        self,
        completed_1d: torch.Tensor,
        labels: torch.Tensor,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        class_h = self.one_d._class_tokens(labels, force_drop_ids)
        one_d_h = self.one_d._one_d_embeddings(completed_1d)
        hidden = torch.cat((class_h, one_d_h), dim=1)
        valid = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        context = self.one_d._run_backbone(hidden, None, valid)
        if self.detach_1d_context:
            context = context.detach()
        return context

    def forward_2d(
        self,
        completed_1d: torch.Tensor,
        input_tokens: torch.Tensor,
        route_indices: torch.Tensor,
        route_valid: torch.Tensor,
        labels: torch.Tensor,
        force_drop_ids: torch.Tensor | None = None,
        base_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = self.encode_1d_context(completed_1d, labels, force_drop_ids)
        return self.two_d.forward_2d(
            completed_1d,
            input_tokens,
            route_indices,
            route_valid,
            labels,
            force_drop_ids,
            base_features=base_features,
            one_d_context=context,
        )


def _build_official_one_d(
    *,
    assets_root: str | Path = ASSETS,
    attention_implementation: str | None = None,
):
    from h20.assets import official_model
    from h20.model import TiTokSparseImageBert

    if attention_implementation is None:
        return official_model(assets_root)

    # ``official_model`` mirrors the upstream attention implementation by
    # default.  This override is useful for capacity probes, while still loading
    # exactly the same official tensors.
    import os
    import sys
    import torch
    os.environ.setdefault("USE_TF", "0")
    from omegaconf import OmegaConf
    from h20.model import load_official_state

    root = Path(__import__("h20").__file__).resolve().parents[1]
    third_party = root / "third_party" / "TiTok"
    if str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))
    from modeling.maskgit import ImageBert

    reference = ImageBert(OmegaConf.load(third_party / "configs/infer/TiTok/titok_l32.yaml"))
    one_d = TiTokSparseImageBert(attention_implementation=attention_implementation)
    state = torch.load(Path(assets_root) / "weights/generator_titok_l32.bin", map_location="cpu", mmap=True, weights_only=True)
    report = load_official_state(one_d, state, reference.model.config)
    return one_d, report


def _freeze_historical_sparse_2d(one_d: nn.Module) -> list[str]:
    frozen_old_2d = []
    for name, parameter in one_d.named_parameters():
        if name.startswith(("embedding_2d.", "pos_embedding_2d.", "budget_embedding.", "output_2d.")):
            parameter.requires_grad_(False)
            frozen_old_2d.append(name)
    return frozen_old_2d


def build_fullctx_unified(
    *,
    assets_root: str | Path = ASSETS,
    halton_ckpt: str | Path = HALTON_BASE_CKPT,
    device: str | torch.device = "cuda",
    feature_chunk: int = 16,
    attention_implementation: str | None = None,
) -> tuple[FullContextUnifiedMaskGIT, dict]:
    """Build and initialize the conservative unified model.

    Returns ``(model, audit)``.  The audit records both pretrained sources and
    the fact that the old sparse-sequence 2D parameters inside the TiTok wrapper
    are frozen and not used by this model.
    """

    from h20.assets import official_model
    from h20.model import TiTokSparseImageBert

    if attention_implementation is None:
        one_d, one_d_report = official_model(assets_root)
    else:
        # ``official_model`` mirrors the upstream attention implementation by
        # default.  This override is useful for capacity probes, while still
        # loading exactly the same official tensors.
        import os
        import sys
        import torch
        os.environ.setdefault("USE_TF", "0")
        from omegaconf import OmegaConf
        from h20.model import load_official_state

        root = Path(__import__("h20").__file__).resolve().parents[1]
        third_party = root / "third_party" / "TiTok"
        if str(third_party) not in sys.path:
            sys.path.insert(0, str(third_party))
        from modeling.maskgit import ImageBert

        reference = ImageBert(OmegaConf.load(third_party / "configs/infer/TiTok/titok_l32.yaml"))
        one_d = TiTokSparseImageBert(attention_implementation=attention_implementation)
        state = torch.load(Path(assets_root) / "weights/generator_titok_l32.bin", map_location="cpu", mmap=True, weights_only=True)
        one_d_report = load_official_state(one_d, state, reference.model.config)

    # TiTokSparseImageBert contains an old sparse-sequence 2D head only for
    # historical unified experiments.  This fullctx wrapper never calls it.
    frozen_old_2d = []
    for name, parameter in one_d.named_parameters():
        if name.startswith(("embedding_2d.", "pos_embedding_2d.", "budget_embedding.", "output_2d.")):
            parameter.requires_grad_(False)
            frozen_old_2d.append(name)

    two_d = HaltonFullContextSparse2DAdapter()
    two_d_report = load_halton_base_state(two_d, Path(halton_ckpt))
    provider = PortableBaseFeatures(str(device), chunk=int(feature_chunk))
    two_d.set_feature_provider(provider)

    model = FullContextUnifiedMaskGIT(one_d, two_d).to(device)
    return model, {
        "one_d": one_d_report,
        "two_d": two_d_report,
        "old_sparse_sequence_2d_frozen": frozen_old_2d,
        "feature_provider": {
            "class": provider.__class__.__module__ + "." + provider.__class__.__name__,
            "chunk": int(feature_chunk),
        },
        "model_contract": "1D TiTok MaskGIT + full 16x16 canvas sparse-output 2D MaskGIT",
    }


def build_unified_halton_mix(
    *,
    assets_root: str | Path = ASSETS,
    halton_ckpt: str | Path = HALTON_BASE_CKPT,
    device: str | torch.device = "cuda",
    feature_chunk: int = 16,
    attention_implementation: str | None = None,
    detach_1d_context: bool = False,
) -> tuple[UnifiedHaltonMixMaskGIT, dict]:
    """Build v2 unified MaskGIT by moving the 2D-only structure into unified.

    The 1D branch is initialized from official TiTok MaskGIT.  The 2D branch is
    initialized from the pretrained Halton/LlamaGen full-context MaskGIT and
    augments it with a zero-initialized 1D-hidden cross-attention residual.
    """

    one_d, one_d_report = _build_official_one_d(
        assets_root=assets_root,
        attention_implementation=attention_implementation,
    )
    frozen_old_2d = _freeze_historical_sparse_2d(one_d)
    two_d = OneDConditionedHaltonFullContextSparse2DAdapter()
    two_d_report = load_halton_base_state(two_d, Path(halton_ckpt))
    provider = PortableBaseFeatures(str(device), chunk=int(feature_chunk))
    two_d.set_feature_provider(provider)
    model = UnifiedHaltonMixMaskGIT(one_d, two_d, detach_1d_context=detach_1d_context).to(device)
    audit = {
        "one_d": one_d_report,
        "two_d": two_d_report,
        "old_sparse_sequence_2d_frozen": frozen_old_2d,
        "feature_provider": {
            "class": provider.__class__.__module__ + "." + provider.__class__.__name__,
            "chunk": int(feature_chunk),
        },
    }
    audit["model_contract"] = (
        "UnifiedHaltonMixMaskGIT: official TiTok 1D branch + Halton full-context "
        "sparse 2D branch + zero-ended 1D-hidden cross-attention coupling"
    )
    audit["variant"] = "haltonmix_v2"
    audit["detach_1d_context"] = bool(detach_1d_context)
    audit["one_d_context_residual_zero_initialized"] = bool(
        model.two_d.one_d_context_out.weight.detach().count_nonzero().item() == 0
    )
    return model, audit

