"""Sparse selected-token adapter for pretrained Halton/LlamaGen MaskGIT.

The external checkpoint is a class-to-image MaskGIT trained on 16x16 LlamaGen
VQ ids.  This module keeps its DiT-like transformer and gathers positional
embeddings for Router-selected sparse cells, returning logits only for those
selected cells.  A zero-ended full-grid 1D feature residual can then be trained
without corrupting the pretrained logits at initialization.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


HALTON_BASE_CKPT = Path(
    "/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/external_weights/"
    "halton_maskgit/ImageNet_256_base.pth"
)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, linear: bool = True, bias: bool = True):
        super().__init__()
        self.eps = eps
        if linear:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        y = y.to(x.dtype)
        if self.weight is not None:
            y = y * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


class HaltonAttention(nn.Module):
    def __init__(self, dim: int = 768, heads: int = 12, dropout: float = 0.1):
        super().__init__()
        if dim % heads:
            raise ValueError("hidden dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.query_norm = RMSNorm(dim, linear=False, bias=False)
        self.key_norm = RMSNorm(dim, linear=False, bias=False)

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = self.query_norm(q).to(v.dtype).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.key_norm(k).to(v.dtype).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        mask = None
        if valid is not None:
            if valid.shape != (b, n):
                raise ValueError("attention valid mask shape mismatch")
            mask = torch.zeros((b, 1, 1, n), dtype=x.dtype, device=x.device)
            mask.masked_fill_(~valid[:, None, None, :], torch.finfo(x.dtype).min)
        y = F.scaled_dot_product_attention(q, k, v, mask, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(b, n, -1)
        y = self.wo(y)
        if self.dropout > 0.0 and self.training:
            y = F.dropout(y, self.dropout)
        return y


class HaltonFeedForward(nn.Module):
    def __init__(self, dim: int = 768, mlp_dim: int = 3072, dropout: float = 0.1):
        super().__init__()
        hidden = 256 * ((int(2 * mlp_dim / 3) + 255) // 256)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.silu(self.w1(x)) * self.w3(x)
        if self.dropout > 0.0 and self.training:
            y = F.dropout(y, self.dropout)
        return self.w2(y)


class HaltonBlock(nn.Module):
    def __init__(self, dim: int = 768, heads: int = 12, mlp_dim: int = 3072, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.ln1 = RMSNorm(dim, linear=True, bias=False)
        self.attn = HaltonAttention(dim, heads, dropout)
        self.ln2 = RMSNorm(dim, linear=True, bias=False)
        self.ff = HaltonFeedForward(dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.mlp(cond).chunk(6, dim=1)
        x = x + alpha1.unsqueeze(1) * self.attn(modulate(self.ln1(x), gamma1, beta1), valid)
        x = x + alpha2.unsqueeze(1) * self.ff(modulate(self.ln2(x), gamma2, beta2))
        if valid is not None:
            x = x * valid[..., None].to(x.dtype)
        return x


class AdaNorm(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.norm_final = RMSNorm(dim, linear=True, bias=True)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 2))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.mlp(cond).chunk(2, dim=1)
        return modulate(self.norm_final(x), shift, scale)


class HaltonSparse2DAdapter(nn.Module):
    """Pretrained Halton MaskGIT restricted to selected sparse positions."""

    codebook_size = 16384
    mask_token_2d = 16384
    pad_token_2d = 16384
    grid_size = 16
    grid_tokens = 256
    titok_num_tokens = 32
    max_sparse_tokens = 128
    llamagen_vocab_size = 16384

    def __init__(self, *, dim: int = 768, depth: int = 12, heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.cls_emb = nn.Embedding(1001, dim)
        self.tok_emb = nn.Embedding(self.codebook_size + 1, dim)
        self.pos_emb = nn.Embedding(self.grid_tokens, dim)
        self.reg_tokens = nn.Embedding(1, dim)
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList(
            HaltonBlock(dim=dim, heads=heads, mlp_dim=dim * 4, dropout=dropout)
            for _ in range(depth)
        )
        self.last_norm = AdaNorm(dim)
        self.head = nn.Linear(dim, self.codebook_size + 1)
        self.head.weight = self.tok_emb.weight
        self.feature_norm = nn.LayerNorm(256, elementwise_affine=False)
        self.feature_to_sparse = nn.Sequential(
            nn.Linear(256, dim),
            nn.GELU(),
            nn.Linear(dim, dim, bias=False),
        )
        nn.init.zeros_(self.feature_to_sparse[-1].weight)
        self.feature_memory_norm = nn.LayerNorm(256, elementwise_affine=False)
        self.feature_memory_projection = nn.Linear(256, dim)
        self.feature_memory_position = nn.Embedding(self.grid_tokens, dim)
        self.feature_cross_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.feature_cross_attn = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.feature_cross_out = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.feature_cross_out.weight)
        object.__setattr__(self, "_feature_provider", None)

    def set_feature_provider(self, provider) -> None:
        object.__setattr__(self, "_feature_provider", provider)

    def _validate(self, completed_1d, input_tokens, route_indices, route_valid, labels):
        if completed_1d.shape != (len(input_tokens), self.titok_num_tokens):
            raise ValueError("completed_1d must be [B,32] for feature lookup")
        if input_tokens.shape != route_indices.shape or input_tokens.shape != route_valid.shape:
            raise ValueError("sparse token/index/valid shapes must match")
        if input_tokens.ndim != 2 or input_tokens.shape[1] > self.max_sparse_tokens:
            raise ValueError("expected [B,K<=128] sparse selected tokens")
        if route_indices.dtype != torch.long or route_valid.dtype != torch.bool:
            raise ValueError("route index/valid dtype mismatch")
        if labels.shape != (len(input_tokens),):
            raise ValueError("labels must be [B]")
        if bool(((input_tokens[route_valid] < 0) | (input_tokens[route_valid] > self.mask_token_2d)).any()):
            raise ValueError("2D token outside content+mask range")
        if bool(((route_indices[route_valid] < 0) | (route_indices[route_valid] >= self.grid_tokens)).any()):
            raise ValueError("valid route index outside 16x16 grid")

    def _flat_features(self, base_features: torch.Tensor, batch: int) -> torch.Tensor:
        if base_features.shape != (batch, 256, self.grid_size, self.grid_size):
            raise ValueError("expected base_features [B,256,16,16]")
        if base_features.dtype != torch.float32 or not bool(torch.isfinite(base_features).all()):
            raise ValueError("expected finite FP32 frozen 1D features")
        return base_features.detach().flatten(2).transpose(1, 2)

    def full_feature_memory(self, base_features: torch.Tensor, batch: int) -> torch.Tensor:
        flat = self._flat_features(base_features, batch)
        positions = torch.arange(self.grid_tokens, device=flat.device)
        return self.feature_memory_projection(self.feature_memory_norm(flat)) + self.feature_memory_position(positions)[None]

    def feature_delta(self, base_features: torch.Tensor, route_indices: torch.Tensor, route_valid: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        flat = self._flat_features(base_features, len(route_indices))
        selected = flat.gather(1, route_indices.clamp(0, 255)[..., None].expand(-1, -1, 256))
        local_delta = self.feature_to_sparse(self.feature_norm(selected))
        memory = self.full_feature_memory(base_features, len(route_indices)).to(sparse.dtype)
        query = self.feature_cross_norm(sparse)
        global_delta = self.feature_cross_attn(query, memory, memory, need_weights=False)[0]
        global_delta = self.feature_cross_out(global_delta)
        delta = local_delta.to(global_delta.dtype) + global_delta
        return delta * route_valid[..., None].to(delta.dtype)

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
        self._validate(completed_1d, input_tokens, route_indices, route_valid, labels)
        if base_features is None:
            if self._feature_provider is None:
                raise RuntimeError("Halton sparse adapter requires a feature provider")
            base_features = self._feature_provider(completed_1d)
        safe_tokens = input_tokens.clamp(0, self.mask_token_2d)
        safe_indices = route_indices.clamp(0, self.grid_tokens - 1)
        if force_drop_ids is None:
            force_drop_ids = torch.zeros_like(labels)
        y = torch.where(force_drop_ids.bool(), torch.full_like(labels, 1000), labels)
        cond = self.cls_emb(y)
        sparse = self.tok_emb(safe_tokens) + self.pos_emb(safe_indices)
        sparse = sparse + self.feature_delta(base_features, safe_indices, route_valid, sparse).to(sparse.dtype)
        reg = self.reg_tokens.weight[None].expand(len(labels), -1, -1)
        hidden = torch.cat((sparse, reg), dim=1)
        valid = torch.cat(
            (
                route_valid,
                torch.ones((len(labels), 1), dtype=torch.bool, device=route_valid.device),
            ),
            dim=1,
        )
        for block in self.transformer.layers:
            hidden = block(hidden, cond, valid)
        hidden = self.last_norm(hidden[:, : input_tokens.shape[1]], cond)
        logits = self.head(hidden).float()[..., : self.codebook_size]
        return logits

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.forward_2d(*args, **kwargs)

    def _cfg_logits(self, cond: torch.Tensor, uncond: torch.Tensor, scale: float, mode: str = "standard") -> torch.Tensor:
        if mode != "standard" or cond.shape != uncond.shape:
            raise ValueError("Halton adapter supports standard CFG with matching logits")
        return uncond + float(scale) * (cond - uncond)


class HaltonFullContextSparse2DAdapter(HaltonSparse2DAdapter):
    """Run the pretrained 16x16 MaskGIT context, but return selected logits only.

    This variant still generates only Router-selected sparse tokens.  The
    difference from :class:`HaltonSparse2DAdapter` is that the transformer sees a
    full 256-position 16x16 sequence, matching the Halton/LlamaGen-token
    MaskGIT pretraining layout.  Non-selected positions are fixed context MASK
    tokens with frozen 1D feature residuals; they are never sampled and never
    contribute to the loss.
    """

    def feature_delta_full(self, base_features: torch.Tensor, full_sparse: torch.Tensor) -> torch.Tensor:
        batch = len(full_sparse)
        flat = self._flat_features(base_features, batch)
        local_delta = self.feature_to_sparse(self.feature_norm(flat))
        memory = self.full_feature_memory(base_features, batch).to(full_sparse.dtype)
        query = self.feature_cross_norm(full_sparse)
        global_delta = self.feature_cross_attn(query, memory, memory, need_weights=False)[0]
        global_delta = self.feature_cross_out(global_delta)
        return local_delta.to(global_delta.dtype) + global_delta

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


class HaltonBaseContextSparse2DAdapter(HaltonFullContextSparse2DAdapter):
    """Full 16x16 context with a separate non-selected base-context embedding.

    Selected positions still use regular content/MASK 2D token embeddings and
    are the only positions returned by ``forward_2d``.  Non-selected positions
    are never sampled and never enter the loss; their hidden state is a learned
    base-context marker plus frozen 1D feature residual, avoiding the semantic
    mismatch of treating base-only context cells as unknown 2D MASK targets.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_context_token = nn.Parameter(torch.zeros(self.dim))

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
        self._validate(completed_1d, input_tokens, route_indices, route_valid, labels)
        if base_features is None:
            if self._feature_provider is None:
                raise RuntimeError("Halton base-context sparse adapter requires a feature provider")
            base_features = self._feature_provider(completed_1d)
        batch, k = input_tokens.shape
        if force_drop_ids is None:
            force_drop_ids = torch.zeros_like(labels)
        y = torch.where(force_drop_ids.bool(), torch.full_like(labels, 1000), labels)
        cond = self.cls_emb(y)

        full_indices = torch.arange(self.grid_tokens, device=input_tokens.device).expand(batch, -1)
        safe_tokens = input_tokens.clamp(0, self.mask_token_2d)
        safe_indices = route_indices.clamp(0, self.grid_tokens - 1)
        selected = torch.zeros((batch, self.grid_tokens), dtype=torch.bool, device=input_tokens.device)
        selected.scatter_(1, safe_indices, route_valid)
        selected_values = torch.full((batch, self.grid_tokens), self.mask_token_2d, dtype=input_tokens.dtype, device=input_tokens.device)
        scatter_values = torch.where(route_valid, safe_tokens, torch.full_like(safe_tokens, self.mask_token_2d))
        selected_values.scatter_(1, safe_indices, scatter_values)

        selected_h = self.tok_emb(selected_values) + self.pos_emb(full_indices)
        context_h = self.base_context_token[None, None, :].to(selected_h.dtype) + self.pos_emb(full_indices)
        hidden_grid = torch.where(selected[..., None], selected_h, context_h)
        hidden_grid = hidden_grid + self.feature_delta_full(base_features, hidden_grid).to(hidden_grid.dtype)
        reg = self.reg_tokens.weight[None].expand(batch, -1, -1)
        hidden = torch.cat((hidden_grid, reg), dim=1)
        valid = torch.ones((batch, self.grid_tokens + 1), dtype=torch.bool, device=input_tokens.device)
        for block in self.transformer.layers:
            hidden = block(hidden, cond, valid)
        grid_hidden = self.last_norm(hidden[:, : self.grid_tokens], cond)
        selected_hidden = grid_hidden.gather(1, safe_indices[..., None].expand(-1, -1, self.dim))
        logits = self.head(selected_hidden).float()[..., : self.codebook_size]
        if logits.shape != (batch, k, self.codebook_size):
            raise RuntimeError("base-context adapter violated sparse-output contract")
        return logits


def load_halton_base_state(
    model: HaltonSparse2DAdapter,
    checkpoint: Path = HALTON_BASE_CKPT,
) -> dict:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = ckpt["model_state_dict"]
    target = model.state_dict()
    expected_new = {
        name
        for name in target
        if name.startswith((
            "feature_to_sparse.",
            "feature_memory_projection.",
            "feature_memory_position.",
            "feature_cross_attn.",
            "feature_cross_out.",
            "base_context_token",
        ))
    }
    mapped = {name: value for name, value in state.items() if name in target}
    if set(target) - set(mapped) != expected_new:
        missing = sorted(set(target) - set(mapped) - expected_new)
        extra_new = sorted(expected_new - (set(target) - set(mapped)))
        raise ValueError(f"unexpected Halton mapping; missing={missing}, extra_new={extra_new}")
    result = model.load_state_dict(mapped, strict=False)
    if set(result.missing_keys) != expected_new or result.unexpected_keys:
        raise ValueError("Halton load_state_dict accounting mismatch")
    return dict(
        checkpoint=str(checkpoint),
        iter=int(ckpt.get("iter", -1)),
        global_epoch=int(ckpt.get("global_epoch", -1)),
        loaded_tensors=len(mapped),
        fresh_tensors=sorted(expected_new),
        codebook_size=model.codebook_size,
        sparse_output_only=True,
        feature_branch_zero_initialized=bool(
            model.feature_to_sparse[-1].weight.detach().count_nonzero().item() == 0
            and model.feature_cross_out.weight.detach().count_nonzero().item() == 0
        ),
    )


def halton_sparse2d_contract() -> dict:
    return dict(
        source="llvictorll/Halton-MaskGIT ImageNet_256_base.pth",
        output="selected sparse positions only, shape [B,K,16384]",
        decoder_contract="unchanged LlamaGen VQ-16 selected ids",
        feature_condition="zero-ended local selected feature residual plus full-grid feature cross-attention",
        variants=("sparse_sequence", "full_16x16_context", "full_16x16_base_context"),
    )
