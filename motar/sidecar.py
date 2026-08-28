"""A standalone causal TiTok-1D sidecar for truly disjoint optimization.

The unified AR trunk is kept as an ordinary ``TiTokLlamaGenUnifiedAR`` model.
This module is wrapped by its own DDP instance and optimized separately, so its
parameters, gradient buckets, clipping, and AdamW state cannot perturb the 2D
trunk update.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from RandAR.model.llamagen_gpt import KVCache, RMSNorm
from RandAR.model.randar_gpt import TransformerBlock
from RandAR.model.utils import interleave_tokens


class TiTok1DDisjointSidecar(nn.Module):
    """Short prefix-only Transformer producing residual TiTok logits."""

    def __init__(self, main_model, depth: int = 4, drop_path: float = 0.0) -> None:
        super().__init__()
        if main_model.titok_conditioning != "prefix":
            raise ValueError("1D sidecar requires titok_conditioning=prefix")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")

        self.depth = int(depth)
        self.cls_token_num = int(main_model.cls_token_num)
        self.titok_num_tokens = int(main_model.titok_num_tokens)
        self.titok_vocab_size = int(main_model.titok_vocab_size)
        self.dim = int(main_model.dim)
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.causal_mask = None

        # Model construction happens on CPU. Restoring the CPU RNG means adding
        # this tower cannot change the baseline sampler or initialization stream.
        rng_state = torch.get_rng_state()
        try:
            self.layers = nn.ModuleList(
                [
                    TransformerBlock(
                        dim=main_model.dim,
                        n_layer=depth,
                        n_head=main_model.n_head,
                        n_kv_head=main_model.n_kv_head,
                        multiple_of=main_model.multiple_of,
                        ffn_dim_multiplier=main_model.ffn_dim_multiplier,
                        rope_base=main_model.rope_base,
                        norm_eps=main_model.norm_eps,
                        token_dropout_p=main_model.token_dropout_p,
                        attn_dropout_p=main_model.attn_dropout_p,
                        resid_dropout_p=main_model.resid_dropout_p,
                        ffn_dropout_p=main_model.ffn_dropout_p,
                        drop_path=drop_path,
                    )
                    for _ in range(depth)
                ]
            )
            self.norm = RMSNorm(main_model.dim, eps=main_model.norm_eps)
            self.output = nn.Linear(
                main_model.dim,
                main_model.titok_vocab_size,
                bias=False,
            )
            self.layers.apply(main_model._init_weights)
            main_model._init_weights(self.output)
            nn.init.zeros_(self.output.weight)
        finally:
            torch.set_rng_state(rng_state)

    def forward(self, prefix_h: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        expected_length = self.cls_token_num + 2 * self.titok_num_tokens
        if prefix_h.ndim != 3 or prefix_h.shape[1:] != (
            expected_length,
            self.dim,
        ):
            raise ValueError(
                f"prefix_h must have shape [B,{expected_length},{self.dim}], "
                f"got {tuple(prefix_h.shape)}"
            )
        h = prefix_h
        for layer in self.layers:
            h = layer(h, freqs_cis, None, None)
        h = self.norm(h)
        start = self.cls_token_num
        end = start + 2 * self.titok_num_tokens
        return self.output(h[:, start:end:2]).float()

    def setup_caches(self, max_batch_size: int, max_seq_length: int, dtype) -> None:
        """Allocate a cache independent from the unified AR trunk cache."""

        head_dim = self.dim // self.layers[0].attention.n_head
        self.max_batch_size = int(max_batch_size)
        self.max_seq_length = int(max_seq_length)
        for layer in self.layers:
            layer.attention.kv_cache = KVCache(
                self.max_batch_size,
                self.max_seq_length,
                layer.attention.n_kv_head,
                head_dim,
                dtype,
            )
        causal_mask = torch.tril(
            torch.ones(
                self.max_seq_length,
                self.max_seq_length,
                dtype=torch.bool,
            )
        )
        self.causal_mask = causal_mask.unsqueeze(0).repeat(
            self.max_batch_size,
            1,
            1,
        )

    def remove_caches(self) -> None:
        for layer in self.layers:
            layer.attention.kv_cache = None
        self.causal_mask = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def forward_incremental(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Return residual logits for the newest TiTok query token."""

        if self.causal_mask is None:
            raise RuntimeError("setup_caches must be called before incremental decoding")
        mask = self.causal_mask[: x.shape[0], None, input_pos]
        h = x
        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, mask)
        h = self.norm(h)
        return self.output(h[:, -1:]).float()


def detached_sidecar_inputs(main_model, z1d, cond_idx):
    """Build causal sidecar inputs without creating gradients into the trunk."""

    batch_size = z1d.shape[0]
    device = z1d.device
    cond_h = main_model.cls_embedding(
        cond_idx,
        train=main_model.training,
    )[:, : main_model.cls_token_num]
    query_h, content_h = main_model._titok_prefix(z1d)
    prefix_h = torch.cat(
        (cond_h, interleave_tokens(query_h, content_h)),
        dim=1,
    ).detach()

    main_model.freqs_cis = main_model.freqs_cis.to(device)
    cls_freqs = main_model.freqs_cis[: main_model.cls_token_num]
    cls_freqs = cls_freqs.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    titok_freqs = main_model._titok_freqs(batch_size, device)
    titok_freqs = interleave_tokens(titok_freqs, titok_freqs)
    freqs_cis = torch.cat((cls_freqs, titok_freqs), dim=1)
    return prefix_h, freqs_cis
