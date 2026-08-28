
"""Unified TiTok-L32 1D + MoT/LlamaGen VQ-16 2D autoregressive model."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

RANDAR_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "RandAR"
if str(RANDAR_ROOT) not in sys.path:
    sys.path.insert(0, str(RANDAR_ROOT))

from RandAR.model.randar_gpt import RandARTransformer
from RandAR.model.utils import interleave_tokens


class TiTokLlamaGenUnifiedAR(RandARTransformer):
    """RandAR/LlamaGen 2D AR with a 32-token TiTok prefix.

    The loaded RandAR path is intentionally left intact for the 2D tokens:
    ``tok_embeddings``, ``pos_instruct_embeddings``, 2D RoPE, transformer blocks,
    and ``output`` all keep the original key names, so official RandAR weights can
    be loaded with ``strict=False`` and only the TiTok-specific modules are missing.
    """

    def __init__(
        self,
        titok_vocab_size=4096,
        titok_num_tokens=32,
        loss_1d_weight=0.0,
        loss_2d_weight=1.0,
        titok_conditioning="prefix",
        titok_adapter_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.titok_vocab_size = titok_vocab_size
        self.titok_num_tokens = titok_num_tokens
        self.loss_1d_weight = loss_1d_weight
        self.loss_2d_weight = loss_2d_weight
        self.titok_conditioning = titok_conditioning
        if self.titok_conditioning not in {"prefix", "adapter"}:
            raise ValueError(f"Unsupported titok_conditioning: {self.titok_conditioning}")

        self.titok_embeddings = nn.Embedding(titok_vocab_size, self.dim)
        self.titok_pos_embeddings = nn.Embedding(titok_num_tokens, self.dim)
        self.titok_query_embedding = nn.Parameter(torch.randn(1, self.dim) * self.initializer_range)
        self.titok_output = nn.Linear(self.dim, titok_vocab_size, bias=False)

        # Zero-init keeps the loaded RandAR 2D path unchanged at initialization.
        self.modality_embeddings = nn.Embedding(2, self.dim)
        self.titok_adapter_query_norm = nn.LayerNorm(self.dim)
        self.titok_adapter_context_norm = nn.LayerNorm(self.dim)
        self.titok_adapter_attn = nn.MultiheadAttention(
            self.dim,
            self.n_head,
            dropout=titok_adapter_dropout,
            batch_first=True,
        )
        self.titok_adapter_out = nn.Linear(self.dim, self.dim, bias=False)
        self._init_weights(self.titok_embeddings)
        self._init_weights(self.titok_pos_embeddings)
        self._init_weights(self.titok_output)
        nn.init.constant_(self.modality_embeddings.weight, 0.0)
        nn.init.constant_(self.titok_adapter_out.weight, 0.0)
        self._freeze_inactive_conditioning_params()

    def _freeze_inactive_conditioning_params(self):
        if self.titok_conditioning == "adapter":
            self.titok_query_embedding.requires_grad_(False)
            self.modality_embeddings.weight.requires_grad_(False)
            return

        for module in (
            self.titok_adapter_query_norm,
            self.titok_adapter_context_norm,
            self.titok_adapter_attn,
            self.titok_adapter_out,
        ):
            for param in module.parameters():
                param.requires_grad_(False)

    def _make_token_order(self, batch_size, device):
        if self.position_order == "random":
            token_order = torch.arange(self.block_size, device=device, dtype=torch.long)
            token_order = token_order.unsqueeze(0).repeat(batch_size, 1)
            for i in range(batch_size):
                token_order[i] = token_order[i][torch.randperm(self.block_size, device=device)]
            return token_order.contiguous()
        if self.position_order == "raster":
            return torch.arange(self.block_size, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
        raise ValueError(f"Invalid position_order: {self.position_order}")

    def _titok_prefix(self, z1d):
        bs = z1d.shape[0]
        pos = torch.arange(self.titok_num_tokens, device=z1d.device)
        pos_h = self.titok_pos_embeddings(pos).unsqueeze(0).expand(bs, -1, -1)
        query_h = self.titok_query_embedding.view(1, 1, -1).expand(bs, self.titok_num_tokens, -1)
        query_h = query_h + pos_h + self.modality_embeddings.weight[0].view(1, 1, -1)
        content_h = self.titok_embeddings(z1d) + pos_h + self.modality_embeddings.weight[0].view(1, 1, -1)
        return query_h, content_h

    def _titok_context(self, z1d):
        bs = z1d.shape[0]
        pos = torch.arange(self.titok_num_tokens, device=z1d.device)
        pos_h = self.titok_pos_embeddings(pos).unsqueeze(0).expand(bs, -1, -1)
        return self.titok_embeddings(z1d) + pos_h

    def _apply_titok_adapter(self, h, z1d):
        context = self._titok_context(z1d)
        adapter_h = self.titok_adapter_attn(
            self.titok_adapter_query_norm(h),
            self.titok_adapter_context_norm(context),
            self.titok_adapter_context_norm(context),
            need_weights=False,
        )[0]
        return h + self.titok_adapter_out(adapter_h)

    def _titok_aux_logits(self, z1d):
        return self.titok_output(self._titok_context(z1d)).float()

    def _titok_freqs(self, batch_size, device):
        self.freqs_cis = self.freqs_cis.to(device)
        spatial_freqs = self.freqs_cis[self.cls_token_num : self.cls_token_num + self.titok_num_tokens]
        if spatial_freqs.shape[0] < self.titok_num_tokens:
            pad = spatial_freqs[-1:].repeat(self.titok_num_tokens - spatial_freqs.shape[0], 1, 1)
            spatial_freqs = torch.cat([spatial_freqs, pad], dim=0)
        return spatial_freqs.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    def forward(
        self,
        z1d,
        z2d,
        cond_idx,
        token_order=None,
        targets_1d=None,
        targets_2d=None,
        input_pos=None,
        mask=None,
    ):
        if self.titok_conditioning == "adapter":
            return self.forward_adapter(
                z1d=z1d,
                z2d=z2d,
                cond_idx=cond_idx,
                token_order=token_order,
                targets_1d=targets_1d,
                targets_2d=targets_2d,
                input_pos=input_pos,
                mask=mask,
            )

        if z1d.ndim != 2 or z1d.shape[1] != self.titok_num_tokens:
            raise ValueError(f"z1d must have shape [B, {self.titok_num_tokens}], got {tuple(z1d.shape)}")
        if z2d.ndim != 2 or z2d.shape[1] != self.block_size:
            raise ValueError(f"z2d must have shape [B, {self.block_size}], got {tuple(z2d.shape)}")

        bs = z1d.shape[0]
        device = z1d.device
        if targets_1d is None:
            targets_1d = z1d
        if targets_2d is None:
            targets_2d = z2d
        if token_order is None:
            token_order = self._make_token_order(bs, device)

        z2d_ordered = torch.gather(z2d.unsqueeze(-1), 1, token_order.unsqueeze(-1)).squeeze(-1).contiguous()
        targets_2d_ordered = torch.gather(targets_2d.unsqueeze(-1), 1, token_order.unsqueeze(-1)).squeeze(-1).contiguous()

        self.freqs_cis = self.freqs_cis.to(device)
        cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:, : self.cls_token_num]

        titok_query_h, titok_content_h = self._titok_prefix(z1d)
        titok_h = interleave_tokens(titok_query_h, titok_content_h)

        pos_query_h = self.get_position_instruction_tokens(token_order)
        tok_content_h = self.tok_dropout(self.tok_embeddings(z2d_ordered))
        tok_content_h = tok_content_h + self.modality_embeddings.weight[1].view(1, 1, -1)
        pos_query_h = pos_query_h + self.modality_embeddings.weight[1].view(1, 1, -1)
        token_h = interleave_tokens(pos_query_h, tok_content_h)

        h = torch.cat((cond_embeddings, titok_h, token_h), dim=1)

        cls_freqs = self.freqs_cis[: self.cls_token_num].unsqueeze(0).repeat(bs, 1, 1, 1)
        titok_freqs = self._titok_freqs(bs, device)
        titok_freqs = interleave_tokens(titok_freqs, titok_freqs)
        token_freqs = self.freqs_cis[self.cls_token_num :].clone().to(device)[token_order]
        token_freqs = interleave_tokens(token_freqs, token_freqs)
        freqs_cis = torch.cat((cls_freqs, titok_freqs, token_freqs), dim=1)

        for layer in self.layers:
            if self.grad_checkpointing:
                h = checkpoint(layer, h, freqs_cis, input_pos, mask, use_reentrant=False)
            else:
                h = layer(h, freqs_cis, input_pos, mask)

        h = self.norm(h)
        one_d_start = self.cls_token_num
        one_d_end = one_d_start + self.titok_num_tokens * 2
        two_d_start = one_d_end

        logits_1d = self.titok_output(h[:, one_d_start:one_d_end:2]).float()
        logits_2d = self.output(h[:, two_d_start::2]).float()

        loss_1d = F.cross_entropy(logits_1d.reshape(-1, logits_1d.size(-1)), targets_1d.reshape(-1))
        loss_2d = F.cross_entropy(logits_2d.reshape(-1, logits_2d.size(-1)), targets_2d_ordered.reshape(-1))
        loss = self.loss_1d_weight * loss_1d + self.loss_2d_weight * loss_2d

        with torch.no_grad():
            acc_1d = (logits_1d.argmax(dim=-1) == targets_1d).float().mean()
            acc_2d = (logits_2d.argmax(dim=-1) == targets_2d_ordered).float().mean()

        return {
            "logits_1d": logits_1d,
            "logits_2d": logits_2d,
            "loss": loss,
            "loss_1d": loss_1d.detach(),
            "loss_2d": loss_2d.detach(),
            "acc_1d": acc_1d.detach(),
            "acc_2d": acc_2d.detach(),
            "token_order": token_order,
        }


    def forward_adapter(
        self,
        z1d,
        z2d,
        cond_idx,
        token_order=None,
        targets_1d=None,
        targets_2d=None,
        input_pos=None,
        mask=None,
    ):
        if z1d.ndim != 2 or z1d.shape[1] != self.titok_num_tokens:
            raise ValueError(f"z1d must have shape [B, {self.titok_num_tokens}], got {tuple(z1d.shape)}")
        if z2d.ndim != 2 or z2d.shape[1] != self.block_size:
            raise ValueError(f"z2d must have shape [B, {self.block_size}], got {tuple(z2d.shape)}")

        bs = z2d.shape[0]
        device = z2d.device
        if targets_1d is None:
            targets_1d = z1d
        if targets_2d is None:
            targets_2d = z2d
        if token_order is None:
            token_order = self._make_token_order(bs, device)

        z2d_ordered = torch.gather(z2d.unsqueeze(-1), 1, token_order.unsqueeze(-1)).squeeze(-1).contiguous()
        targets_2d_ordered = torch.gather(targets_2d.unsqueeze(-1), 1, token_order.unsqueeze(-1)).squeeze(-1).contiguous()

        self.freqs_cis = self.freqs_cis.to(device)
        cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:, : self.cls_token_num]
        token_embeddings = self.tok_dropout(self.tok_embeddings(z2d_ordered))
        position_instruction_tokens = self.get_position_instruction_tokens(token_order)
        h = torch.cat((cond_embeddings, interleave_tokens(position_instruction_tokens, token_embeddings)), dim=1)
        h = self._apply_titok_adapter(h, z1d)

        token_freqs_cis = self.freqs_cis[self.cls_token_num :].clone().to(device)[token_order]
        freqs_cis = torch.cat(
            (
                self.freqs_cis[: self.cls_token_num].unsqueeze(0).repeat(bs, 1, 1, 1),
                interleave_tokens(token_freqs_cis, token_freqs_cis),
            ),
            dim=1,
        )

        for layer in self.layers:
            if self.grad_checkpointing:
                h = checkpoint(layer, h, freqs_cis, input_pos, mask, use_reentrant=False)
            else:
                h = layer(h, freqs_cis, input_pos, mask)

        h = self.norm(h)
        logits_2d = self.output(h[:, self.cls_token_num :: 2]).float()
        logits_1d = self._titok_aux_logits(z1d)

        loss_1d = F.cross_entropy(logits_1d.reshape(-1, logits_1d.size(-1)), targets_1d.reshape(-1))
        loss_2d = F.cross_entropy(logits_2d.reshape(-1, logits_2d.size(-1)), targets_2d_ordered.reshape(-1))
        loss = self.loss_1d_weight * loss_1d + self.loss_2d_weight * loss_2d

        with torch.no_grad():
            acc_1d = (logits_1d.argmax(dim=-1) == targets_1d).float().mean()
            acc_2d = (logits_2d.argmax(dim=-1) == targets_2d_ordered).float().mean()

        return {
            "logits_1d": logits_1d,
            "logits_2d": logits_2d,
            "loss": loss,
            "loss_1d": loss_1d.detach(),
            "loss_2d": loss_2d.detach(),
            "acc_1d": acc_1d.detach(),
            "acc_2d": acc_2d.detach(),
            "token_order": token_order,
        }



def split_new_and_base_params(model):
    new_prefixes = ("titok_", "modality_embeddings")
    base_decay, base_nodecay, new_decay, new_nodecay = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_new = name.startswith(new_prefixes)
        is_decay = param.dim() >= 2
        if is_new and is_decay:
            new_decay.append(param)
        elif is_new:
            new_nodecay.append(param)
        elif is_decay:
            base_decay.append(param)
        else:
            base_nodecay.append(param)
    return base_decay, base_nodecay, new_decay, new_nodecay
