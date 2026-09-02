"""Sparse E117 refinement decoding without fabricating unselected 2D codes."""

from __future__ import annotations

import torch


def _validate_pointwise_projection(post_quant_conv: torch.nn.Module) -> None:
    required = {
        "kernel_size": (1, 1),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "groups": 1,
    }
    for name, expected in required.items():
        if getattr(post_quant_conv, name, None) != expected:
            raise ValueError(
                f"sparse decoding requires pointwise post_quant_conv: {name}="
                f"{getattr(post_quant_conv, name, None)!r}, expected {expected!r}"
            )


@torch.inference_mode()
def decode_e117_sparse_codes(
    tokenizer_model: torch.nn.Module,
    f_1d: torch.Tensor,
    sparse_codes: torch.Tensor,
    route_indices: torch.Tensor,
    route_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Replace only routed 16x16 latent cells and run the frozen decoder.

    LlamaGen's ``post_quant_conv`` is 1x1, so projecting selected codebook
    vectors independently is exactly equivalent to projecting a dense grid at
    those cells.  No dummy 2D codes are introduced for unselected positions.
    """

    if f_1d.ndim != 4 or tuple(f_1d.shape[-2:]) != (16, 16):
        raise ValueError(f"f_1d must be [B,C,16,16], got {tuple(f_1d.shape)}")
    if sparse_codes.shape != route_indices.shape or sparse_codes.shape != route_valid.shape:
        raise ValueError("sparse code/index/valid shapes must agree")
    if sparse_codes.ndim != 2 or sparse_codes.shape[0] != f_1d.shape[0]:
        raise ValueError("sparse inputs must be [B,K] and match f_1d batch")
    if route_valid.dtype != torch.bool:
        raise ValueError("route_valid must be boolean")
    counts = route_valid.sum(dim=1)
    if not bool(torch.all((counts == 64) | (counts == 128))):
        raise ValueError("E117 sparse decoding requires K in {64,128}")
    selected_indices = route_indices[route_valid]
    selected_codes = sparse_codes[route_valid]
    if bool(torch.any(selected_indices < 0)) or bool(torch.any(selected_indices >= 256)):
        raise ValueError("route index is outside the 16x16 grid")
    quantizer = tokenizer_model.llamagen_vq.quantize
    if bool(torch.any(selected_codes < 0)) or bool(torch.any(selected_codes >= int(quantizer.n_e))):
        raise ValueError("selected LlamaGen code is outside the codebook")
    post_quant_conv = tokenizer_model.llamagen_vq.post_quant_conv
    _validate_pointwise_projection(post_quant_conv)

    embeddings = quantizer.get_codebook_entry(selected_codes)
    if embeddings.ndim != 2:
        raise ValueError(f"unexpected selected codebook shape: {tuple(embeddings.shape)}")
    projected = post_quant_conv(embeddings.t()[None, :, None, :])
    projected = projected.squeeze(0).squeeze(1).t().contiguous()
    if projected.shape[1] != f_1d.shape[1]:
        raise ValueError("projected 2D channel count does not match f_1d")

    f_mix_flat = f_1d.flatten(2).transpose(1, 2).clone()
    batch_ids = torch.arange(f_1d.shape[0], device=f_1d.device)[:, None]
    batch_ids = batch_ids.expand_as(route_indices)[route_valid]
    f_mix_flat[batch_ids, selected_indices] = projected.to(f_mix_flat.dtype)
    f_mix = f_mix_flat.transpose(1, 2).reshape_as(f_1d)
    image = tokenizer_model.llamagen_vq.decoder(f_mix)
    return {"image": image, "f_mix": f_mix}
