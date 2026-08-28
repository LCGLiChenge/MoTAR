import torch

from motar.model import TiTokLlamaGenUnifiedAR


def test_tiny_unified_forward_is_finite():
    torch.manual_seed(0)
    model = TiTokLlamaGenUnifiedAR(
        n_layer=1,
        n_head=2,
        dim=64,
        model_type="c2i",
        vocab_size=16,
        block_size=4,
        num_classes=10,
        cls_token_num=1,
        resid_dropout_p=0.0,
        ffn_dropout_p=0.0,
        drop_path_rate=0.0,
        token_dropout_p=0.0,
        position_order="raster",
        grad_checkpointing=False,
        zero_class_qk=True,
        num_inference_steps=2,
        titok_vocab_size=8,
        titok_num_tokens=2,
        titok_conditioning="prefix",
        loss_1d_weight=1.0,
        loss_2d_weight=1.0,
    )
    z1d = torch.randint(0, 8, (2, 2))
    z2d = torch.randint(0, 16, (2, 4))
    labels = torch.randint(0, 10, (2,))
    output = model(
        z1d=z1d,
        z2d=z2d,
        cond_idx=labels,
        targets_1d=z1d,
        targets_2d=z2d,
    )
    assert tuple(output["logits_1d"].shape) == (2, 2, 8)
    assert tuple(output["logits_2d"].shape) == (2, 4, 16)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
