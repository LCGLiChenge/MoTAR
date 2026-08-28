import torch
import torch.nn.functional as F

from motar.model import TiTokLlamaGenUnifiedAR
from motar.sidecar import (
    TiTok1DDisjointSidecar,
    detached_sidecar_inputs,
)


def model_kwargs():
    return dict(
        n_layer=2,
        n_head=4,
        dim=64,
        multiple_of=16,
        model_type="c2i",
        vocab_size=128,
        block_size=16,
        num_classes=10,
        cls_token_num=1,
        class_dropout_prob=0.1,
        resid_dropout_p=0.0,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        drop_path_rate=0.0,
        token_dropout_p=0.0,
        position_order="raster",
        grad_checkpointing=False,
        titok_vocab_size=32,
        titok_num_tokens=4,
        titok_conditioning="prefix",
        loss_1d_weight=1.5,
        loss_2d_weight=1.0,
    )


def fixed_batch():
    z1d = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    z2d = torch.arange(32).reshape(2, 16) % 128
    labels = torch.tensor([2, 7])
    order = torch.arange(16).unsqueeze(0).repeat(2, 1)
    return z1d, z2d, labels, order


def make_pair(seed=123):
    torch.manual_seed(seed)
    baseline = TiTokLlamaGenUnifiedAR(**model_kwargs())
    baseline_rng = torch.get_rng_state()

    torch.manual_seed(seed)
    main = TiTokLlamaGenUnifiedAR(**model_kwargs())
    sidecar = TiTok1DDisjointSidecar(main, depth=2)
    disjoint_rng = torch.get_rng_state()

    torch.testing.assert_close(disjoint_rng, baseline_rng, rtol=0, atol=0)
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(main.state_dict()[key], value, rtol=0, atol=0)
    return baseline, main, sidecar


def sidecar_forward(main, sidecar, z1d, labels):
    with torch.random.fork_rng(enabled=True):
        prefix_h, freqs_cis = detached_sidecar_inputs(main, z1d, labels)
        return sidecar(prefix_h, freqs_cis)


def test_zero_init_and_rng_neutrality():
    baseline, main, sidecar = make_pair()
    baseline.eval()
    main.eval()
    sidecar.eval()
    z1d, z2d, labels, order = fixed_batch()

    with torch.no_grad():
        baseline_out = baseline(z1d, z2d, labels, token_order=order)
        main_out = main(z1d, z2d, labels, token_order=order)
        residual = sidecar_forward(main, sidecar, z1d, labels)

    torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0, atol=0)
    torch.testing.assert_close(
        main_out["logits_1d"] + residual,
        baseline_out["logits_1d"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        main_out["logits_2d"],
        baseline_out["logits_2d"],
        rtol=0,
        atol=0,
    )


def test_separate_backward_clip_and_step_leave_main_exact():
    baseline, main, sidecar = make_pair(seed=77)
    baseline.train()
    main.train()
    sidecar.train()
    z1d, z2d, labels, order = fixed_batch()

    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=1e-3)
    main_optimizer = torch.optim.AdamW(main.parameters(), lr=1e-3)
    sidecar_optimizer = torch.optim.AdamW(sidecar.parameters(), lr=2e-3)

    torch.manual_seed(991)
    baseline_out = baseline(z1d, z2d, labels, token_order=order)
    baseline_out["loss"].backward()
    baseline_rng = torch.get_rng_state()

    torch.manual_seed(991)
    main_out = main(z1d, z2d, labels, token_order=order)
    residual = sidecar_forward(main, sidecar, z1d, labels)
    sidecar_loss = F.cross_entropy(
        (main_out["logits_1d"].detach() + residual).reshape(-1, residual.shape[-1]),
        z1d.reshape(-1),
    )
    main_out["loss"].backward()
    sidecar_loss.backward()
    disjoint_rng = torch.get_rng_state()

    torch.testing.assert_close(disjoint_rng, baseline_rng, rtol=0, atol=0)
    main_parameters = dict(main.named_parameters())
    for name, parameter in baseline.named_parameters():
        other = main_parameters[name]
        if parameter.grad is None:
            assert other.grad is None
        else:
            torch.testing.assert_close(other.grad, parameter.grad, rtol=0, atol=0)
    assert sum(
        parameter.grad.abs().sum()
        for parameter in sidecar.parameters()
        if parameter.grad is not None
    ) > 0

    torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(main.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(sidecar.parameters(), 1.0)
    baseline_optimizer.step()
    main_optimizer.step()
    sidecar_optimizer.step()

    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(main.state_dict()[key], value, rtol=0, atol=0)
