import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from e117_sparse_maskgit import E117SparseUnifiedMaskGIT, fixed_ratio_mask
from scripts.validate_e117_maskgit_setup import E117_SHA256, validate_route_cache
from train_e117_sparse_maskgit import exposure_audit


def tiny_model():
    return E117SparseUnifiedMaskGIT(
        dim=64,
        n_layer=2,
        n_head=4,
        multiple_of=32,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        ffn_dropout_p=0.0,
        token_dropout_p=0.0,
        class_dropout_prob=0.1,
        grad_checkpointing=False,
        zero_init_output=False,
        output_bias=True,
    )


def test_1d_attention_is_bidirectional_and_generation_stays_in_vocab():
    torch.manual_seed(7)
    model = tiny_model().eval()
    first = torch.full((1, 32), model.mask_token_1d)
    second = first.clone()
    first[:, 20] = 11
    second[:, 20] = 17
    labels = torch.tensor([3])
    logits_first = model(stage="1d", input_tokens=first, labels=labels)
    logits_second = model(stage="1d", input_tokens=second, labels=labels)
    assert not torch.allclose(logits_first[:, 2], logits_second[:, 2])
    generated = model.generate_1d(
        labels,
        num_steps=2,
        cfg_scale=1.0,
        randomize_temperature=0.0,
        guidance_decay="constant",
        cfg_formula="standard",
    )
    assert generated.shape == (1, 32)
    assert 0 <= int(generated.min()) and int(generated.max()) < 4096


def test_sparse_2d_forward_and_fixed_masks():
    model = tiny_model().eval()
    batch = 2
    z1d = torch.randint(0, 4096, (batch, 32))
    indices = torch.arange(128)[None].expand(batch, -1)
    valid = torch.ones(batch, 128, dtype=torch.bool)
    tokens = torch.full((batch, 128), model.mask_token_2d)
    logits = model(
        stage="2d",
        completed_1d=z1d,
        input_tokens=tokens,
        route_indices=indices,
        route_valid=valid,
        labels=torch.tensor([1, 2]),
    )
    assert logits.shape == (batch, 128, 16384)
    assert fixed_ratio_mask(valid, 0.75, seed=9).sum(dim=1).tolist() == [96, 96]


def test_h200_rounded_exposure_budget_is_bounded_to_half_a_step():
    config = OmegaConf.create(
        {
            "data": {"full_1d_replay": True},
            "training": {
                "gradient_accumulation_steps": 1,
                "per_gpu_batch_size": 192,
                "replay_1d_per_gpu_batch_size": 192,
                "max_steps": 666667,
                "target_image_exposures": 1_024_000_000,
                "reference_num_train_sources": 1_281_167,
                "reference_max_steps": 500_000,
                "reference_global_batch_size": 2_048,
                "require_equal_branch_exposures": True,
            },
        }
    )
    audit = exposure_audit(config, world_size=8, num_train_sources=1_281_167)
    assert audit["global_batch_1d"] == audit["global_batch_2d"] == 1536
    assert audit["exposure_error"] == 512
    assert abs(audit["exposure_error"]) <= audit["global_batch_2d"] // 2


def test_registered_route_cache_contract():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        route_k = np.array([[64, 128], [128, 64]], dtype=np.uint8)
        route_indices = np.zeros((2, 2, 128), dtype=np.uint8)
        for row in range(2):
            for augmentation in range(2):
                route_indices[row, augmentation] = np.arange(128, dtype=np.uint8)
        np.save(root / "route_k.npy", route_k)
        np.save(root / "route_indices.npy", route_indices)
        np.save(root / "source_indices.npy", np.array([0, 1], dtype=np.int64))
        np.save(root / "written.npy", np.ones((2, 2), dtype=np.uint8))
        (root / "meta.json").write_text(
            json.dumps(
                {
                    "format": "e117_sparse_route_cache_v1",
                    "completed": True,
                    "e117_checkpoint_sha256": E117_SHA256,
                    "num_samples": 2,
                    "num_aug": 2,
                }
            )
        )
        report = validate_route_cache(
            root,
            {"num_samples": 2, "num_augmentations": 2},
            spot_check_rows=2,
        )
        assert report["k_values"] == [64, 128]
        assert report["completed"]
