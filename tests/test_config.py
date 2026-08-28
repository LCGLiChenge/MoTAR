import pytest
from omegaconf import OmegaConf

from train import resolve_wandb_run_id, validate_registered_config


def test_h200_registered_invariants():
    config = OmegaConf.load("configs/h200_8gpu_150epoch.yaml")
    assert config.training.total_epochs == 150
    assert config.training.gradient_accumulation_steps == 1
    assert config.h200.required_gpu_count == 8
    assert config.model.n_layer == 24
    assert config.model.dim == 1024
    assert config.model.titok_num_tokens == 32
    assert config.model.loss_1d_weight == 1.5
    assert config.model.loss_2d_weight == 1.0
    assert config.training.mixed_precision == "bf16"
    assert config.training.log_with == "wandb"
    assert config.checkpoint.save_every_epochs == 1
    assert config.checkpoint.stable_name == "latest"
    validate_registered_config(config, 150)


def test_registered_config_rejects_weight_drift():
    config = OmegaConf.load("configs/h200_8gpu_150epoch.yaml")
    config.model.loss_1d_weight = 1.0
    with pytest.raises(ValueError, match="loss_1d_weight"):
        validate_registered_config(config, 150)


def test_wandb_run_id_is_stable_across_resume(tmp_path, monkeypatch):
    class FakeAccelerator:
        is_main_process = True

        def wait_for_everyone(self):
            pass

    accelerator = FakeAccelerator()
    monkeypatch.setenv("WANDB_RUN_ID", "first-run-id")
    assert resolve_wandb_run_id(tmp_path, accelerator, resume=False) == "first-run-id"
    monkeypatch.setenv("WANDB_RUN_ID", "different-id")
    assert resolve_wandb_run_id(tmp_path, accelerator, resume=True) == "first-run-id"
    assert resolve_wandb_run_id(tmp_path, accelerator, resume=False) == "different-id"
