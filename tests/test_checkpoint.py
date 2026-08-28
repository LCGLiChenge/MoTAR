import json
from pathlib import Path

from motar.checkpoint import (
    PREVIOUS_NAME,
    STAGED_NAME,
    promote_staged_checkpoint,
    recover_checkpoint_tree,
    save_latest_checkpoint,
    validate_checkpoint,
)


def make_checkpoint(path, epoch, world_size=1):
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(b"model")
    (path / "optimizer.bin").write_bytes(b"optimizer")
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "completed_epochs": epoch,
                "world_size": world_size,
            }
        )
        + "\n"
    )


def test_latest_promotion_keeps_only_latest(tmp_path):
    root = tmp_path / "checkpoints"
    make_checkpoint(root / "latest", epoch=4)
    make_checkpoint(root / STAGED_NAME, epoch=5)

    latest = promote_staged_checkpoint(root)
    assert validate_checkpoint(latest)["completed_epochs"] == 5
    assert sorted(path.name for path in root.iterdir()) == ["latest"]


def test_recover_previous_after_interrupted_rotation(tmp_path):
    root = tmp_path / "checkpoints"
    make_checkpoint(root / PREVIOUS_NAME, epoch=9)

    latest = recover_checkpoint_tree(root)
    assert latest is not None
    assert validate_checkpoint(latest)["completed_epochs"] == 9
    assert sorted(path.name for path in root.iterdir()) == ["latest"]


class FakeAccelerator:
    is_main_process = True

    def wait_for_everyone(self):
        pass

    def save_state(self, output_dir):
        path = Path(output_dir)
        path.mkdir(parents=True)
        (path / "model.safetensors").write_bytes(b"model-state")
        (path / "optimizer.bin").write_bytes(b"optimizer-state")


def test_collective_save_replaces_latest_without_siblings(tmp_path):
    accelerator = FakeAccelerator()
    root = tmp_path / "checkpoints"
    first = {
        "completed_epochs": 1,
        "total_epochs": 150,
        "world_size": 8,
    }
    second = dict(first, completed_epochs=2)

    save_latest_checkpoint(accelerator, root, first)
    save_latest_checkpoint(accelerator, root, second)

    assert validate_checkpoint(root / "latest")["completed_epochs"] == 2
    assert sorted(path.name for path in root.iterdir()) == ["latest"]
