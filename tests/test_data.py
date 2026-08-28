import json

import numpy as np

from motar.data import PackedJointCodeDataset, validate_packed_dataset


def make_dataset(root, count=8):
    rng = np.random.default_rng(0)
    np.save(root / "titok_codes.npy", rng.integers(0, 4096, (count, 2, 32), dtype=np.uint16))
    np.save(
        root / "llamagen_codes.npy",
        rng.integers(0, 16384, (count, 2, 256), dtype=np.uint16),
    )
    np.save(root / "labels.npy", np.arange(count, dtype=np.uint16) % 1000)
    np.save(root / "written.npy", np.ones(count, dtype=np.uint8))
    (root / "meta.json").write_text(
        json.dumps({"completed": True, "num_samples": count}) + "\n"
    )


def test_validate_and_load_packed_dataset(tmp_path):
    make_dataset(tmp_path)
    report = validate_packed_dataset(tmp_path)
    assert report["num_samples"] == 8
    assert report["titok_shape"] == [8, 2, 32]
    assert report["llamagen_shape"] == [8, 2, 256]

    dataset = PackedJointCodeDataset(tmp_path)
    z1d, z2d, label, index = dataset[3]
    assert tuple(z1d.shape) == (32,)
    assert tuple(z2d.shape) == (256,)
    assert label.item() == 3
    assert index.item() == 3


def test_incomplete_dataset_is_rejected(tmp_path):
    make_dataset(tmp_path)
    written = np.load(tmp_path / "written.npy")
    written[-1] = 0
    np.save(tmp_path / "written.npy", written)
    try:
        validate_packed_dataset(tmp_path)
    except RuntimeError as exc:
        assert "incomplete" in str(exc).lower()
    else:
        raise AssertionError("incomplete packed dataset was accepted")
