import hashlib

import numpy as np
import torch
from PIL import Image

from scripts.extract_codes import center_crop_arr, make_aug_batch, write_manifest
from scripts.prepare_extraction_assets import CHECKPOINTS, REPOSITORIES


def test_adm_crop_and_flip_contract():
    pixels = np.arange(300 * 500 * 3, dtype=np.uint32)
    image = Image.fromarray((pixels.reshape(300, 500, 3) % 256).astype(np.uint8))
    cropped = center_crop_arr(image, 256)
    assert cropped.size == (256, 256)

    tensor = torch.arange(2 * 3 * 2 * 4).reshape(2, 3, 2, 4)
    augmented, count = make_aug_batch(tensor, "adm")
    assert count == 2
    assert torch.equal(augmented[0], tensor[0])
    assert torch.equal(augmented[1], torch.flip(tensor[0], dims=[-1]))
    assert torch.equal(augmented[2], tensor[1])
    assert torch.equal(augmented[3], torch.flip(tensor[1], dims=[-1]))


def test_extraction_manifest_hashes_all_training_inputs(tmp_path):
    names = (
        "titok_codes.npy",
        "llamagen_codes.npy",
        "labels.npy",
        "written.npy",
        "meta.json",
    )
    for index, name in enumerate(names):
        (tmp_path / name).write_bytes(f"payload-{index}".encode())

    write_manifest(tmp_path)
    rows = (tmp_path / "manifest.sha256").read_text().splitlines()
    assert len(rows) == len(names)
    for row, name in zip(rows, names):
        expected = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        assert row == f"{expected}  {name}"


def test_extraction_assets_are_pinned():
    assert REPOSITORIES["titok"]["commit"] == "942a96fbdd873780179d1b78d5462911528bf8c8"
    assert REPOSITORIES["llamagen"]["commit"] == "ce98ec41803a74a90ce68c40ababa9eaeffeb4ec"
    assert CHECKPOINTS["mot"]["size"] == 6_400_628_829
    assert CHECKPOINTS["titok"]["size"] == 2_564_477_610
    assert CHECKPOINTS["llamagen"]["size"] == 287_920_306
