import numpy as np

from evaluate_2d import fixed_order


def test_fixed_order_is_a_deterministic_permutation():
    first = fixed_order(sample_id=17, block_size=256, seed=20260828)
    second = fixed_order(sample_id=17, block_size=256, seed=20260828)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first), np.arange(256))


def test_fixed_order_changes_with_sample_or_seed():
    reference = fixed_order(sample_id=17, block_size=256, seed=20260828)
    assert not np.array_equal(
        reference,
        fixed_order(sample_id=18, block_size=256, seed=20260828),
    )
    assert not np.array_equal(
        reference,
        fixed_order(sample_id=17, block_size=256, seed=20260829),
    )
