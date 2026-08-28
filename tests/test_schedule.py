import math

from train import cosine_epoch_lambda, scale_learning_rate


def test_sqrt_lr_scaling():
    value = scale_learning_rate(1.75e-5, 2304, 576, "sqrt")
    assert math.isclose(value, 3.5e-5)


def test_epoch_schedule_uses_absolute_resume_progress():
    schedule = cosine_epoch_lambda(
        start_progress_epochs=25.0,
        total_epochs=150,
        warmup_epochs=2.25,
        steps_per_epoch=100,
        min_lr_ratio=0.05,
        num_cycles=0.5,
    )
    assert 0.05 < schedule(0) < 1.0
    assert schedule(100) < schedule(0)
    assert math.isclose(schedule(12500), 0.05, rel_tol=0.0, abs_tol=1e-8)
