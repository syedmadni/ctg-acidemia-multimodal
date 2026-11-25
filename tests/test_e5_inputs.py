"""Unit tests for the E5 input formats (src/shortcut/inputs_e5.py). Synthetic signals, no data needed."""
import numpy as np

from src.shortcut import inputs as F
from src.shortcut import inputs_e5 as F5

HZ = 4


def _rec(minutes, seed=0, drop=0.05):
    rng = np.random.default_rng(seed)
    n = int(round(minutes * 60 * HZ))
    fhr = (140 + rng.normal(0, 3, n)).astype(np.float32)
    uc = (20 + rng.normal(0, 2, n)).astype(np.float32)
    fhr[rng.random(n) < drop] = 0
    return fhr, uc


def test_shape_and_dtype():
    fhr, uc = _rec(70)
    x = F5.f5_plain_end(fhr, uc)
    assert x.shape == (5400, 2)
    assert x.dtype == np.float32


def test_equals_f1_before_end():
    for minutes in (60, 70.3, 83.77):
        fhr, uc = _rec(minutes, seed=1)
        L = F.recorded_steps_1hz(len(fhr))
        assert np.array_equal(F5.f5_plain_end(fhr, uc)[:L], F.f1_conventional(fhr, uc)[:L])


def test_zero_after_end_in_both_channels():
    fhr, uc = _rec(70.3, seed=2)
    L = F.recorded_steps_1hz(len(fhr))
    x = F5.f5_plain_end(fhr, uc)
    assert L < 5400
    assert np.all(x[L:, 0] == 0) and np.all(x[L:, 1] == 0)


def test_f5x_equals_f1x():
    for minutes in (60, 70.3, 89.9):
        fhr, uc = _rec(minutes, seed=3)
        f5x = F5.f5_end_marker_removed(F5.f5_plain_end(fhr, uc), len(fhr))
        f1x = F.f1_end_marker_removed(F.f1_conventional(fhr, uc), len(fhr))
        assert np.array_equal(f5x, f1x)


def test_long_recording_equals_f1():
    for minutes in (90, 104):
        fhr, uc = _rec(minutes, seed=4)
        assert np.array_equal(F5.f5_plain_end(fhr, uc), F.f1_conventional(fhr, uc))


def test_no_fhr_zero_before_end_with_interior_gaps():
    fhr, uc = _rec(75, seed=5)
    fhr[1000:1400] = 0          # 100-s run of zeros
    fhr[5000:5100] = np.nan     # 25-s run of NaN
    fhr[9000:9004] = 0          # 1-s run
    L = F.recorded_steps_1hz(len(fhr))
    x = F5.f5_plain_end(fhr, uc)
    assert np.all(x[:L, 0] > 0)
    assert not np.any(x[:L, 0] == 0)
