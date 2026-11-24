"""Unit tests for the shortcut-experiment input formats (src/shortcut/inputs.py). No data needed."""
import numpy as np

from src.shortcut import inputs as F

HZ = 4


def _rec(minutes, seed=0, drop=0.05):
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * HZ)
    fhr = (140 + rng.normal(0, 3, n)).astype(np.float32)
    uc = (20 + rng.normal(0, 2, n)).astype(np.float32)
    fhr[rng.random(n) < drop] = 0
    return fhr, uc


def test_f1_shape_and_padding_marker():
    fhr, uc = _rec(70)
    x = F.f1_conventional(fhr, uc)
    assert x.shape == (5400, 2)
    assert np.all(x[4200:, 1] == 0)                 # UC tail is zero padding (the end marker)
    assert np.allclose(x[4200:, 0], x[4200, 0])     # FHR tail holds one value


def test_f1x_removes_marker():
    fhr, uc = _rec(70)
    x = F.f1_conventional(fhr, uc)
    xx = F.f1_end_marker_removed(x, len(fhr))
    assert np.array_equal(xx[:4200], x[:4200])
    assert np.array_equal(xx[4200:], x[:1200])
    assert not np.all(xx[4200:, 1] == 0)


def test_f2_has_no_signal_and_is_deterministic():
    fhr, uc = _rec(65)
    a = F.f2_envelope(len(fhr), "1234")
    b = F.f2_envelope(len(fhr), "1234")
    assert np.array_equal(a, b)
    assert np.all(a[3900:] == 0) and np.all(a[:3900] != 0)
    assert abs(float(a[:3900].mean())) < 0.1        # noise, unrelated to FHR level 140


def test_f3_windows():
    fhr, uc = _rec(80)
    assert F.f3_delivery_anchored(fhr, uc).shape == (3600, 2)
    assert F.f3_start_anchored(fhr, uc).shape == (3600, 2)
    f60, u60 = _rec(60)
    assert np.array_equal(F.f3_delivery_anchored(f60, u60), F.f3_start_anchored(f60, u60))


def test_gap_rule():
    x = np.full(400, 140.0)
    v = np.ones(400, bool)
    v[100:160] = False   # 60 samples = 15 s -> interpolated
    v[200:261] = False   # 61 samples -> zero
    v[:10] = False       # edge gap -> zero
    out = F._fill_short_gaps(x, v)
    assert np.allclose(out[100:160], 140.0)
    assert np.all(out[200:261] == 0)
    assert np.all(out[:10] == 0)


def test_start_anchored_inputs_ignore_future_signal():
    fhr, uc = _rec(85, seed=3)
    n60 = 60 * 60 * HZ
    f2, u2 = fhr.copy(), uc.copy()
    f2[n60:] = np.random.default_rng(9).normal(0, 50, len(f2) - n60)
    u2[n60:] = 0
    assert np.array_equal(F.f3_start_anchored(fhr, uc), F.f3_start_anchored(f2, u2))
    assert np.array_equal(F.f1_cut(fhr, uc, 60), F.f1_cut(f2, u2, 60))
