"""Task A.7: anything computed at checkpoint t must be unchanged when every sample at or
after t is replaced with random noise."""
import numpy as np
import pytest

from src.data.build_windows import downsample_mean, extract_first_window, linear_interpolate_signal
from src.data.fixed_ctg_features import ctg_summary_features_v2
from src.revision.causal_features import (
    causal_ctg_features, causal_deep_input, n_samples, quality_ratio, valid_bins_1hz, fhr_valid,
)
from src.revision.raw_cache import RawData

CHECKPOINTS = [10.0, 20.0, 30.0, 45.0, 60.0]


@pytest.fixture(scope="module")
def raw():
    return RawData()


def _noised(x, t, rng):
    y = np.array(x, dtype=np.float32, copy=True)
    k = n_samples(t)
    noise = rng.uniform(-50, 250, size=len(y) - k).astype(np.float32)
    noise[rng.random(len(noise)) < 0.2] = 0.0      # include "invalid" zeros
    noise[rng.random(len(noise)) < 0.05] = np.nan
    y[k:] = noise
    return y


def _records(raw, n=60):
    # a deterministic spread of real records, including ones with leading/straddling gaps
    return raw.record_ids[:: max(1, len(raw.record_ids) // n)]


def test_quality_and_features_ignore_future(raw):
    rng = np.random.default_rng(0)
    for rid in _records(raw):
        f, u = raw.fhr(rid), raw.uc(rid)
        for t in CHECKPOINTS:
            fn, un = _noised(f, t, rng), _noised(u, t, rng)
            assert quality_ratio(f, t) == quality_ratio(fn, t)
            np.testing.assert_array_equal(causal_ctg_features(f, u, t), causal_ctg_features(fn, un, t))


def test_deep_input_prefix_ignores_future(raw):
    rng = np.random.default_rng(1)
    for rid in _records(raw, 30):
        f, u = raw.fhr(rid), raw.uc(rid)
        a = causal_deep_input(f, u)
        for t in CHECKPOINTS:
            b = causal_deep_input(_noised(f, t, rng), _noised(u, t, rng))
            k = int(t * 60)
            np.testing.assert_array_equal(a[:k], b[:k])


def test_features_by_hand():
    # 3 seconds at 4 Hz, t = 3 s = 0.05 min. Second 2 is all invalid, so it is NaN (no interpolation).
    fhr = np.array([120, 124, 0, 0, 0, 0, 0, 0, 140, 140, 140, np.nan], dtype=float)
    uc = np.array([10, 10, 10, 10, 0, 0, 0, 0, 30, 30, 30, 30], dtype=float)
    b = valid_bins_1hz(fhr, fhr_valid, 0.05)
    np.testing.assert_allclose(b[[0, 2]], [122.0, 140.0])
    assert np.isnan(b[1])
    feats = causal_ctg_features(fhr, uc, 0.05)
    # fhr bins {122, 140}; uc bins {10, 30}
    np.testing.assert_allclose(feats, [131, 20, 9, 10, 122, 10, 140, 30, 131, 20])
    assert quality_ratio(fhr, 0.05) == 5 / 12


def test_published_pipeline_leaks(raw):
    """Documents the audit finding (DEVIATIONS F2): the published interpolate-then-crop features
    DO change when the future is replaced, for at least one real record."""
    rng = np.random.default_rng(2)
    changed = 0
    for rid in raw.record_ids[:120]:
        f = raw.fhr(rid)
        for t in (10.0, 60.0):
            outs = []
            for x in (f, _noised(f, t, rng)):
                w, _, _ = extract_first_window(x, 21600)
                w = linear_interpolate_signal(w, ~(np.isfinite(w) & (w > 0)))
                ds = downsample_mean(w, 4, 1)
                arr = np.stack([ds, ds], axis=-1)[None]
                outs.append(ctg_summary_features_v2(arr, checkpoint_minutes=t))
            changed += not np.array_equal(outs[0], outs[1])
    assert changed > 0
