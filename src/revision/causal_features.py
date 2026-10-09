"""Leak-free quality ratio, CTG summary features and deep-model inputs.

Everything here reads raw 4 Hz samples in [0, t) only; nothing depends on a sample at or
after t. Definitions are frozen in results/revision/DEVIATIONS.md section 3:

* valid FHR sample: finite and > 0 (unchanged from src/data/build_windows.py:153)
* valid UC sample: finite and != 0 (the zero-as-missing convention of ctg_summary_features_v2)
* quality_ratio(fhr, t): valid FHR samples in [0, t*240) / (t*240)
* features: 1-s bins = mean of the valid samples in the bin (NaN if none), then the same ten
  statistics as src/data/fixed_ctg_features.py over the non-NaN bins. No interpolation.
* deep inputs: the same 1-s bins, then causal forward-fill (a leading gap stays 0).
"""
from __future__ import annotations

import warnings

import numpy as np

SOURCE_HZ = 4
FEATURE_NAMES = [
    "fhr_mean", "uc_mean", "fhr_std", "uc_std", "fhr_min",
    "uc_min", "fhr_max", "uc_max", "fhr_median", "uc_median",
]


def fhr_valid(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & (x > 0)


def uc_valid(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & (x != 0)


def n_samples(t_minutes: float, hz: int = SOURCE_HZ) -> int:
    return int(round(t_minutes * 60 * hz))


def quality_ratio(fhr: np.ndarray, t_minutes: float, hz: int = SOURCE_HZ) -> float:
    n = n_samples(t_minutes, hz)
    seg = np.asarray(fhr, dtype=np.float64)[:n]
    return float(fhr_valid(seg).sum() / n)


def valid_bins_1hz(x: np.ndarray, valid_fn, t_minutes: float | None = None,
                   hz: int = SOURCE_HZ) -> np.ndarray:
    """Per-second mean of the valid samples in [0, t); NaN for a second with none."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x) if t_minutes is None else min(len(x), n_samples(t_minutes, hz))
    nb = n // hz
    seg = x[: nb * hz].reshape(nb, hz)
    v = valid_fn(seg)
    cnt = v.sum(axis=1)
    tot = np.where(v, seg, 0.0).sum(axis=1)
    out = np.full(nb, np.nan)
    ok = cnt > 0
    out[ok] = tot[ok] / cnt[ok]
    return out


def _stats(b: np.ndarray) -> list[float]:
    if not np.isfinite(b).any():
        return [np.nan] * 5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return [float(np.nanmean(b)), float(np.nanstd(b)), float(np.nanmin(b)),
                float(np.nanmax(b)), float(np.nanmedian(b))]


def causal_ctg_features(fhr: np.ndarray, uc: np.ndarray, t_minutes: float) -> np.ndarray:
    """Ten features, stat-major and channel-minor (same order as FEATURE_NAMES)."""
    f = _stats(valid_bins_1hz(fhr, fhr_valid, t_minutes))
    u = _stats(valid_bins_1hz(uc, uc_valid, t_minutes))
    out = []
    for i in range(5):
        out += [f[i], u[i]]
    return np.asarray(out, dtype=np.float64)


def causal_feature_matrix(raw, record_ids, t_minutes: float) -> np.ndarray:
    return np.vstack([causal_ctg_features(raw.fhr(r), raw.uc(r), t_minutes) for r in record_ids])


def _ffill_causal(b: np.ndarray) -> np.ndarray:
    out = b.copy()
    last = 0.0  # leading gap stays 0, the existing missing code
    for i in range(len(out)):
        if np.isfinite(out[i]):
            last = out[i]
        else:
            out[i] = last
    return out


def causal_deep_input(fhr: np.ndarray, uc: np.ndarray, t_max_minutes: float = 90.0) -> np.ndarray:
    """[T_max, 2] float32 at 1 Hz: valid-only bins, causal forward-fill, zero past the end."""
    t_steps = int(round(t_max_minutes * 60))
    out = np.zeros((t_steps, 2), dtype=np.float32)
    for c, (x, fn) in enumerate(((fhr, fhr_valid), (uc, uc_valid))):
        b = _ffill_causal(valid_bins_1hz(x, fn, t_max_minutes))
        out[: min(len(b), t_steps), c] = b[:t_steps]
    return out
