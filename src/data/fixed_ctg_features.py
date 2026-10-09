"""Corrected CTG summary features.

The original `ctg_summary_features` (src/training/train_cv.py:63-71) reduces over the
FULL 5400-sample axis. Because `crop_ctg_array` writes 0.0 rather than NaN past the
checkpoint, every statistic mixed padding into the result. Measured consequences on the
primary cohort:

  * `uc_min` is constant at every checkpoint, `fhr_min` at five of six, and both medians
    at 10/20/30 min. Four of ten features carry no information at early checkpoints.
  * `fhr_std` becomes a proxy for the FHR baseline level, not variability: correlation
    with the true mean is +0.9987 at 10 min and +0.9969 at 30 min, while correlation with
    the true SD is -0.113 and -0.057.

This module computes the same ten statistics over the real window only. Spec frozen in
outputs/quality_cohorts/primary_705_anytime_vr060/maternal_value/FEATURE_FIX_SPEC.md.

Deep-model inputs are deliberately untouched; zero padding is intended there.
"""
from __future__ import annotations

import numpy as np


def ctg_summary_features_v2(
    X_ctg: np.ndarray,
    checkpoint_minutes: float | None = None,
    target_sample_rate_hz: int = 1,
    treat_zero_as_missing: bool = True,
    legacy: bool = False,
) -> np.ndarray:
    """Ten CTG summary statistics, stat-major and channel-minor.

    Order: [fhr_mean, uc_mean, fhr_std, uc_std, fhr_min, uc_min,
            fhr_max, uc_max, fhr_median, uc_median]

    Parameters
    ----------
    X_ctg
        `[N, T, C]` array. Pass the UNCROPPED array and give `checkpoint_minutes`;
        this function does its own windowing so that padding is never introduced.
    checkpoint_minutes
        Window is `[0, checkpoint_minutes * 60 * hz)`. `None` uses the whole array.
    treat_zero_as_missing
        Exclude zeros from the statistics. CTU-UHB codes signal loss as 0.
    legacy
        Reproduce the original buggy behaviour exactly, for comparison. Reduces over the
        full axis with `nan*` and no zero masking.
    """
    if legacy:
        stats = [
            np.nanmean(X_ctg, axis=1),
            np.nanstd(X_ctg, axis=1),
            np.nanmin(X_ctg, axis=1),
            np.nanmax(X_ctg, axis=1),
            np.nanmedian(X_ctg, axis=1),
        ]
        return np.concatenate(stats, axis=1).astype(np.float32)

    window = X_ctg
    if checkpoint_minutes is not None:
        n = int(round(checkpoint_minutes * 60 * target_sample_rate_hz))
        n = min(n, X_ctg.shape[1])
        window = X_ctg[:, :n, :]

    w = window.astype(np.float64, copy=True)
    if treat_zero_as_missing:
        w[w == 0] = np.nan

    # An all-NaN channel yields NaN here, which is the intended signal to the
    # within-fold imputer. numpy warns on all-NaN slices, so suppress only that.
    with np.errstate(invalid="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN slice encountered")
            warnings.filterwarnings("ignore", r"Mean of empty slice")
            warnings.filterwarnings("ignore", r"Degrees of freedom <= 0")
            stats = [
                np.nanmean(w, axis=1),
                np.nanstd(w, axis=1),
                np.nanmin(w, axis=1),
                np.nanmax(w, axis=1),
                np.nanmedian(w, axis=1),
            ]
    return np.concatenate(stats, axis=1).astype(np.float32)


def make_tabular_features_v2(
    X_ctg: np.ndarray,
    X_maternal: np.ndarray,
    checkpoint_minutes: float | None = None,
    target_sample_rate_hz: int = 1,
    legacy: bool = False,
) -> np.ndarray:
    """Counterpart of `make_tabular_features`: 10 fixed CTG stats + 8 maternal."""
    ctg = ctg_summary_features_v2(
        X_ctg,
        checkpoint_minutes=checkpoint_minutes,
        target_sample_rate_hz=target_sample_rate_hz,
        legacy=legacy,
    )
    return np.concatenate([ctg, X_maternal], axis=1)


FEATURE_NAMES = [
    "fhr_mean", "uc_mean", "fhr_std", "uc_std", "fhr_min",
    "uc_min", "fhr_max", "uc_max", "fhr_median", "uc_median",
]
