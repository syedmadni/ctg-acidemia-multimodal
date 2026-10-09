"""Phase 4: clinically-motivated CTG features (FIGO/NICE-style), computed from
the same [N, T, 2] arrays the existing ctg_summary_features() consumes.

Why: the current tabular models see only mean/std/min/max/median per channel.
The clinically-established markers of fetal distress -- reduced FHR variability
and (especially) decelerations -- are invisible to that feature set. DeepCTG 2.0
(the target-journal reference) pretrains on exactly these kinds of features.

CRITICAL representation detail (verified against the data 2026-07-21): the
stored 90-min arrays contain NO NaN and NO zeros -- every sample is a real
(interpolated) value, FHR in ~[30, 275] bpm. Zeros appear ONLY after
crop_ctg_array() zeroes the tail past a checkpoint. Therefore every feature here
is computed over the **active prefix only** (FHR > 0). Computing across the
crop boundary would read the 140->0 bpm drop as a huge fake deceleration -- the
single most important bug to avoid in this module.

Channel order: 0 = FHR, 1 = UC (build_windows.py:181). Sample rate is 1 Hz for
the modeling arrays, so 1 sample = 1 second.

All features are NaN/degenerate-safe: records with a too-short active region
return 0.0 for every feature rather than raising.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

# FIGO/NICE thresholds, in the signal's native units (bpm, seconds at 1 Hz).
ACCEL_RISE_BPM = 15.0
DECEL_DROP_BPM = 15.0
EPISODE_MIN_SEC = 15  # sustained >= 15 s to count as an accel/deceleration
MIN_ACTIVE_SEC = 30   # below this many active samples, features are undefined -> 0

FEATURE_NAMES: List[str] = [
    "baseline_fhr",
    "fhr_stv",              # short-term variability (mean |successive diff|)
    "fhr_ltv",              # long-term variability (mean per-minute range)
    "accel_rate_per10min",
    "decel_rate_per10min",
    "decel_total_dur_frac", # fraction of active time spent in deceleration
    "decel_max_depth_bpm",
    "uc_rate_per10min",     # contraction peaks per 10 min
]


def _active_prefix(fhr: np.ndarray, uc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the leading real portion of the record (FHR > 0). The crop zeroes
    the tail, so the active region is a contiguous prefix."""
    valid = fhr > 0.0
    if not valid.any():
        return fhr[:0], uc[:0]
    last = int(np.nonzero(valid)[0][-1]) + 1
    return fhr[:last], uc[:last]


def _episode_count_and_stats(
    fhr: np.ndarray, baseline: float, drop: bool
) -> Tuple[int, int, float]:
    """Count sustained episodes where FHR stays >= EPISODE_MIN_SEC beyond a
    threshold from baseline. Returns (n_episodes, total_samples_in_episodes,
    max_depth_or_height_bpm)."""
    if drop:
        beyond = fhr <= (baseline - DECEL_DROP_BPM)
        extremity = baseline - fhr  # positive depth below baseline
    else:
        beyond = fhr >= (baseline + ACCEL_RISE_BPM)
        extremity = fhr - baseline

    n_episodes = 0
    total_samples = 0
    max_extreme = 0.0
    run = 0
    run_max = 0.0
    for i in range(len(fhr)):
        if beyond[i]:
            run += 1
            run_max = max(run_max, float(extremity[i]))
        else:
            if run >= EPISODE_MIN_SEC:
                n_episodes += 1
                total_samples += run
                max_extreme = max(max_extreme, run_max)
            run = 0
            run_max = 0.0
    if run >= EPISODE_MIN_SEC:  # episode running at the end
        n_episodes += 1
        total_samples += run
        max_extreme = max(max_extreme, run_max)
    return n_episodes, total_samples, max_extreme


def _uc_peak_count(uc: np.ndarray) -> int:
    """Crude contraction count: local maxima that rise clearly above the record's
    own UC baseline, with a refractory gap so one contraction isn't counted
    twice. Robust to the arbitrary UC units in CTU-UHB."""
    n = len(uc)
    if n < MIN_ACTIVE_SEC:
        return 0
    baseline = float(np.median(uc))
    spread = float(np.std(uc))
    if spread < 1e-6:
        return 0
    thresh = baseline + spread  # ~1 SD above baseline
    refractory = 30  # >= 30 s between contraction peaks
    count = 0
    last_peak = -refractory
    for i in range(1, n - 1):
        if uc[i] > thresh and uc[i] >= uc[i - 1] and uc[i] > uc[i + 1]:
            if i - last_peak >= refractory:
                count += 1
                last_peak = i
    return count


def _one_record(fhr_full: np.ndarray, uc_full: np.ndarray) -> np.ndarray:
    fhr, uc = _active_prefix(fhr_full, uc_full)
    active_sec = len(fhr)
    if active_sec < MIN_ACTIVE_SEC:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    baseline = float(np.median(fhr))
    stv = float(np.mean(np.abs(np.diff(fhr)))) if active_sec >= 2 else 0.0

    # long-term variability: mean of per-minute (max - min); drop a short tail minute.
    n_min = active_sec // 60
    if n_min >= 1:
        trimmed = fhr[: n_min * 60].reshape(n_min, 60)
        ltv = float(np.mean(trimmed.max(axis=1) - trimmed.min(axis=1)))
    else:
        ltv = float(fhr.max() - fhr.min())

    n_accel, _, _ = _episode_count_and_stats(fhr, baseline, drop=False)
    n_decel, decel_samples, decel_depth = _episode_count_and_stats(fhr, baseline, drop=True)
    uc_peaks = _uc_peak_count(uc)

    active_min = active_sec / 60.0
    per10 = 10.0 / active_min  # scale a count to a per-10-min rate

    return np.array([
        baseline,
        stv,
        ltv,
        n_accel * per10,
        n_decel * per10,
        decel_samples / active_sec,  # fraction of active time in deceleration
        decel_depth,
        uc_peaks * per10,
    ], dtype=np.float32)


def clinical_ctg_features(X_ctg: np.ndarray, target_sample_rate_hz: int = 1) -> np.ndarray:
    """Compute the clinical feature block for a batch of records.

    X_ctg: [N, T, 2] (channel 0 = FHR, 1 = UC). target_sample_rate_hz is accepted
    for interface symmetry; the episode thresholds above assume 1 Hz (1 sample =
    1 s), which matches the modeling arrays. Returns [N, len(FEATURE_NAMES)].
    """
    if target_sample_rate_hz != 1:
        raise ValueError(
            "clinical_ctg_features assumes 1 Hz modeling arrays; "
            f"got target_sample_rate_hz={target_sample_rate_hz}."
        )
    n = X_ctg.shape[0]
    out = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    for i in range(n):
        out[i] = _one_record(X_ctg[i, :, 0].astype(np.float64), X_ctg[i, :, 1].astype(np.float64))
    return out


if __name__ == "__main__":
    # Quick self-check on the anytime cohort's raw arrays.
    from pathlib import Path

    X = np.load(Path("data/processed/X_ctg_705_anytime_vr060.npy"))
    feats = clinical_ctg_features(X, target_sample_rate_hz=1)
    print(f"X {X.shape} -> clinical features {feats.shape}")
    for j, name in enumerate(FEATURE_NAMES):
        col = feats[:, j]
        print(f"  {name:22s} mean={col.mean():8.3f} min={col.min():8.3f} max={col.max():8.3f} n_nan={int(np.isnan(col).sum())}")
