"""Input formats for the pre-registered shortcut experiments (analysis plan, section 4).

All formats return a float32 array [T, 2] at 1 Hz (channel 0 = FHR, channel 1 = UC).

F1   conventional fixed length: first 90 min from recording start, zero-padded at the tail, then
     the interpolation of src/data/build_windows.py (identical code path).
F1x  F1 with the end marker removed: the tail after the end of the recording is filled by
     repeating the recording's own F1 signal from its start.
F1c60 F1 built from the first 60 min only (as if the recording ended at 60 min); evaluation only.
F2   envelope only: same 90-min frame as F1, every recorded second replaced by N(0, 1) noise
     (seed = record ID), zeros after the end of the recording.
F3   delivery-anchored: the last 60 min before the end of the recording.
F3s  start-anchored counterpart of F3: the first 60 min of the recording.
     F3/F3s gaps: runs of invalid samples of at most 15 s (60 samples at 4 Hz) with valid samples
     on both sides are filled by linear interpolation; all other invalid samples are set to 0.
     Valid FHR: finite and > 0. Valid UC: finite and != 0 (src/revision/causal_features.py).
"""
from __future__ import annotations

import numpy as np

from src.data.build_windows import downsample_mean, extract_first_window, linear_interpolate_signal
from src.revision.causal_features import fhr_valid, uc_valid

SOURCE_HZ = 4
MAX_GAP_SAMPLES = 15 * SOURCE_HZ


def recorded_steps_1hz(n_samples_4hz: int, minutes: float = 90.0) -> int:
    """Number of 1-Hz steps that hold recorded signal inside a frame of `minutes`."""
    return int(min(n_samples_4hz // SOURCE_HZ, round(minutes * 60)))


def f1_conventional(fhr: np.ndarray, uc: np.ndarray, minutes: float = 90.0) -> np.ndarray:
    n = int(round(minutes * 60 * SOURCE_HZ))
    f, _, _ = extract_first_window(np.asarray(fhr, dtype=np.float32), n)
    u, _, _ = extract_first_window(np.asarray(uc, dtype=np.float32), n)
    f_valid = np.isfinite(f) & (f > 0)                       # build_windows.py: FHR invalid <= 0 or non-finite
    f_clean = linear_interpolate_signal(f, ~f_valid, fill_value=0.0)
    u_clean = linear_interpolate_signal(u, ~np.isfinite(u), fill_value=0.0)  # UC invalid: non-finite only
    return np.stack([downsample_mean(f_clean, SOURCE_HZ, 1), downsample_mean(u_clean, SOURCE_HZ, 1)],
                    axis=-1).astype(np.float32)


def f1_end_marker_removed(f1: np.ndarray, n_samples_4hz: int) -> np.ndarray:
    """F1x: fill the tail after the recording's end by repeating its own F1 signal from the start."""
    out = f1.copy()
    L = recorded_steps_1hz(n_samples_4hz, out.shape[0] / 60.0)
    if 0 < L < out.shape[0]:
        tail = np.arange(L, out.shape[0])
        out[tail] = out[(tail - L) % L]
    return out


def f1_cut(fhr: np.ndarray, uc: np.ndarray, cut_minutes: float = 60.0, minutes: float = 90.0) -> np.ndarray:
    """F1c60: the F1 pipeline applied to the first `cut_minutes` only (real-time input at that time)."""
    n = int(round(cut_minutes * 60 * SOURCE_HZ))
    return f1_conventional(np.asarray(fhr)[:n], np.asarray(uc)[:n], minutes)


def f2_envelope(n_samples_4hz: int, record_id: str, minutes: float = 90.0) -> np.ndarray:
    T = int(round(minutes * 60))
    out = np.zeros((T, 2), dtype=np.float32)
    L = recorded_steps_1hz(n_samples_4hz, minutes)
    rng = np.random.default_rng(int(record_id))
    out[:L] = rng.standard_normal((L, 2)).astype(np.float32)
    return out


def _fill_short_gaps(x: np.ndarray, valid: np.ndarray, max_gap: int = MAX_GAP_SAMPLES) -> np.ndarray:
    """Linear interpolation across interior invalid runs of length <= max_gap; other invalid -> 0."""
    x = np.asarray(x, dtype=np.float64).copy()
    valid = np.asarray(valid, dtype=bool)
    out = np.where(valid, x, 0.0)
    n = len(x)
    i = 0
    while i < n:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < n and not valid[j]:
            j += 1
        run = j - i
        if i > 0 and j < n and run <= max_gap:            # interior and short
            out[i:j] = np.interp(np.arange(i, j), [i - 1, j], [x[i - 1], x[j]])
        i = j
    return out.astype(np.float32)


def _window_1hz(fhr: np.ndarray, uc: np.ndarray, start: int, end: int) -> np.ndarray:
    f = np.asarray(fhr, dtype=np.float64)[start:end]
    u = np.asarray(uc, dtype=np.float64)[start:end]
    fc = _fill_short_gaps(f, fhr_valid(f))
    ucl = _fill_short_gaps(u, uc_valid(u))
    return np.stack([downsample_mean(fc, SOURCE_HZ, 1), downsample_mean(ucl, SOURCE_HZ, 1)], axis=-1).astype(np.float32)


def f3_delivery_anchored(fhr: np.ndarray, uc: np.ndarray, minutes: float = 60.0) -> np.ndarray:
    n = int(round(minutes * 60 * SOURCE_HZ))
    L = len(fhr)
    if L < n:
        raise ValueError(f"recording shorter than {minutes} min")
    return _window_1hz(fhr, uc, L - n, L)


def f3_start_anchored(fhr: np.ndarray, uc: np.ndarray, minutes: float = 60.0) -> np.ndarray:
    n = int(round(minutes * 60 * SOURCE_HZ))
    if len(fhr) < n:
        raise ValueError(f"recording shorter than {minutes} min")
    return _window_1hz(fhr, uc, 0, n)
