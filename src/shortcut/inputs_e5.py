"""Input formats for pre-registered experiment E5 (results/shortcut_e5/PREREGISTRATION_E5.md, section 4).

Both formats return a float32 array [5400, 2] at 1 Hz (channel 0 = FHR, channel 1 = UC) and reuse the
registered E1-E4 code in src/shortcut/inputs.py (imported, not copied).

F5   plain end marker: identical to F1 for every second before the end of the recording; every second at
     and after the end is 0 in both channels (raw units, before standardisation). Valid FHR is > 0 and gaps
     inside the recording are filled by interpolation, so FHR = 0 appears only after the end. A recording
     of 90 min or longer gives F5 == F1.
F5x  F5 with the end marker removed by the same rule as F1-x: the part after the end is filled by repeating
     the recording's own signal from its start. Because F5 == F1 before the end, F5x == F1x.
"""
from __future__ import annotations

import numpy as np

from src.shortcut import inputs


def f5_plain_end(fhr: np.ndarray, uc: np.ndarray, minutes: float = 90.0) -> np.ndarray:
    x = inputs.f1_conventional(fhr, uc, minutes)
    L = inputs.recorded_steps_1hz(len(fhr), minutes)
    x[L:] = 0.0
    return x.astype(np.float32, copy=False)


def f5_end_marker_removed(f5: np.ndarray, n_samples_4hz: int) -> np.ndarray:
    return inputs.f1_end_marker_removed(f5, n_samples_4hz)
