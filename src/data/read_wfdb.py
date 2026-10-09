from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import wfdb


def load_record_ids(data_dir: Path) -> List[str]:
    records_path = data_dir / "RECORDS"
    if not records_path.exists():
        raise FileNotFoundError(f"RECORDS file not found: {records_path}")

    ids: List[str] = []
    with records_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rid = line.strip()
            if rid and not rid.startswith("#"):
                ids.append(rid)
    return ids


def read_record_signals(
    data_dir: Path,
    record_id: str,
    fhr_channel: int = 0,
    uc_channel: int = 1,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Read one WFDB record and return FHR, UC, fs."""
    rec = wfdb.rdrecord(str(data_dir / record_id))
    if rec.p_signal is None or rec.p_signal.ndim != 2 or rec.p_signal.shape[1] < 2:
        raise ValueError(f"Bad signal format for record {record_id}: {getattr(rec.p_signal, 'shape', None)}")

    fhr = rec.p_signal[:, fhr_channel].astype(np.float32, copy=False)
    uc = rec.p_signal[:, uc_channel].astype(np.float32, copy=False)
    fs = float(getattr(rec, "fs", np.nan))
    return fhr, uc, fs
