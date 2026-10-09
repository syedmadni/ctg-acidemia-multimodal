"""Read all CTU-UHB records once: raw 4 Hz FHR/UC plus every header field.

The existing `build_labels.parse_hea_to_kv` splits on two-or-more spaces and falls back to
the first single space. Header lines with a single space before the value
(`#Liq. praecox 1`) are therefore parsed as key `Liq.`, which is why `liq_praecox` is NaN
for all 552 records in `data/processed/record_meta_labels_acidemia.csv`. This parser
takes the LAST whitespace-separated token as the value and maps every known field
explicitly. It is cross-checked against the existing file for every shared field.

Cache (git-ignored): results/revision/cache/raw_signals.npz + header_meta.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.read_wfdb import load_record_ids, read_record_signals

RAW_DIR = Path("data/raw/ctu-uhb-ctgdb")
CACHE_DIR = Path("results/revision/cache")
SOURCE_HZ = 4

HEADER_FIELDS = {
    "pH": "ph", "BDecf": "bdecf", "pCO2": "pco2", "BE": "be",
    "Apgar1": "apgar1", "Apgar5": "apgar5",
    "NICU days": "nicu_days", "Seizures": "seizures", "HIE": "hie",
    "Intubation": "intubation", "Main diag.": "main_diag", "Other diag.": "other_diag",
    "Gest. weeks": "gest_weeks", "Weight(g)": "weight_g", "Sex": "sex",
    "Age": "age", "Gravidity": "gravidity", "Parity": "parity",
    "Diabetes": "diabetes", "Hypertension": "hypertension", "Preeclampsia": "preeclampsia",
    "Liq. praecox": "liq_praecox", "Pyrexia": "pyrexia", "Meconium": "meconium",
    "Presentation": "presentation", "Induced": "induced",
    "I.stage": "i_stage", "NoProgress": "no_progress", "CK/KP": "ck_kp",
    "II.stage": "ii_stage", "Deliv. type": "deliv_type",
    "dbID": "db_id", "Rec. type": "rec_type", "Pos. II.st.": "pos_ii_st", "Sig2Birth": "sig2birth",
}


def parse_header(text: str) -> dict:
    """Return {canonical_name: float} for every known field; raises if one is missing."""
    out = {}
    for line in text.splitlines():
        if not line.startswith("#"):
            continue
        content = line[1:].strip()
        parts = content.rsplit(None, 1)
        if len(parts) != 2:
            continue
        key, val = parts[0].strip(), parts[1].strip()
        if key in HEADER_FIELDS:
            name = HEADER_FIELDS[key]
            if name in out:
                raise ValueError(f"duplicate header field {key}")
            try:
                out[name] = float(val)
            except ValueError:
                out[name] = np.nan
    missing = set(HEADER_FIELDS.values()) - set(out)
    if missing:
        raise ValueError(f"header fields missing: {sorted(missing)}")
    return out


def build_cache(raw_dir: Path = RAW_DIR, cache_dir: Path = CACHE_DIR) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ids = load_record_ids(raw_dir)
    rows, fhrs, ucs = [], [], []
    for rid in ids:
        fhr, uc, fs = read_record_signals(raw_dir, rid, 0, 1)
        if int(round(fs)) != SOURCE_HZ:
            raise ValueError(f"record {rid}: fs={fs}, expected {SOURCE_HZ}")
        hdr = parse_header((raw_dir / f"{rid}.hea").read_text(encoding="utf-8", errors="ignore"))
        hdr["record_id"] = str(rid)
        hdr["length_samples"] = int(len(fhr))
        rows.append(hdr)
        fhrs.append(np.asarray(fhr, dtype=np.float32))
        ucs.append(np.asarray(uc, dtype=np.float32))
    max_len = max(len(x) for x in fhrs)
    F = np.full((len(ids), max_len), np.nan, dtype=np.float32)
    U = np.full((len(ids), max_len), np.nan, dtype=np.float32)
    for i, (f, u) in enumerate(zip(fhrs, ucs)):
        F[i, : len(f)] = f
        U[i, : len(u)] = u
    np.savez_compressed(cache_dir / "raw_signals.npz", fhr=F, uc=U,
                        record_id=np.array(ids), length=np.array([len(f) for f in fhrs]))
    meta = pd.DataFrame(rows)
    meta = meta[["record_id", "length_samples"] + list(HEADER_FIELDS.values())]
    meta.to_csv(cache_dir / "header_meta.csv", index=False)


class RawData:
    """Raw signals and header fields for all 552 records, in RECORDS order."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        if not (cache_dir / "raw_signals.npz").exists():
            build_cache(cache_dir=cache_dir)
        z = np.load(cache_dir / "raw_signals.npz")
        self.record_ids = [str(r) for r in z["record_id"]]
        self.length = z["length"].astype(int)
        self._fhr = z["fhr"]
        self._uc = z["uc"]
        self.meta = pd.read_csv(cache_dir / "header_meta.csv", dtype={"record_id": str})
        assert self.meta["record_id"].tolist() == self.record_ids
        self.index = {r: i for i, r in enumerate(self.record_ids)}

    def fhr(self, rid: str) -> np.ndarray:
        i = self.index[rid]
        return self._fhr[i, : self.length[i]]

    def uc(self, rid: str) -> np.ndarray:
        i = self.index[rid]
        return self._uc[i, : self.length[i]]

    def length_minutes(self, rid: str) -> float:
        return self.length[self.index[rid]] / (SOURCE_HZ * 60.0)


def gapped_label(ph: float) -> float:
    """Primary label: 1 if pH < 7.05, 0 if pH >= 7.15, NaN otherwise (build_labels.py:112-114)."""
    if not np.isfinite(ph):
        return np.nan
    if ph < 7.05:
        return 1.0
    if ph >= 7.15:
        return 0.0
    return np.nan
