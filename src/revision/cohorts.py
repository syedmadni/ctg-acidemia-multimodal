"""Per-checkpoint cohorts, CONSORT flow and on-disk CV splits for the EJOG revision.

Variants (DEVIATIONS.md A1):
  ckfilter           PRIMARY. gapped label valid, length >= t, quality_ratio(fhr, t) >= 0.60
  unfiltered         gapped label valid, length >= t
  primary345_causal  the 345 records of the published cohort (unchanged membership)

Record order is RECORDS order (ascending record_id), which is also the order of the published
arrays, so primary345_causal gets folds identical to the published analysis.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.revision.causal_features import quality_ratio
from src.revision.predictors import MATERNAL_8
from src.revision.raw_cache import RawData, gapped_label
from src.training.train_cv import precompute_quality_splits

CHECKPOINTS = [10.0, 20.0, 30.0, 45.0, 60.0]
VARIANTS = ["ckfilter", "unfiltered", "primary345_causal"]
PRIMARY_VARIANT = "ckfilter"
QUALITY_THRESHOLD = 0.60
SEED = 42
SPLITS_DIR = Path("results/revision/splits")
PUBLISHED_IDS = Path("data/processed/record_ids_705_anytime_vr060.csv")


def labelled_meta(raw: RawData) -> pd.DataFrame:
    m = raw.meta.copy()
    m["y"] = m["ph"].map(gapped_label)
    m["length_min"] = m["length_samples"] / 240.0
    return m


def quality_table(raw: RawData, checkpoints=CHECKPOINTS) -> pd.DataFrame:
    rows = []
    for rid in raw.record_ids:
        r = {"record_id": rid}
        for t in checkpoints:
            r[f"q{int(t)}"] = quality_ratio(raw.fhr(rid), t) if raw.length_minutes(rid) >= t else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def members(meta: pd.DataFrame, q: pd.DataFrame, variant: str, t: float) -> pd.DataFrame:
    """Cohort rows (record_id, y, maternal, length) for one variant and checkpoint."""
    m = meta.merge(q, on="record_id")
    m = m[m["y"].notna()]
    if variant == "primary345_causal":
        ids = set(pd.read_csv(PUBLISHED_IDS)["record_id"].astype(str))
        m = m[m["record_id"].isin(ids)]
    else:
        m = m[m["length_min"] >= t]
        if variant == "ckfilter":
            m = m[m[f"q{int(t)}"] >= QUALITY_THRESHOLD]
    m = m.copy()
    m["y"] = m["y"].astype(int)
    return m.reset_index(drop=True)


def cohort_flow(meta: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n0 = len(meta)
    lab = meta[meta["y"].notna()]
    for t in CHECKPOINTS:
        for variant in ["ckfilter", "unfiltered"]:
            le = lab[lab["length_min"] >= t]
            steps = [
                ("source records", meta, None),
                ("excluded: pH 7.05-7.15 or missing (gapped label)", lab, n0 - len(lab)),
                (f"excluded: recording shorter than {t:g} min", le, len(lab) - len(le)),
            ]
            if variant == "ckfilter":
                qq = le.merge(q[["record_id", f"q{int(t)}"]], on="record_id")
                kept = qq[qq[f"q{int(t)}"] >= QUALITY_THRESHOLD]
                steps.append((f"excluded: FHR valid ratio in [0,{t:g}) < 0.60", kept, len(le) - len(kept)))
            for order, (step, df, excl) in enumerate(steps):
                y = df["y"] if "y" in df and df["y"].notna().all() else None
                rows.append({
                    "variant": variant, "checkpoint_minutes": t, "step_order": order, "step": step,
                    "n_excluded_at_step": excl,
                    "excl_cases_at_step": None,
                    "n_remaining": len(df),
                    "cases_remaining": int(y.sum()) if y is not None else None,
                    "prevalence_remaining": round(float(y.mean()), 4) if y is not None else None,
                })
    flow = pd.DataFrame(rows)
    # cases lost at each labelled step
    for (v, t), g in flow.groupby(["variant", "checkpoint_minutes"]):
        prev = np.nan
        for i in g.index:
            cur = flow.at[i, "cases_remaining"]
            if pd.notna(prev) and pd.notna(cur):
                flow.at[i, "excl_cases_at_step"] = int(prev - cur)
            prev = cur
    return flow


# ---------------------------------------------------------------- splits on disk

def split_path(variant: str, t: float) -> Path:
    return SPLITS_DIR / f"splits_{variant}_t{int(t)}.csv"


def make_and_save_splits(cohort: pd.DataFrame, variant: str, t: float) -> Path:
    y = cohort["y"].to_numpy()
    splits = precompute_quality_splits(y, 5, 3, SEED, 0.2)
    rows = []
    rid = cohort["record_id"].to_numpy()
    for sp in splits:
        for role, key in (("inner_train", "inner_train_idx"), ("val", "val_idx"), ("outer_test", "outer_test_idx")):
            for ix in np.asarray(sp[key], dtype=int):
                rows.append({"record_id": rid[ix], "repeat": int(sp["repeat"]),
                             "fold": int(sp["fold"]), "role": role})
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    p = split_path(variant, t)
    pd.DataFrame(rows).sort_values(["repeat", "fold", "role", "record_id"]).to_csv(p, index=False)
    return p


def load_splits(cohort: pd.DataFrame, variant: str, t: float) -> list[dict]:
    """Splits as index arrays into `cohort` rows, read from disk (never regenerated here)."""
    df = pd.read_csv(split_path(variant, t), dtype={"record_id": str})
    pos = {r: i for i, r in enumerate(cohort["record_id"])}
    if set(df["record_id"]) != set(pos):
        raise ValueError(f"split file {split_path(variant, t)} does not match cohort membership")
    out = []
    for (rep, fold), g in df.groupby(["repeat", "fold"], sort=True):
        d = {"repeat": int(rep), "fold": int(fold), "fold_name": f"repeat{rep:02d}_fold{fold:02d}"}
        for role, key in (("inner_train", "inner_train_idx"), ("val", "val_idx"), ("outer_test", "outer_test_idx")):
            d[key] = np.sort(np.array([pos[r] for r in g.loc[g.role == role, "record_id"]], dtype=int))
        d["outer_train_idx"] = np.sort(np.concatenate([d["inner_train_idx"], d["val_idx"]]))
        out.append(d)
    return out


def maternal_matrix(cohort: pd.DataFrame) -> np.ndarray:
    return cohort[MATERNAL_8].to_numpy(dtype=np.float64)
