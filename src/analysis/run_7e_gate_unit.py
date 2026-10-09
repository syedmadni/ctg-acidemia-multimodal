"""7e gate test: unit of analysis (requested 2026-09-22).

Each patient is scored once per CV repeat (3 repeats), so a patient contributes 3 gate
values per checkpoint. This script reports the gate test (CTG-right vs maternal-right
disagreement cases) three ways, for both encoder sizes:

  A. PATIENT-LEVEL, PRIMARY: each patient's gate averaged over its 3 repeats (the
     pooled file written by run_anytime_disagreement_pooled.py, n=345 per checkpoint),
     then a two-sided Mann-Whitney U. This is what 19.14 reported.
  B. PATIENT-LEVEL BOOTSTRAP: patients resampled with replacement within each group
     (CTG-right, maternal-right), 2000 iterations, seed 42, percentile 95% CI for the gap
     and the proportion of resampled gaps <= 0.
  C. NAIVE, for contrast only: every patient-repeat row treated as independent (n = 3x),
     Mann-Whitney U. Rows are regenerated from the saved per-fold checkpoints, using the
     same fold-own preprocessing as the pooled script.

Disagreement group membership (CTG-right / maternal-right) is patient-level in all three
versions: it comes from the repeat-averaged seed-42 ctg_only / maternal_only OOF, exactly
as in the pooled analysis. Holm across the 5 checkpoints within each (size, version).

Self-check: C's rows averaged per patient must reproduce A's gate values (max |diff| < 1e-12).

Outputs: maternal_value/step7e_gate_unit_of_analysis.csv
Usage:   python -m src.analysis.run_7e_gate_unit
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from src.analysis.run_anytime_disagreement_pooled import (
    discover_fold_dirs,
    encoder_kwargs_from_snapshot,
    gate_values_for_fold,
)
from src.training.train_cv import load_cohort_arrays

BASE = Path("outputs/quality_cohorts")
OUT = BASE / "primary_705_anytime_vr060" / "maternal_value" / "step7e_gate_unit_of_analysis.csv"
CHECKPOINTS = [10.0, 20.0, 30.0, 45.0, 60.0]
SIZES = {"full": "full-size", "cap64": "reduced"}
BOOT = 2000
SEED = 42


def per_repeat_rows(gate_dir: Path) -> pd.DataFrame:
    suffix = json.loads((gate_dir / "cv_summary.json").read_text(encoding="utf-8"))["suffix"]
    X_ctg, X_mat, y, record_ids, _ = load_cohort_arrays(Path("data/processed"), suffix)
    manifest = pd.read_csv(gate_dir / "split_manifest.csv")
    enc = encoder_kwargs_from_snapshot(gate_dir)
    rows = []
    for fold in discover_fold_dirs(gate_dir):
        rows.extend(gate_values_for_fold(fold, X_ctg, X_mat, y, record_ids, manifest, enc))
    df = pd.DataFrame(rows)
    df["record_id"] = df["record_id"].astype(str)
    return df[df.checkpoint_minutes.isin(CHECKPOINTS)]


def mwu(a, b):
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def main():
    rng = np.random.default_rng(SEED)
    out = []
    for size, label in SIZES.items():
        gate_dir = BASE / f"step7e_gate_allfolds_{size}"
        pooled = pd.read_csv(gate_dir / "anytime_disagreement_analysis_pooled.csv", float_precision="round_trip")
        pooled["record_id"] = pooled["record_id"].astype(str)
        pooled = pooled[pooled.checkpoint_minutes.isin(CHECKPOINTS)]
        reps = per_repeat_rows(gate_dir)

        # self-check: repeat rows averaged per patient == pooled patient gate
        avg = reps.groupby(["checkpoint_minutes", "record_id"], as_index=False)["gate_value"].mean()
        chk = pooled.merge(avg, on=["checkpoint_minutes", "record_id"], suffixes=("", "_rep"))
        maxdiff = float(np.max(np.abs(chk.gate_value - chk.gate_value_rep)))
        assert len(chk) == len(pooled) == 5 * 345 and maxdiff < 1e-12, (len(chk), maxdiff)
        assert reps.groupby(["checkpoint_minutes", "record_id"]).size().eq(3).all()
        print(f"[{label}] self-check: repeat rows -> patient means reproduce pooled gate (max |diff| {maxdiff:.1e})")

        rows = []
        for ck in CHECKPOINTS:
            p = pooled[(pooled.checkpoint_minutes == ck) & pooled.disagreement]
            grp = p.set_index("record_id")["ctg_reading_closer_to_truth"].astype(bool)
            ctg_ids, mat_ids = grp[grp].index, grp[~grp].index
            a = p.set_index("record_id").loc[ctg_ids, "gate_value"].to_numpy()
            b = p.set_index("record_id").loc[mat_ids, "gate_value"].to_numpy()
            gap = a.mean() - b.mean()

            boot = np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(BOOT)])
            lo, hi = np.percentile(boot, [2.5, 97.5])

            r = reps[reps.checkpoint_minutes == ck]
            ra = r[r.record_id.isin(ctg_ids)]["gate_value"].to_numpy()
            rb = r[r.record_id.isin(mat_ids)]["gate_value"].to_numpy()

            rows.append({"size": label, "checkpoint_minutes": ck,
                         "n_patients_ctg_right": len(a), "n_patients_maternal_right": len(b),
                         "gap_patient_level": gap,
                         "A_patient_mwu_p": mwu(a, b),
                         "B_boot_ci_low": lo, "B_boot_ci_high": hi, "B_boot_prop_gap_le_0": float(np.mean(boot <= 0)),
                         "n_rows_naive": len(ra) + len(rb),
                         "gap_naive": ra.mean() - rb.mean(),
                         "C_naive_rows_mwu_p": mwu(ra, rb)})
        s = pd.DataFrame(rows)
        s["A_patient_mwu_holm_p"] = multipletests(s.A_patient_mwu_p, method="holm")[1]
        s["B_boot_holm_p"] = multipletests(np.maximum(2 * np.minimum(s.B_boot_prop_gap_le_0, 1 - s.B_boot_prop_gap_le_0),
                                                      1.0 / BOOT), method="holm")[1]
        s["C_naive_rows_mwu_holm_p"] = multipletests(s.C_naive_rows_mwu_p, method="holm")[1]
        out.append(s)

    res = pd.concat(out, ignore_index=True)
    cols = ["size", "checkpoint_minutes", "n_patients_ctg_right", "n_patients_maternal_right", "gap_patient_level",
            "A_patient_mwu_p", "A_patient_mwu_holm_p", "B_boot_ci_low", "B_boot_ci_high", "B_boot_prop_gap_le_0",
            "B_boot_holm_p", "n_rows_naive", "gap_naive", "C_naive_rows_mwu_p", "C_naive_rows_mwu_holm_p"]
    res = res[cols]
    res.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    print(res.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
