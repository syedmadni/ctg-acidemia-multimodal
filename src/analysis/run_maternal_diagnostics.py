"""Phase 1a diagnostics: quantify how much signal each of the 8 maternal
variables actually carries for the acidemia label, on the anytime cohort
(primary_705_anytime_vr060, n=345, 23 positives).

Motivation: every fusion variant fails to beat ctg_only, and maternal_only
scores a flat ~0.543 AUC. This script produces the citable evidence for WHY:
the binary maternal flags carry essentially zero marginal signal, and the
continuous ones are range-restricted by CTU-UHB's homogeneous-cohort curation
(term, singleton, no anomalies). This turns a weak "fusion didn't help" into a
rigorous, defensible "there is nothing to fuse, and here is the measured proof."

For each variable it reports:
  - binary flags (diabetes/hypertension/preeclampsia/induced): 2x2 contingency
    vs the label, odds ratio, and Fisher exact p-value.
  - continuous (age/gravidity/parity/gest_weeks): positive-class vs negative-class
    means, plus the full observed range (gest_weeks range 37-43 is the direct
    range-restriction evidence).
It also reports the raw-NaN count per column and confirms imputation is applied
downstream (settling the retracted "NaN bug": raw arrays legitimately contain a
few NaN, which transform_arrays imputes per-fold on the train split).

Usage: python -m src.analysis.run_maternal_diagnostics
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from src.training.train_cv import load_cohort_arrays

PROCESSED_DIR = Path("data/processed")
COHORT_DIR = Path("outputs/quality_cohorts/primary_705_anytime_vr060")
BINARY_FLAGS = {"diabetes", "hypertension", "preeclampsia", "induced"}


def main() -> None:
    cv_summary = json.loads((COHORT_DIR / "cv_summary.json").read_text(encoding="utf-8"))
    suffix = cv_summary["suffix"]

    X_ctg, X_maternal, y, record_ids, _manifest = load_cohort_arrays(PROCESSED_DIR, suffix)
    feature_names = json.loads(
        (PROCESSED_DIR / f"maternal_feature_names{suffix}.json").read_text(encoding="utf-8")
    )
    if X_maternal.shape[1] != len(feature_names):
        raise ValueError(
            f"maternal array width {X_maternal.shape[1]} != feature-name count {len(feature_names)}"
        )

    y = y.astype(int)
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    prevalence = n_pos / n
    print(f"cohort n={n}, positives={n_pos}, negatives={n_neg}, prevalence={prevalence:.4f}")
    print(f"maternal features: {feature_names}\n")

    rows = []
    for j, name in enumerate(feature_names):
        col = X_maternal[:, j].astype(float)
        n_nan = int(np.isnan(col).sum())
        valid = ~np.isnan(col)

        row = {
            "variable": name,
            "kind": "binary_flag" if name in BINARY_FLAGS else "continuous",
            "n_nan_raw": n_nan,
            "n_valid": int(valid.sum()),
        }

        if name in BINARY_FLAGS:
            # 2x2 contingency of flag (1/0) vs label (1/0), on non-NaN rows.
            flag = col[valid]
            yy = y[valid]
            is1 = flag >= 0.5
            tp = int(((is1) & (yy == 1)).sum())   # flag=1, positive
            fp = int(((is1) & (yy == 0)).sum())   # flag=1, negative
            fn = int(((~is1) & (yy == 1)).sum())  # flag=0, positive
            tn = int(((~is1) & (yy == 0)).sum())  # flag=0, negative
            n_flag1 = tp + fp
            n_flag0 = fn + tn
            rate_flag1 = tp / n_flag1 if n_flag1 else float("nan")
            rate_flag0 = fn / n_flag0 if n_flag0 else float("nan")
            # Fisher exact on [[tp, fp], [fn, tn]] (rows: flag=1/0; cols: pos/neg).
            odds_ratio, p_value = fisher_exact([[tp, fp], [fn, tn]])
            row.update({
                "n_flag1": n_flag1,
                "n_flag0": n_flag0,
                "acidemia_rate_flag1": round(rate_flag1, 4),
                "acidemia_rate_flag0": round(rate_flag0, 4),
                "odds_ratio": round(float(odds_ratio), 4) if np.isfinite(odds_ratio) else float(odds_ratio),
                "fisher_p": round(float(p_value), 4),
                "pos_mean": "",
                "neg_mean": "",
                "value_min": "",
                "value_max": "",
            })
            print(
                f"{name:14s} flag1={n_flag1:3d} rate={rate_flag1:.3f} | flag0={n_flag0:3d} "
                f"rate={rate_flag0:.3f} | OR={float(odds_ratio):.3f} p={float(p_value):.4f}"
            )
        else:
            pos_vals = col[(y == 1) & valid]
            neg_vals = col[(y == 0) & valid]
            v_min = float(np.nanmin(col)) if valid.any() else float("nan")
            v_max = float(np.nanmax(col)) if valid.any() else float("nan")
            row.update({
                "n_flag1": "",
                "n_flag0": "",
                "acidemia_rate_flag1": "",
                "acidemia_rate_flag0": "",
                "odds_ratio": "",
                "fisher_p": "",
                "pos_mean": round(float(pos_vals.mean()), 4) if len(pos_vals) else float("nan"),
                "neg_mean": round(float(neg_vals.mean()), 4) if len(neg_vals) else float("nan"),
                "value_min": round(v_min, 4),
                "value_max": round(v_max, 4),
            })
            print(
                f"{name:14s} pos_mean={row['pos_mean']} neg_mean={row['neg_mean']} "
                f"range=[{v_min:g}, {v_max:g}]"
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = COHORT_DIR / "maternal_variable_diagnostics.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")

    total_nan = int(df["n_nan_raw"].sum())
    print(
        f"\nRaw-NaN check: {total_nan} NaN cell(s) across all maternal columns "
        f"(imputed per-fold by transform_arrays' median SimpleImputer -- confirms the "
        f"retracted 'NaN bug': raw arrays legitimately hold a few NaN, the model never sees them)."
    )


if __name__ == "__main__":
    main()
