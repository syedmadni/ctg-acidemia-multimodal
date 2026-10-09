"""Final phase: paired significance tests for the deep-model improvement
experiments (Phase 3/6). Answers, honestly: are the gains statistically real,
or point-estimate noise at 23 positives?

All comparisons are paired on the SAME held-out records at the SAME checkpoint
(every cohort here was built from the same _705_anytime_vr060 arrays with the
same 5x3 seed-42 CV, so record sets and folds are identical -- verified by an
inner join on (checkpoint, record_id) with a y_true consistency check).

Comparisons (each at every checkpoint):
  1. shrink: baseline ctg_only          vs cap64 ctg_only
  2. ablation: cap64 ctg_only           vs cnn_only ctg_only (Transformer effect)
  3. ensemble: LR (baseline cohort)     vs cap64 ctg_only + LR ensemble
  4. pretrain: cap64 gated_fusion        vs pretrain gated_fusion

Writes outputs/quality_cohorts/improvement_significance_tests.csv.

Usage: python -m src.analysis.run_improvement_significance
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.analysis.significance import delong_test, paired_permutation_test
from src.training.train_cv import aggregate_repeated_oof_anytime

OUT_BASE = Path("outputs/quality_cohorts")
CHECKPOINTS = [10.0, 20.0, 30.0, 45.0, 60.0, 90.0]
N_PERM = 2000
SEED = 42


def load(cohort: str) -> pd.DataFrame:
    return aggregate_repeated_oof_anytime(pd.read_csv(OUT_BASE / cohort / "oof_predictions.csv"))


def series(df: pd.DataFrame, model: str, ck: float) -> pd.DataFrame:
    return df[(df["model"] == model) & (df["checkpoint_minutes"] == ck)][["record_id", "y_true", "y_proba"]]


def ensemble_series(deep_df: pd.DataFrame, lr_df: pd.DataFrame, deep_model: str, ck: float) -> pd.DataFrame:
    d = series(deep_df, deep_model, ck).rename(columns={"y_proba": "pd"})
    lr = series(lr_df, "logistic_regression", ck).rename(columns={"y_proba": "plr"})
    m = d.merge(lr, on=["record_id", "y_true"], how="inner")
    m["y_proba"] = 0.5 * m["pd"] + 0.5 * m["plr"]
    return m[["record_id", "y_true", "y_proba"]]


def compare(name: str, ck: float, a: pd.DataFrame, b: pd.DataFrame) -> Dict[str, object]:
    m = a.rename(columns={"y_proba": "pa"}).merge(
        b.rename(columns={"y_proba": "pb"}), on=["record_id", "y_true"], how="inner"
    )
    y = m["y_true"].astype(int).to_numpy()
    pa = m["pa"].to_numpy()
    pb = m["pb"].to_numpy()
    dl = delong_test(y, pa, pb)
    perm = paired_permutation_test(y, pa, pb, n_permutations=N_PERM, seed=SEED)
    return {
        "comparison": name,
        "checkpoint_minutes": ck,
        "n_paired": len(m),
        "n_pos": int(y.sum()),
        "auc_a": round(dl["auc_a"], 4),
        "auc_b": round(dl["auc_b"], 4),
        "auc_diff_b_minus_a": round(dl["auc_b"] - dl["auc_a"], 4),
        "delong_p": round(dl["p_value"], 4),
        "permutation_p": round(perm["p_value"], 4),
    }


def main() -> None:
    baseline = load("primary_705_anytime_vr060")
    cap64 = load("anytime_cap64")
    cnn_only = load("anytime_cnn_only")
    pretrain = load("anytime_cap64_pretrain")

    rows: List[Dict[str, object]] = []
    for ck in CHECKPOINTS:
        rows.append(compare("shrink: baseline_ctg_only -> cap64_ctg_only", ck,
                             series(baseline, "ctg_only", ck), series(cap64, "ctg_only", ck)))
        rows.append(compare("ablation: cap64_ctg_only -> cnn_only_ctg_only", ck,
                             series(cap64, "ctg_only", ck), series(cnn_only, "ctg_only", ck)))
        rows.append(compare("ensemble: LR -> cap64_ctg_only+LR", ck,
                             series(baseline, "logistic_regression", ck),
                             ensemble_series(cap64, baseline, "ctg_only", ck)))
        rows.append(compare("pretrain: cap64_gated -> pretrain_gated", ck,
                             series(cap64, "gated_fusion", ck), series(pretrain, "gated_fusion", ck)))

    df = pd.DataFrame(rows)
    out = OUT_BASE / "improvement_significance_tests.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} comparisons to {out}\n")

    for name, g in df.groupby("comparison", sort=False):
        sig = g[g["delong_p"] < 0.05]
        best = g.loc[g["auc_diff_b_minus_a"].idxmax()]
        print(f"{name}")
        print(f"   mean AUC delta = {g['auc_diff_b_minus_a'].mean():+.4f} | "
              f"best {best['auc_diff_b_minus_a']:+.4f} @ {best['checkpoint_minutes']:.0f}min "
              f"(DeLong p={best['delong_p']}) | comparisons with p<0.05: {len(sig)}/{len(g)}")


if __name__ == "__main__":
    main()
