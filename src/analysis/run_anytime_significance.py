"""Paired DeLong + permutation significance tests for the any-elapsed-time
cohort (primary_705_anytime_vr060).

Unlike the earlier six-separate-fixed-window-cohort significance suite
(run_significance_tests.py), every comparison here is naturally paired by
construction: it's the SAME trained model and the SAME held-out records,
just evaluated at a different elapsed-time checkpoint (or a different
architecture at the same checkpoint) -- no cross-cohort record-set inner
join is needed, since checkpoint_minutes is a column within one cohort's
OOF predictions, not a different cohort.

Three comparison families:
  1. checkpoint_vs_checkpoint (ctg_only): does more elapsed time
     significantly improve prediction? 10-min is compared against each of
     the other five checkpoints.
  2. within_checkpoint_fusion_family: fusion/gated_fusion vs. ctg_only, at
     every checkpoint.
  3. within_checkpoint_baseline: logistic_regression/random_forest vs.
     ctg_only, at every checkpoint.

Usage: python -m src.analysis.run_anytime_significance
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.analysis.significance import delong_test, paired_permutation_test
from src.training.train_cv import aggregate_repeated_oof_anytime

COHORT_DIR = Path("outputs/quality_cohorts/primary_705_anytime_vr060")
OUTPUT_PATH = COHORT_DIR / "anytime_significance_tests.csv"
N_PERMUTATIONS = 2000
SEED = 42

CHECKPOINTS = [10.0, 20.0, 30.0, 45.0, 60.0, 90.0]
BASELINE_CHECKPOINT = 10.0


def load_averaged_oof() -> pd.DataFrame:
    oof = pd.read_csv(COHORT_DIR / "oof_predictions.csv")
    return aggregate_repeated_oof_anytime(oof)


def paired_arrays(averaged_oof: pd.DataFrame, checkpoint_a: float, model_a: str, checkpoint_b: float, model_b: str):
    a = averaged_oof[(averaged_oof["checkpoint_minutes"] == checkpoint_a) & (averaged_oof["model"] == model_a)][
        ["record_id", "y_true", "y_proba"]
    ].rename(columns={"y_true": "y_true_a", "y_proba": "proba_a"})
    b = averaged_oof[(averaged_oof["checkpoint_minutes"] == checkpoint_b) & (averaged_oof["model"] == model_b)][
        ["record_id", "y_true", "y_proba"]
    ].rename(columns={"y_true": "y_true_b", "y_proba": "proba_b"})
    merged = a.merge(b, on="record_id", how="inner")
    mismatched = merged[merged["y_true_a"] != merged["y_true_b"]]
    if len(mismatched) > 0:
        raise ValueError(f"y_true mismatch on {len(mismatched)} shared record(s) -- should never happen within one cohort.")
    return merged["y_true_a"].to_numpy(), merged["proba_a"].to_numpy(), merged["proba_b"].to_numpy(), len(merged)


def run_comparison(
    averaged_oof: pd.DataFrame,
    comparison_type: str,
    checkpoint_a: float,
    model_a: str,
    checkpoint_b: float,
    model_b: str,
) -> Dict[str, object]:
    y_true, proba_a, proba_b, n_paired = paired_arrays(averaged_oof, checkpoint_a, model_a, checkpoint_b, model_b)
    dl = delong_test(y_true, proba_a, proba_b)
    perm = paired_permutation_test(y_true, proba_a, proba_b, n_permutations=N_PERMUTATIONS, seed=SEED)
    return {
        "comparison_type": comparison_type,
        "checkpoint_minutes_a": checkpoint_a,
        "model_a": model_a,
        "checkpoint_minutes_b": checkpoint_b,
        "model_b": model_b,
        "n_paired_records": n_paired,
        "n_pos_paired": dl["n_pos"],
        "auc_a": dl["auc_a"],
        "auc_b": dl["auc_b"],
        "auc_diff": dl["auc_diff"],
        "delong_z": dl["z"],
        "delong_p": dl["p_value"],
        "permutation_p": perm["p_value"],
        "permutation_iterations": perm["n_permutations"],
    }


def main() -> None:
    averaged_oof = load_averaged_oof()
    rows: List[Dict[str, object]] = []

    for checkpoint in CHECKPOINTS:
        if checkpoint == BASELINE_CHECKPOINT:
            continue
        print(f"checkpoint_vs_checkpoint: ctg_only@{BASELINE_CHECKPOINT}min vs ctg_only@{checkpoint}min")
        rows.append(run_comparison(
            averaged_oof, "checkpoint_vs_checkpoint", BASELINE_CHECKPOINT, "ctg_only", checkpoint, "ctg_only",
        ))

    for checkpoint in CHECKPOINTS:
        for fusion_model in ["fusion", "gated_fusion"]:
            print(f"within_checkpoint_fusion_family: ctg_only vs {fusion_model} @ {checkpoint}min")
            rows.append(run_comparison(
                averaged_oof, "within_checkpoint_fusion_family", checkpoint, "ctg_only", checkpoint, fusion_model,
            ))
        for baseline_model in ["logistic_regression", "random_forest"]:
            print(f"within_checkpoint_baseline: ctg_only vs {baseline_model} @ {checkpoint}min")
            rows.append(run_comparison(
                averaged_oof, "within_checkpoint_baseline", checkpoint, "ctg_only", checkpoint, baseline_model,
            ))

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(df)} comparisons to {OUTPUT_PATH}")

    notes = {
        "method": "Paired DeLong (1988) test + paired permutation test (2000 iterations, seed 42). Pairing is exact by construction -- same trained model, same held-out records, only checkpoint_minutes and/or architecture differ.",
        "significance_threshold": 0.05,
        "n_comparisons": len(df),
        "multiple_comparisons_correction": (
            f"none applied -- with {len(df)} comparisons run here, treat any single p<0.05 as suggestive, "
            f"not confirmatory, unless it also survives a Bonferroni-style threshold (~0.05/{len(df)}) "
            "or is a pre-registered primary comparison (the checkpoint_vs_checkpoint family, 5 comparisons, "
            "is the pre-registered primary family -- alpha=0.05/5=0.01 for that family specifically)."
        ),
    }
    (COHORT_DIR / "anytime_significance_tests_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")

    print("\n=== Comparisons with DeLong p < 0.05 ===")
    sig = df[df["delong_p"] < 0.05]
    if len(sig) == 0:
        print("(none)")
    else:
        print(sig[["comparison_type", "checkpoint_minutes_a", "model_a", "checkpoint_minutes_b", "model_b", "auc_diff", "delong_p", "permutation_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
