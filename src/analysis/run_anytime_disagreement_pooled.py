"""Properly-powered (all-folds) version of run_anytime_disagreement.py.

The original script reports the gate-vs-correctness finding from ONE fold's
saved checkpoint (n=69 holdout, 5 positives), because run_anytime_cohort()
keeps only the best-val-PR-AUC checkpoint per model and discards the other 14.
This version pools gate values across all 15 fold checkpoints, reaching the
full cohort (n=345, 23 positives).

Design decisions, and why:

  * Gate values come from a SEPARATE cohort trained with --save-all-fold-models
    (default: primary_705_anytime_vr060_allfolds). The original cohort is never
    read for gate values and never written to -- its numbers are already cited
    in the Midterm Report and the manuscript draft. Same precedent as
    primary_705_vr060_gated_ab vs primary_705_vr060.

  * ctg_only / maternal_only predictions come from the ORIGINAL cohort's
    oof_predictions.csv. Those are already repeat-averaged across all 15 folds
    (aggregate_repeated_oof_anytime), i.e. already full-power -- only the gate
    values were ever the bottleneck, so retraining them would change nothing
    while risking the cited numbers.

  * Each fold's gate values are computed ONLY on that fold's own outer-test
    rows, using that fold's OWN saved preprocessors.joblib. Never on rows the
    fold trained on, and never with another fold's preprocessing.

  * Pooling uses the exact aggregate_repeated_oof_anytime() pattern
    (train_cv.py): group by record and checkpoint, take the mean across
    repeats. Each record appears in exactly one fold's test set per repeat, so
    every record contributes n_repeats gate values.

  * With --save-all-fold-models there is NO models/<model>/best/ directory --
    the layout is models/gated_fusion/repeat{NN}_fold{NN}/. That is why this is
    a separate script rather than a flag on the original one.

Outputs (written to the all-folds cohort dir, leaving the original untouched):
  anytime_disagreement_analysis_pooled.csv
  anytime_disagreement_summary_pooled.csv

The summary keeps every column of the original single-fold summary for direct
comparability, and adds gate_mean_when_ctg_right / gate_mean_when_maternal_right
-- the two figures the project's headline table actually reports, which the
original summary never contained (they had to be re-derived from the per-record
CSV each time).

Usage:
  python -m src.analysis.run_anytime_disagreement_pooled
  python -m src.analysis.run_anytime_disagreement_pooled \
      --gate-cohort-dir outputs/quality_cohorts/primary_705_anytime_vr060_allfolds \
      --oof-cohort-dir  outputs/quality_cohorts/primary_705_anytime_vr060
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch

from src.training.train_cv import (
    aggregate_repeated_oof_anytime,
    crop_ctg_array,
    deep_model_for_name,
    load_cohort_arrays,
)
from src.training.train_deep import transform_arrays

DEFAULT_GATE_COHORT = Path("outputs/quality_cohorts/primary_705_anytime_vr060_allfolds")
DEFAULT_OOF_COHORT = Path("outputs/quality_cohorts/primary_705_anytime_vr060")
PROCESSED_DIR = Path("data/processed")
CHECKPOINTS = [10.0, 20.0, 30.0, 45.0, 60.0, 90.0]
TARGET_SAMPLE_RATE_HZ = 1
MODEL_NAME = "gated_fusion"
FOLD_DIR_RE = re.compile(r"^repeat(\d+)_fold(\d+)$")


def discover_fold_dirs(gate_cohort_dir: Path) -> List[Dict[str, object]]:
    """Find every per-fold checkpoint directory. Raises loudly if the layout
    looks like a default (best-only) run rather than --save-all-fold-models,
    because silently falling back to one fold is exactly the bug this script
    exists to fix."""
    model_root = gate_cohort_dir / "models" / MODEL_NAME
    if not model_root.exists():
        raise FileNotFoundError(f"No {MODEL_NAME} models directory at {model_root}")

    folds = []
    for child in sorted(model_root.iterdir()):
        if not child.is_dir():
            continue
        match = FOLD_DIR_RE.match(child.name)
        if not match:
            continue
        folds.append({
            "repeat": int(match.group(1)),
            "fold": int(match.group(2)),
            "fold_name": child.name,
            "dir": child,
        })

    if not folds:
        if (model_root / "best").exists():
            raise RuntimeError(
                f"{model_root} contains only 'best/' -- this cohort was run WITHOUT "
                "--save-all-fold-models, so per-fold checkpoints do not exist. Re-run "
                "the cohort with that flag; this script cannot pool from a single checkpoint."
            )
        raise RuntimeError(f"No repeat{{NN}}_fold{{NN}} directories found under {model_root}")
    return folds


def encoder_kwargs_from_snapshot(gate_cohort_dir: Path) -> Dict[str, object]:
    """Rebuild the model with the same encoder configuration it was trained
    with. Absent deep_model block -> {} -> default encoder, matching every
    cohort that did not opt into the capacity overrides."""
    snapshot_path = gate_cohort_dir / "config_snapshot.json"
    if not snapshot_path.exists():
        return {}
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    overrides = snapshot.get("deep_model_overrides") or {}
    return dict(overrides.get("encoder", {}) or {})


def gate_values_for_fold(
    fold: Dict[str, object],
    X_ctg: np.ndarray,
    X_maternal: np.ndarray,
    y: np.ndarray,
    record_ids: pd.DataFrame,
    split_manifest: pd.DataFrame,
    encoder_kwargs: Dict[str, object],
) -> List[Dict[str, object]]:
    fold_dir = fold["dir"]
    checkpoint_path = fold_dir / f"{MODEL_NAME}.pt"
    preprocessor_path = fold_dir / "preprocessors.joblib"
    for path in (checkpoint_path, preprocessor_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path} -- fold {fold['fold_name']} is incomplete.")

    holdout = split_manifest[
        (split_manifest["repeat"] == fold["repeat"])
        & (split_manifest["fold"] == fold["fold"])
        & (split_manifest["outer_split"] == "test")
    ]
    holdout_idx = holdout["row_index"].to_numpy()
    if len(holdout_idx) == 0:
        raise RuntimeError(f"No outer-test rows found for {fold['fold_name']} in split_manifest.csv")

    # This fold's OWN preprocessors -- fitted on its inner-train split only.
    pre = joblib.load(preprocessor_path)
    X_ctg_norm, X_mat_norm = transform_arrays(
        X_ctg, X_maternal,
        pre["maternal_imputer"], pre["maternal_scaler"], pre["ctg_mean"], pre["ctg_std"],
    )

    model = deep_model_for_name(MODEL_NAME, X_ctg, X_maternal, encoder_kwargs)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    holdout_record_ids = record_ids.iloc[holdout_idx]["record_id"].astype(str).to_numpy()
    x_mat_t = torch.tensor(X_mat_norm[holdout_idx], dtype=torch.float32)

    rows = []
    for checkpoint_minutes in CHECKPOINTS:
        cropped = crop_ctg_array(X_ctg_norm[holdout_idx], checkpoint_minutes, TARGET_SAMPLE_RATE_HZ)
        x_ctg_t = torch.tensor(cropped, dtype=torch.float32)
        with torch.no_grad():
            logits, gate = model.forward_with_gate(x_ctg_t, x_mat_t)
            gated_proba = torch.sigmoid(logits).numpy()
            gate_values = gate.numpy()
        for local_i, rid in enumerate(holdout_record_ids):
            rows.append({
                "checkpoint_minutes": checkpoint_minutes,
                "record_id": rid,
                "repeat": fold["repeat"],
                "fold": fold["fold"],
                "y_true": int(y[holdout_idx[local_i]]),
                "gated_fusion_proba": float(gated_proba[local_i]),
                "gate_value": float(gate_values[local_i]),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Pooled all-folds gate/disagreement analysis.")
    parser.add_argument("--gate-cohort-dir", default=str(DEFAULT_GATE_COHORT),
                        help="Cohort with per-fold gated_fusion checkpoints (--save-all-fold-models).")
    parser.add_argument("--oof-cohort-dir", default=str(DEFAULT_OOF_COHORT),
                        help="Cohort providing ctg_only/maternal_only OOF predictions.")
    args = parser.parse_args()

    gate_cohort_dir = Path(args.gate_cohort_dir)
    oof_cohort_dir = Path(args.oof_cohort_dir)

    cv_summary = json.loads((gate_cohort_dir / "cv_summary.json").read_text(encoding="utf-8"))
    suffix = cv_summary["suffix"]

    folds = discover_fold_dirs(gate_cohort_dir)
    encoder_kwargs = encoder_kwargs_from_snapshot(gate_cohort_dir)
    print(f"Gate cohort:  {gate_cohort_dir}")
    print(f"OOF cohort:   {oof_cohort_dir}")
    print(f"Found {len(folds)} per-fold {MODEL_NAME} checkpoints")
    print(f"Encoder kwargs: {encoder_kwargs or '(defaults)'}\n")

    X_ctg, X_maternal, y, record_ids, _manifest = load_cohort_arrays(PROCESSED_DIR, suffix)
    split_manifest = pd.read_csv(gate_cohort_dir / "split_manifest.csv")

    per_fold_rows: List[Dict[str, object]] = []
    for fold in folds:
        print(f"  {fold['fold_name']} ...")
        per_fold_rows.extend(
            gate_values_for_fold(fold, X_ctg, X_maternal, y, record_ids, split_manifest, encoder_kwargs)
        )

    per_fold = pd.DataFrame(per_fold_rows)

    # Same pooling pattern as aggregate_repeated_oof_anytime(): mean across
    # repeats, keyed on record + checkpoint.
    pooled_gate = (
        per_fold.groupby(["checkpoint_minutes", "record_id"], as_index=False)
        .agg(
            y_true=("y_true", "first"),
            gate_value=("gate_value", "mean"),
            gated_fusion_proba=("gated_fusion_proba", "mean"),
            n_folds_pooled=("gate_value", "size"),
        )
    )

    # ctg_only / maternal_only from the ORIGINAL cohort -- already full-power.
    oof = pd.read_csv(oof_cohort_dir / "oof_predictions.csv")
    averaged_oof = aggregate_repeated_oof_anytime(oof)

    all_rows = []
    for checkpoint_minutes in CHECKPOINTS:
        ctg_ckpt = averaged_oof[
            (averaged_oof["checkpoint_minutes"] == checkpoint_minutes)
            & (averaged_oof["model"] == "ctg_only")
        ]
        mat_ckpt = averaged_oof[
            (averaged_oof["checkpoint_minutes"] == checkpoint_minutes)
            & (averaged_oof["model"] == "maternal_only")
        ]
        ctg_by_rid = dict(zip(ctg_ckpt["record_id"].astype(str), ctg_ckpt["y_proba"]))
        mat_by_rid = dict(zip(mat_ckpt["record_id"].astype(str), mat_ckpt["y_proba"]))

        gate_ckpt = pooled_gate[pooled_gate["checkpoint_minutes"] == checkpoint_minutes]
        for _, row in gate_ckpt.iterrows():
            rid = str(row["record_id"])
            if rid not in ctg_by_rid or rid not in mat_by_rid:
                continue
            ctg_p = float(ctg_by_rid[rid])
            mat_p = float(mat_by_rid[rid])
            disagreement = (ctg_p >= 0.5) != (mat_p >= 0.5)
            true_label = int(row["y_true"])
            ctg_closer = abs(ctg_p - true_label) < abs(mat_p - true_label)
            all_rows.append({
                "checkpoint_minutes": checkpoint_minutes,
                "record_id": rid,
                "y_true": true_label,
                "ctg_only_oof_proba": ctg_p,
                "maternal_only_oof_proba": mat_p,
                "disagreement": bool(disagreement),
                "ctg_reading_closer_to_truth": bool(ctg_closer) if disagreement else None,
                "gated_fusion_proba": float(row["gated_fusion_proba"]),
                "gate_value": float(row["gate_value"]),
                "n_folds_pooled": int(row["n_folds_pooled"]),
            })

    df = pd.DataFrame(all_rows)
    analysis_path = gate_cohort_dir / "anytime_disagreement_analysis_pooled.csv"
    df.to_csv(analysis_path, index=False)
    print(f"\nWrote {len(df)} rows to {analysis_path}")

    summary_rows = []
    for checkpoint_minutes, group in df.groupby("checkpoint_minutes"):
        disagree = group[group["disagreement"]]
        agree = group[~group["disagreement"]]
        ctg_right = disagree[disagree["ctg_reading_closer_to_truth"] == True]   # noqa: E712
        mat_right = disagree[disagree["ctg_reading_closer_to_truth"] == False]  # noqa: E712
        summary_rows.append({
            "checkpoint_minutes": checkpoint_minutes,
            "n_holdout": len(group),
            "n_pos_holdout": int(group["y_true"].sum()),
            "n_disagreement": len(disagree),
            "n_agreement": len(agree),
            "gate_mean_disagreement": float(disagree["gate_value"].mean()) if len(disagree) else float("nan"),
            "gate_mean_agreement": float(agree["gate_value"].mean()) if len(agree) else float("nan"),
            "n_disagreement_where_ctg_was_closer_to_truth": len(ctg_right),
            "n_disagreement_where_maternal_was_closer_to_truth": len(mat_right),
            # Added: the two figures the headline table actually reports.
            "gate_mean_when_ctg_right": float(ctg_right["gate_value"].mean()) if len(ctg_right) else float("nan"),
            "gate_mean_when_maternal_right": float(mat_right["gate_value"].mean()) if len(mat_right) else float("nan"),
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = gate_cohort_dir / "anytime_disagreement_summary_pooled.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote summary to {summary_path}\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
