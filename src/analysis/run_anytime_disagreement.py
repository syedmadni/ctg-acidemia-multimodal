"""Disagreement-case interpretability mini-analysis for the any-elapsed-time
cohort (primary_705_anytime_vr060).

Motivation (see CLAUDE.md / project discussion): the gate value in
GatedFusionModel is most informative exactly when the CTG-only and
maternal-only readings for a record point in *different* directions -- when
they already agree, the gate's exact split barely changes the final
prediction, so averaging gate value over all records (as the original
gate_interpretability.py does) dilutes whatever real signal exists. This
script isolates the "disagreement" subgroup specifically.

Method:
  1. Load gated_fusion's single saved best-fold checkpoint (same scope
     limitation as gate_interpretability.py: only one fold's ~1/15 holdout
     is available, since run_anytime_cohort() only keeps the best checkpoint
     per model) and run forward_with_gate() at each elapsed-time checkpoint
     on that fold's own held-out records (genuinely out-of-sample for that
     model; fold membership doesn't change across checkpoints, only how the
     CTG input is cropped).
  2. Pull ctg_only's and maternal_only's own repeat-averaged OOF predictions
     (from oof_predictions.csv, properly out-of-fold across all 15
     repeat/fold combinations -- a stronger source than re-deriving
     single-fold predictions) for that same held-out subset, at the same
     checkpoint.
  3. Flag "disagreement" records: ctg_only OOF proba and maternal_only OOF
     proba fall on opposite sides of 0.5.
  4. For disagreement records only, report the gate value and check whether
     it leans toward whichever single-modality reading was actually closer
     to the true label.

Usage: python -m src.analysis.run_anytime_disagreement
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.models.fusion_model import GatedFusionModel
from src.training.train_cv import (
    aggregate_repeated_oof_anytime,
    crop_ctg_array,
    load_cohort_arrays,
)
from src.training.train_deep import transform_arrays

COHORT_DIR = Path("outputs/quality_cohorts/primary_705_anytime_vr060")
PROCESSED_DIR = Path("data/processed")
CHECKPOINTS = [10.0, 20.0, 30.0, 45.0, 60.0, 90.0]
TARGET_SAMPLE_RATE_HZ = 1


def main() -> None:
    cv_summary = json.loads((COHORT_DIR / "cv_summary.json").read_text(encoding="utf-8"))
    suffix = cv_summary["suffix"]

    meta_path = COHORT_DIR / "models" / "gated_fusion" / "best" / "best_checkpoint_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fold_name = Path(meta["artifact_dir"]).name
    repeat = int(fold_name.split("_")[0].replace("repeat", ""))
    fold = int(fold_name.split("_")[1].replace("fold", ""))

    split_manifest = pd.read_csv(COHORT_DIR / "split_manifest.csv")
    holdout_rows = split_manifest[
        (split_manifest["repeat"] == repeat) & (split_manifest["fold"] == fold) & (split_manifest["outer_split"] == "test")
    ]
    holdout_idx = holdout_rows["row_index"].to_numpy()
    print(f"gated_fusion best checkpoint: {fold_name}, holdout size = {len(holdout_idx)}")

    X_ctg, X_maternal, y, record_ids, _manifest = load_cohort_arrays(PROCESSED_DIR, suffix)

    preprocessors = joblib.load(COHORT_DIR / "models" / "gated_fusion" / "best" / "preprocessors.joblib")
    X_ctg_norm, X_mat_norm = transform_arrays(
        X_ctg, X_maternal,
        preprocessors["maternal_imputer"], preprocessors["maternal_scaler"],
        preprocessors["ctg_mean"], preprocessors["ctg_std"],
    )

    model = GatedFusionModel(num_maternal_features=X_maternal.shape[1], input_channels=X_ctg.shape[-1])
    model.load_state_dict(torch.load(COHORT_DIR / "models" / "gated_fusion" / "best" / "gated_fusion.pt", map_location="cpu"))
    model.eval()

    oof = pd.read_csv(COHORT_DIR / "oof_predictions.csv")
    averaged_oof = aggregate_repeated_oof_anytime(oof)
    holdout_record_ids = record_ids.iloc[holdout_idx]["record_id"].astype(str).to_numpy()

    all_rows = []
    for checkpoint_minutes in CHECKPOINTS:
        x_mat_t = torch.tensor(X_mat_norm[holdout_idx], dtype=torch.float32)
        cropped = crop_ctg_array(X_ctg_norm[holdout_idx], checkpoint_minutes, TARGET_SAMPLE_RATE_HZ)
        x_ctg_t = torch.tensor(cropped, dtype=torch.float32)
        with torch.no_grad():
            logits, gate = model.forward_with_gate(x_ctg_t, x_mat_t)
            gated_proba = torch.sigmoid(logits).numpy()
            gate_values = gate.numpy()

        ctg_only_ckpt = averaged_oof[(averaged_oof["checkpoint_minutes"] == checkpoint_minutes) & (averaged_oof["model"] == "ctg_only")]
        mat_only_ckpt = averaged_oof[(averaged_oof["checkpoint_minutes"] == checkpoint_minutes) & (averaged_oof["model"] == "maternal_only")]
        ctg_only_by_rid = dict(zip(ctg_only_ckpt["record_id"].astype(str), ctg_only_ckpt["y_proba"]))
        mat_only_by_rid = dict(zip(mat_only_ckpt["record_id"].astype(str), mat_only_ckpt["y_proba"]))

        for local_i, rid in enumerate(holdout_record_ids):
            if rid not in ctg_only_by_rid or rid not in mat_only_by_rid:
                continue
            ctg_p = float(ctg_only_by_rid[rid])
            mat_p = float(mat_only_by_rid[rid])
            disagreement = (ctg_p >= 0.5) != (mat_p >= 0.5)
            true_label = int(y[holdout_idx[local_i]])
            ctg_closer = abs(ctg_p - true_label) < abs(mat_p - true_label)
            all_rows.append({
                "checkpoint_minutes": checkpoint_minutes,
                "record_id": rid,
                "y_true": true_label,
                "ctg_only_oof_proba": ctg_p,
                "maternal_only_oof_proba": mat_p,
                "disagreement": bool(disagreement),
                "ctg_reading_closer_to_truth": bool(ctg_closer) if disagreement else None,
                "gated_fusion_proba": float(gated_proba[local_i]),
                "gate_value": float(gate_values[local_i]),
            })

    df = pd.DataFrame(all_rows)
    out_path = COHORT_DIR / "anytime_disagreement_analysis.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")

    summary_rows = []
    for checkpoint_minutes, group in df.groupby("checkpoint_minutes"):
        disagree = group[group["disagreement"]]
        agree = group[~group["disagreement"]]
        n_disagree_ctg_closer = int(disagree["ctg_reading_closer_to_truth"].sum()) if len(disagree) else 0
        summary_rows.append({
            "checkpoint_minutes": checkpoint_minutes,
            "n_holdout": len(group),
            "n_pos_holdout": int(group["y_true"].sum()),
            "n_disagreement": len(disagree),
            "n_agreement": len(agree),
            "gate_mean_disagreement": float(disagree["gate_value"].mean()) if len(disagree) else float("nan"),
            "gate_mean_agreement": float(agree["gate_value"].mean()) if len(agree) else float("nan"),
            "n_disagreement_where_ctg_was_closer_to_truth": n_disagree_ctg_closer,
            "n_disagreement_where_maternal_was_closer_to_truth": len(disagree) - n_disagree_ctg_closer,
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = COHORT_DIR / "anytime_disagreement_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote summary to {summary_path}\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
