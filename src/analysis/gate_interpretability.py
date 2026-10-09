"""Extract per-record GatedFusionModel gate values on genuinely held-out data,
without retraining anything.

Every gated_fusion cohort was trained with only the single best-val-PR-AUC
fold's checkpoint kept on disk (run_quality_cohorts() deletes the other 14
fold checkpoints once the best one is copied out -- see train_cv.py). That
means we cannot reconstruct a full 15-fold repeat-averaged OOF gate value per
record the way cv_summary.json's headline metrics are computed. What we CAN
do honestly, with zero retraining: load that single saved "best" checkpoint,
look up which specific outer_test_idx rows it was actually evaluated on
(recorded in best_checkpoint_metadata.json's artifact_dir, cross-referenced
against split_manifest.csv), and run forward_with_gate() on exactly that
held-out subset. This is genuinely out-of-sample for that one model (the
model never trained on these rows), just smaller in n than a full OOF sweep
(~1/5 of the cohort, one fold's worth) -- state that scope limitation
wherever these numbers are cited, don't imply it's a full repeat-averaged OOF
gate analysis.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import torch

from src.models.fusion_model import GatedFusionModel
from src.training.train_cv import crop_ctg_array, load_cohort_arrays
from src.training.train_deep import transform_arrays


def load_gate_holdout_at_checkpoint(
    cohort_dir: Path,
    processed_dir: Path,
    checkpoint_minutes: float,
    target_sample_rate_hz: int,
) -> pd.DataFrame:
    """Anytime-cohort counterpart to load_gate_holdout(): the gated_fusion
    "best" checkpoint is still a single fold's model (same scope limitation
    as documented above), but that one model was trained with random elapsed-
    time crops, so it can be evaluated at any chosen checkpoint -- crop the
    held-out rows' CTG arrays to checkpoint_minutes (same crop-and-zero
    convention as crop_ctg_array/AnyTimeCTGDataset) before running
    forward_with_gate()."""
    cv_summary = json.loads((cohort_dir / "cv_summary.json").read_text(encoding="utf-8"))
    suffix = cv_summary["suffix"]
    cohort_name = cv_summary["cohort_name"]

    meta_path = cohort_dir / "models" / "gated_fusion" / "best" / "best_checkpoint_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fold_name = Path(meta["artifact_dir"]).name  # e.g. "repeat03_fold05"
    repeat = int(fold_name.split("_")[0].replace("repeat", ""))
    fold = int(fold_name.split("_")[1].replace("fold", ""))

    split_manifest = pd.read_csv(cohort_dir / "split_manifest.csv")
    holdout_rows = split_manifest[
        (split_manifest["repeat"] == repeat)
        & (split_manifest["fold"] == fold)
        & (split_manifest["outer_split"] == "test")
    ]
    holdout_idx = holdout_rows["row_index"].to_numpy()

    X_ctg, X_maternal, y, record_ids, _manifest = load_cohort_arrays(processed_dir, suffix)

    preprocessors = joblib.load(cohort_dir / "models" / "gated_fusion" / "best" / "preprocessors.joblib")
    X_ctg_norm, X_mat_norm = transform_arrays(
        X_ctg, X_maternal,
        preprocessors["maternal_imputer"], preprocessors["maternal_scaler"],
        preprocessors["ctg_mean"], preprocessors["ctg_std"],
    )
    X_ctg_norm = crop_ctg_array(X_ctg_norm, checkpoint_minutes, target_sample_rate_hz)

    model = GatedFusionModel(num_maternal_features=X_maternal.shape[1], input_channels=X_ctg.shape[-1])
    model.load_state_dict(torch.load(cohort_dir / "models" / "gated_fusion" / "best" / "gated_fusion.pt", map_location="cpu"))
    model.eval()

    x_ctg_t = torch.tensor(X_ctg_norm[holdout_idx], dtype=torch.float32)
    x_mat_t = torch.tensor(X_mat_norm[holdout_idx], dtype=torch.float32)
    with torch.no_grad():
        logits, gate = model.forward_with_gate(x_ctg_t, x_mat_t)
        proba = torch.sigmoid(logits).numpy()
        gate_values = gate.numpy()

    rows: list[Dict[str, object]] = []
    for local_i, row_idx in enumerate(holdout_idx):
        record_row = record_ids.iloc[row_idx]
        rows.append({
            "cohort_name": cohort_name,
            "checkpoint_minutes": checkpoint_minutes,
            "record_id": record_row["record_id"],
            "row_index": int(row_idx),
            "y_true": int(y[row_idx]),
            "y_proba": float(proba[local_i]),
            "gate_value": float(gate_values[local_i]),
            "valid_ratio": float(record_row["valid_ratio"]),
        })
    return pd.DataFrame(rows)


def load_gate_holdout(cohort_dir: Path, processed_dir: Path) -> pd.DataFrame:
    cv_summary = json.loads((cohort_dir / "cv_summary.json").read_text(encoding="utf-8"))
    suffix = cv_summary["suffix"]
    cohort_name = cv_summary["cohort_name"]

    meta_path = cohort_dir / "models" / "gated_fusion" / "best" / "best_checkpoint_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fold_name = Path(meta["artifact_dir"]).name  # e.g. "repeat03_fold05"
    repeat = int(fold_name.split("_")[0].replace("repeat", ""))
    fold = int(fold_name.split("_")[1].replace("fold", ""))

    split_manifest = pd.read_csv(cohort_dir / "split_manifest.csv")
    holdout_rows = split_manifest[
        (split_manifest["repeat"] == repeat)
        & (split_manifest["fold"] == fold)
        & (split_manifest["outer_split"] == "test")
    ]
    holdout_idx = holdout_rows["row_index"].to_numpy()

    X_ctg, X_maternal, y, record_ids, _manifest = load_cohort_arrays(processed_dir, suffix)

    preprocessors = joblib.load(cohort_dir / "models" / "gated_fusion" / "best" / "preprocessors.joblib")
    X_ctg_norm, X_mat_norm = transform_arrays(
        X_ctg, X_maternal,
        preprocessors["maternal_imputer"], preprocessors["maternal_scaler"],
        preprocessors["ctg_mean"], preprocessors["ctg_std"],
    )

    model = GatedFusionModel(num_maternal_features=X_maternal.shape[1], input_channels=X_ctg.shape[-1])
    model.load_state_dict(torch.load(cohort_dir / "models" / "gated_fusion" / "best" / "gated_fusion.pt", map_location="cpu"))
    model.eval()

    x_ctg_t = torch.tensor(X_ctg_norm[holdout_idx], dtype=torch.float32)
    x_mat_t = torch.tensor(X_mat_norm[holdout_idx], dtype=torch.float32)
    with torch.no_grad():
        logits, gate = model.forward_with_gate(x_ctg_t, x_mat_t)
        proba = torch.sigmoid(logits).numpy()
        gate_values = gate.numpy()

    rows: list[Dict[str, object]] = []
    for local_i, row_idx in enumerate(holdout_idx):
        record_row = record_ids.iloc[row_idx]
        rows.append({
            "cohort_name": cohort_name,
            "record_id": record_row["record_id"],
            "row_index": int(row_idx),
            "y_true": int(y[row_idx]),
            "y_proba": float(proba[local_i]),
            "gate_value": float(gate_values[local_i]),
            "valid_ratio": float(record_row["valid_ratio"]),
            "window_minutes": int(record_row.get("window_minutes", -1)),
            "window_position": str(record_row.get("window_position", "unknown")),
        })
    return pd.DataFrame(rows)
