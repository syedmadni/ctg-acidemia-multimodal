"""Step 7d-seeds: check that --resume is equivalent to an uninterrupted run.

READ-ONLY on the cohort directory. For every fold saved in <cohort>/_progress/, the
fold's rows are rebuilt IN MEMORY using exactly the row-construction code of
run_anytime_cohort (train_cv.py), fed with that fold's own saved predictions and
train_info. These "in-memory" rows are what an uninterrupted run holds; the JSON-loaded
rows are what a resumed run holds. The check compares, per fold:

  1. row counts and key sets/order,
  2. per-value equality (exact, NaN-aware) and the Python type of every value,
  3. DataFrame columns and dtypes (esp. record_id),
  4. the full downstream path: a "resumed" frame (fold 1 from JSON, later folds in
     memory) vs an "uninterrupted" frame (all folds in memory) through
     pd.DataFrame -> aggregate_repeated_oof_anytime, with assert_frame_equal(check_exact).

Predictions themselves come from the JSON in both arms (the in-memory originals are gone
once a fold is written), so this tests the serialisation/merge path, not training.
Training-side independence of skipped folds comes from the per-fold reseed in
train_deep_fold_anytime (fold_seed = seed + repeat*100 + fold).

Usage:
  python -m src.analysis.check_resume_equivalence --config configs/step7d_cap64_gapped_seed43.yaml
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.training.evaluate import evaluate_predictions
from src.training.train_cv import (
    aggregate_repeated_oof_anytime,
    load_cohort_arrays,
    precompute_quality_splits,
    safe_ranking_metrics,
)


def _same(a: object, b: object) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    try:
        if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
            return (math.isnan(a) and math.isnan(b)) or float(a) == float(b)
    except TypeError:
        pass
    return a == b


def rebuild_fold_rows(payload, split, y, record_ids, cohort_name, valid_ratio_threshold,
                      label_name, suffix, checkpoints_minutes, threshold, models):
    """Mirror of the deep-model branch of run_anytime_cohort's row construction."""
    loaded_oof = pd.DataFrame(payload["oof_rows"])
    repeat, fold = int(split["repeat"]), int(split["fold"])
    outer_train_idx = np.asarray(split["outer_train_idx"], dtype=int)
    outer_test_idx = np.asarray(split["outer_test_idx"], dtype=int)
    inner_train_idx = np.asarray(split["inner_train_idx"], dtype=int)
    val_idx = np.asarray(split["val_idx"], dtype=int)
    fm_rows, oof_rows = [], []
    for model_name in models:
        train_info = payload["checkpoint_candidates"][model_name][0]
        for checkpoint_minutes in checkpoints_minutes:
            sub = loaded_oof[(loaded_oof.model == model_name)
                             & (loaded_oof.checkpoint_minutes == checkpoint_minutes)]
            assert list(sub.row_index) == list(outer_test_idx), "row order differs from split"
            test_proba = sub.y_proba.to_numpy(dtype=float)
            metrics = evaluate_predictions(y[outer_test_idx], test_proba, threshold=threshold)
            metrics.update(safe_ranking_metrics(y[outer_test_idx], test_proba))
            row = {
                "cohort_name": cohort_name, "valid_ratio_threshold": valid_ratio_threshold,
                "label": label_name, "suffix": suffix, "model": model_name,
                "checkpoint_minutes": checkpoint_minutes, "repeat": repeat, "fold": fold,
                "train_size": int(len(outer_train_idx)), "inner_train_size": int(len(inner_train_idx)),
                "val_size": int(len(val_idx)), "test_size": int(len(outer_test_idx)),
                "train_positives": int(y[outer_train_idx].sum()), "inner_train_positives": int(y[inner_train_idx].sum()),
                "val_positives": int(y[val_idx].sum()), "test_positives": int(y[outer_test_idx].sum()),
                "best_val_pr_auc": train_info["best_val_pr_auc"], "best_epoch": train_info["best_epoch"],
                "threshold": threshold,
            }
            row.update({k: v for k, v in train_info.items() if k.startswith("pretrain")})
            row.update(metrics)
            fm_rows.append(row)
            for local_i, row_idx in enumerate(outer_test_idx):
                oof_rows.append({
                    "cohort_name": cohort_name, "valid_ratio_threshold": valid_ratio_threshold,
                    "checkpoint_minutes": checkpoint_minutes, "repeat": repeat, "fold": fold,
                    "record_id": record_ids.iloc[row_idx]["record_id"], "row_index": int(row_idx),
                    "y_true": int(y[row_idx]), "y_proba": float(test_proba[local_i]), "model": model_name,
                })
    return fm_rows, oof_rows


def compare_rows(kind, mem, loaded, problems, type_notes):
    if len(mem) != len(loaded):
        problems.append(f"{kind}: row count {len(mem)} (memory) vs {len(loaded)} (json)")
        return
    for i, (m, l) in enumerate(zip(mem, loaded)):
        if list(m.keys()) != list(l.keys()):
            problems.append(f"{kind} row {i}: keys differ {list(m.keys())} vs {list(l.keys())}")
            return
        for k in m:
            if not _same(m[k], l[k]):
                problems.append(f"{kind} row {i} key {k}: {m[k]!r} vs {l[k]!r}")
            tm, tl = type(m[k]).__name__, type(l[k]).__name__
            if tm != tl:
                type_notes.setdefault((kind, k), set()).add(f"{tm} -> {tl}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    exp_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_cfg = yaml.safe_load(Path(exp_cfg["base_config"]).read_text(encoding="utf-8"))
    cohort = exp_cfg["cohorts"][0]
    cohort_name, suffix = str(cohort["name"]), str(cohort["suffix"])
    cv = exp_cfg["cross_validation"]
    checkpoints_minutes = [float(m) for m in exp_cfg["checkpoints_minutes"]]
    threshold = float(base_cfg["training"]["threshold"])
    models = list(exp_cfg["models"])
    progress_dir = Path(exp_cfg["output"]["base_dir"]) / cohort_name / "_progress"

    _, _, y, record_ids, _ = load_cohort_arrays(Path(base_cfg["paths"]["processed_dir"]), suffix)
    splits = {s["fold_name"]: s for s in precompute_quality_splits(
        y, int(cv["n_splits"]), int(cv["n_repeats"]), int(cv["seed"]),
        float(base_cfg["training"]["val_size"]))}

    files = sorted(progress_dir.glob("*.json"))
    print(f"cohort {cohort_name}: {len(files)} progress file(s): {[f.name for f in files]}")
    all_ok = True
    mem_fm_all, mem_oof_all, json_fm_all, json_oof_all = [], [], [], []
    for pf in files:
        payload = json.loads(pf.read_text(encoding="utf-8"))
        fm_mem, oof_mem = rebuild_fold_rows(
            payload, splits[payload["fold_name"]], y, record_ids, cohort_name,
            float(cohort["min_valid_ratio"]), exp_cfg["label"]["name"], suffix,
            checkpoints_minutes, threshold, models)
        problems, type_notes = [], {}
        compare_rows("fold_metric_rows", fm_mem, payload["fold_metric_rows"], problems, type_notes)
        compare_rows("oof_rows", oof_mem, payload["oof_rows"], problems, type_notes)
        dm, dl = pd.DataFrame(oof_mem), pd.DataFrame(payload["oof_rows"])
        dtype_diff = {c: (str(dm[c].dtype), str(dl[c].dtype)) for c in dm.columns if dm[c].dtype != dl[c].dtype}
        print(f"\n[{payload['fold_name']}] fold_metric_rows memory={len(fm_mem)} json={len(payload['fold_metric_rows'])}"
              f" | oof_rows memory={len(oof_mem)} json={len(payload['oof_rows'])}")
        print(f"  OOF columns identical: {list(dm.columns) == list(dl.columns)}")
        print(f"  record_id dtype: memory={dm.record_id.dtype} json={dl.record_id.dtype}; "
              f"element type memory={type(oof_mem[0]['record_id']).__name__} json={type(payload['oof_rows'][0]['record_id']).__name__}")
        print(f"  OOF DataFrame dtype differences: {dtype_diff or 'none'}")
        print(f"  per-value Python type changes (json round-trip): "
              f"{ {f'{k[0]}.{k[1]}': sorted(v) for k, v in type_notes.items()} or 'none'}")
        print(f"  value mismatches: {len(problems)}" + ("" if not problems else f" e.g. {problems[:3]}"))
        all_ok &= not problems and not dtype_diff
        mem_fm_all += fm_mem; mem_oof_all += oof_mem
        json_fm_all.append(payload["fold_metric_rows"]); json_oof_all.append(payload["oof_rows"])

    if files:
        # resumed run = first fold from JSON, remaining folds in memory
        resumed_oof = json_oof_all[0] + [r for r in mem_oof_all[len(json_oof_all[0]):]]
        resumed_fm = json_fm_all[0] + mem_fm_all[len(json_fm_all[0]):]
        for label, a, b in [("fold_metrics", mem_fm_all, resumed_fm), ("oof_predictions", mem_oof_all, resumed_oof)]:
            da, db = pd.DataFrame(a), pd.DataFrame(b)
            pd.testing.assert_frame_equal(da, db, check_exact=True)
            print(f"\n{label}: uninterrupted vs resumed DataFrame -> assert_frame_equal(check_exact) PASSED "
                  f"({len(da)} rows, dtypes: {dict(da.dtypes.astype(str))})" if label == "oof_predictions"
                  else f"\n{label}: uninterrupted vs resumed DataFrame -> assert_frame_equal(check_exact) PASSED ({len(da)} rows)")
        agg_a = aggregate_repeated_oof_anytime(pd.DataFrame(mem_oof_all))
        agg_b = aggregate_repeated_oof_anytime(pd.DataFrame(resumed_oof))
        pd.testing.assert_frame_equal(agg_a, agg_b, check_exact=True)
        print(f"aggregate_repeated_oof_anytime: uninterrupted vs resumed -> IDENTICAL ({len(agg_a)} rows)")
    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
