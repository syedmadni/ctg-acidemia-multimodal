"""Phase 3 validation for the all-folds gate rerun.

Four checks, run in order of severity. Any failure exits non-zero and says
what broke rather than printing a table that looks fine.

  1. DETERMINISM CROSS-CHECK (the important one). The all-folds cohort re-trains
     gated_fusion with the same seed, splits, data and hyperparameters as the
     original cohort. train_deep_fold_anytime() calls set_global_seed(fold_seed)
     per (model, fold), so training gated_fusion ALONE must reproduce the run
     where it trained alongside five other models -- bit-exactly. If OOF
     predictions differ at all, either determinism regressed or the configs
     diverged, and the pooled gate values cannot be compared to the published
     single-fold ones. This is a hard gate: it must pass before anything below
     is trustworthy.

  2. Pooled coverage: n=345 records, 23 positives, at every checkpoint.

  3. Fold structure: every record appears in exactly one fold's outer-test set
     per repeat -- 3 gate values pooled per record, no duplicates, no gaps.

  4. Side-by-side comparison of the pooled table against the original
     single-fold table, with the deltas made explicit so the finding can be
     judged rather than asserted.

Usage:
  python -m src.analysis.validate_gate_allfolds
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.train_cv import aggregate_repeated_oof_anytime

DEFAULT_ALLFOLDS = Path("outputs/quality_cohorts/primary_705_anytime_vr060_allfolds")
DEFAULT_ORIGINAL = Path("outputs/quality_cohorts/primary_705_anytime_vr060")
EXPECTED_N = 345
EXPECTED_POSITIVES = 23
EXPECTED_REPEATS = 3
CHECKPOINTS = [10.0, 20.0, 30.0, 45.0, 60.0, 90.0]


def check_determinism(allfolds_dir: Path, original_dir: Path) -> bool:
    print("=" * 78)
    print("CHECK 1 -- DETERMINISM CROSS-CHECK (gated_fusion OOF, new vs original)")
    print("=" * 78)

    new_oof = pd.read_csv(allfolds_dir / "oof_predictions.csv")
    old_oof = pd.read_csv(original_dir / "oof_predictions.csv")

    new_g = new_oof[new_oof["model"] == "gated_fusion"].copy()
    old_g = old_oof[old_oof["model"] == "gated_fusion"].copy()

    key = ["checkpoint_minutes", "repeat", "fold", "record_id"]
    new_g = new_g.sort_values(key).reset_index(drop=True)
    old_g = old_g.sort_values(key).reset_index(drop=True)

    print(f"  new rows: {len(new_g)}   original rows: {len(old_g)}")
    if len(new_g) != len(old_g):
        print("  FAIL: row counts differ -- the two runs do not cover the same fold/record grid.")
        return False

    merged = new_g.merge(old_g, on=key, suffixes=("_new", "_old"), validate="one_to_one")
    if len(merged) != len(new_g):
        print(f"  FAIL: join produced {len(merged)} rows from {len(new_g)} -- key mismatch.")
        return False

    label_mismatch = merged[merged["y_true_new"] != merged["y_true_old"]]
    if len(label_mismatch) > 0:
        print(f"  FAIL: {len(label_mismatch)} label mismatches -- different underlying data.")
        return False

    diff = (merged["y_proba_new"] - merged["y_proba_old"]).abs()
    max_diff = float(diff.max())
    n_exact = int((diff == 0.0).sum())
    print(f"  exact matches: {n_exact}/{len(merged)}")
    print(f"  max |difference|: {max_diff:.3e}")

    if max_diff == 0.0:
        print("  PASS: bit-exact reproduction. The rerun is the same model.\n")
        return True
    if max_diff < 1e-9:
        print(f"  PASS (within float tolerance, {max_diff:.3e} < 1e-9).\n")
        return True
    print("  FAIL: predictions differ beyond float tolerance.")
    print("        Do NOT report pooled gate values against the published single-fold")
    print("        numbers -- they came from a measurably different model.\n")
    return False


def check_pooled_coverage(pooled: pd.DataFrame) -> bool:
    print("=" * 78)
    print("CHECK 2 -- POOLED COVERAGE (expect n=345, 23 positives per checkpoint)")
    print("=" * 78)
    ok = True
    for ck in CHECKPOINTS:
        g = pooled[pooled["checkpoint_minutes"] == ck]
        n = len(g)
        n_pos = int(g["y_true"].sum())
        n_unique = g["record_id"].nunique()
        flag = ""
        if n != EXPECTED_N or n_pos != EXPECTED_POSITIVES or n_unique != n:
            flag = "  <-- MISMATCH"
            ok = False
        print(f"  {ck:5.0f} min: n={n:4d}  unique={n_unique:4d}  positives={n_pos:3d}{flag}")
    print("  PASS\n" if ok else "  FAIL -- pooled cohort is not the full 345/23.\n")
    return ok


def check_fold_structure(pooled: pd.DataFrame) -> bool:
    print("=" * 78)
    print("CHECK 3 -- FOLD STRUCTURE (each record pooled from exactly 3 repeats)")
    print("=" * 78)
    counts = pooled["n_folds_pooled"].value_counts().sort_index()
    print("  n_folds_pooled distribution:")
    for value, count in counts.items():
        print(f"    {int(value)} fold(s): {count} record-checkpoint rows")
    ok = bool((pooled["n_folds_pooled"] == EXPECTED_REPEATS).all())
    print("  PASS\n" if ok else f"  FAIL -- expected every record pooled from exactly {EXPECTED_REPEATS} repeats.\n")
    return ok


def compare_tables(pooled_summary: pd.DataFrame, original_summary: pd.DataFrame) -> None:
    print("=" * 78)
    print("CHECK 4 -- POOLED vs ORIGINAL SINGLE-FOLD")
    print("=" * 78)

    orig = original_summary.set_index("checkpoint_minutes")
    pool = pooled_summary.set_index("checkpoint_minutes")

    print("\nORIGINAL (single fold, n=69, 5 positives)")
    print(f"{'ckpt':>6} {'CTG-right':>11} {'n':>5} {'Mat-right':>11} {'n':>5} {'gap':>8}")
    for ck in CHECKPOINTS:
        if ck not in orig.index:
            continue
        r = orig.loc[ck]
        c = float(r["gate_mean_when_ctg_right"])
        m = float(r["gate_mean_when_maternal_right"])
        print(f"{ck:6.0f} {c:11.4f} {int(r['n_disagreement_where_ctg_was_closer_to_truth']):5d} "
              f"{m:11.4f} {int(r['n_disagreement_where_maternal_was_closer_to_truth']):5d} {c - m:+8.4f}")

    print("\nPOOLED (all 15 folds, n=345, 23 positives)")
    print(f"{'ckpt':>6} {'CTG-right':>11} {'n':>5} {'Mat-right':>11} {'n':>5} {'gap':>8}")
    for ck in CHECKPOINTS:
        if ck not in pool.index:
            continue
        r = pool.loc[ck]
        c = float(r["gate_mean_when_ctg_right"])
        m = float(r["gate_mean_when_maternal_right"])
        print(f"{ck:6.0f} {c:11.4f} {int(r['n_disagreement_where_ctg_was_closer_to_truth']):5d} "
              f"{m:11.4f} {int(r['n_disagreement_where_maternal_was_closer_to_truth']):5d} {c - m:+8.4f}")

    print("\nGAP (CTG-right minus maternal-right): original -> pooled")
    print(f"{'ckpt':>6} {'original':>10} {'pooled':>10} {'change':>10}  direction")
    gaps = []
    for ck in CHECKPOINTS:
        if ck not in orig.index or ck not in pool.index:
            continue
        og = float(orig.loc[ck, "gate_mean_when_ctg_right"]) - float(orig.loc[ck, "gate_mean_when_maternal_right"])
        pg = float(pool.loc[ck, "gate_mean_when_ctg_right"]) - float(pool.loc[ck, "gate_mean_when_maternal_right"])
        gaps.append((ck, og, pg))
        direction = "positive" if pg > 0 else ("zero" if pg == 0 else "NEGATIVE (reversed)")
        print(f"{ck:6.0f} {og:10.4f} {pg:10.4f} {pg - og:+10.4f}  {direction}")

    if not gaps:
        print("\n  No overlapping checkpoints to compare.")
        return

    pooled_gaps = [pg for _, _, pg in gaps]
    orig_gaps = [og for _, og, _ in gaps]
    n_positive = sum(1 for g in pooled_gaps if g > 0)

    print("\n--- Reading of the result ---")
    print(f"  Pooled gap positive at {n_positive}/{len(pooled_gaps)} checkpoints.")
    print(f"  Mean gap: original {np.mean(orig_gaps):+.4f} -> pooled {np.mean(pooled_gaps):+.4f}")

    # Does the gap still widen with elapsed time?
    cks = [ck for ck, _, _ in gaps]
    trend = float(np.corrcoef(cks, pooled_gaps)[0, 1]) if len(cks) > 2 else float("nan")
    print(f"  Correlation of gap with elapsed time (pooled): r = {trend:+.3f}")
    print("    (the original finding was a gap WIDENING with elapsed time, i.e. r > 0)")

    if n_positive == len(pooled_gaps) and np.mean(pooled_gaps) >= np.mean(orig_gaps):
        verdict = "HELD or STRENGTHENED -- positive at every checkpoint, mean gap not reduced."
    elif n_positive == len(pooled_gaps):
        verdict = "HELD but WEAKENED -- still positive everywhere, mean gap smaller than single-fold."
    elif n_positive >= len(pooled_gaps) * 0.5:
        verdict = "AMBIGUOUS -- positive at some checkpoints but not all. Do not report as consistent."
    else:
        verdict = "WEAKENED or REVERSED -- gap is not reliably positive at full power."
    print(f"\n  VERDICT: {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the all-folds gate rerun.")
    parser.add_argument("--allfolds-dir", default=str(DEFAULT_ALLFOLDS))
    parser.add_argument("--original-dir", default=str(DEFAULT_ORIGINAL))
    args = parser.parse_args()

    allfolds_dir = Path(args.allfolds_dir)
    original_dir = Path(args.original_dir)

    determinism_ok = check_determinism(allfolds_dir, original_dir)
    if not determinism_ok:
        print("STOPPING: determinism check failed. Nothing below would be trustworthy.")
        sys.exit(1)

    pooled_path = allfolds_dir / "anytime_disagreement_analysis_pooled.csv"
    pooled_summary_path = allfolds_dir / "anytime_disagreement_summary_pooled.csv"
    original_summary_path = original_dir / "anytime_disagreement_summary.csv"
    original_analysis_path = original_dir / "anytime_disagreement_analysis.csv"

    if not pooled_path.exists():
        print(f"STOPPING: {pooled_path} not found -- run run_anytime_disagreement_pooled.py first.")
        sys.exit(1)

    pooled = pd.read_csv(pooled_path)
    pooled_summary = pd.read_csv(pooled_summary_path)

    coverage_ok = check_pooled_coverage(pooled)
    structure_ok = check_fold_structure(pooled)

    # The original summary predates the two gate-by-correctness columns, so
    # derive them from its per-record CSV rather than assuming they exist.
    original_summary = pd.read_csv(original_summary_path)
    if "gate_mean_when_ctg_right" not in original_summary.columns:
        oa = pd.read_csv(original_analysis_path)
        od = oa[oa["disagreement"] == True]  # noqa: E712
        derived = (
            od.groupby(["checkpoint_minutes", "ctg_reading_closer_to_truth"])["gate_value"]
            .mean().unstack()
        )
        original_summary = original_summary.set_index("checkpoint_minutes")
        original_summary["gate_mean_when_ctg_right"] = derived[True]
        original_summary["gate_mean_when_maternal_right"] = derived[False]
        original_summary = original_summary.reset_index()

    compare_tables(pooled_summary, original_summary)

    print("\n" + "=" * 78)
    all_ok = determinism_ok and coverage_ok and structure_ok
    print(f"VALIDATION: {'ALL CHECKS PASSED' if all_ok else 'ONE OR MORE CHECKS FAILED'}")
    print("=" * 78)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
