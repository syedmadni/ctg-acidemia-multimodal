"""Statistics for pre-registered experiment E5 (results/shortcut_e5/PREREGISTRATION_E5.md, sections 5-7).

    python -m src.shortcut.analyze_e5 [--dir results/shortcut_e5] [--smoke]

Reads <dir>/predictions.csv, <dir>/cohort.csv and <dir>/training_log.csv (written by run_e5.py), plus,
read-only, the E2 predictions and training log in results/shortcut/. Reuses the registered E1-E4 code
(src/shortcut/analyze.py) and the revision statistics (src/revision/stats_utils.py) by importing them.
Writes <dir>/analysis/: auc.csv, primary_test.csv, aauc.csv, secondary.csv, training_summary.csv.

Validation checks (the script stops with an error if any fails): recomputed E2 AUC(F1) = 0.6955 and
AUC(F1x) = 0.7313 (+-0.0001, all 487 recordings); every record has 15 outer-test predictions per format;
no non-finite values; the tertile cut-points equal those of E4. --smoke (pipeline check on
results/shortcut_e5_smoke only) relaxes the completeness checks to the records and folds that were run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.analysis.significance import delong_test
from src.revision.stats_utils import BOOT_ITERS, SEED, auc_pr_ci, paired_delta_ci
from src.shortcut.analyze import aauc_with_ci, averaged

E2_DIR = Path("results/shortcut")
E2_EXPECTED = {"F1": 0.6955, "F1x": 0.7313}
N_PRED = 15                     # 5 seeds x 3 repeats
COMPARISON = "AUC(F5) - AUC(F5x)"
CONCLUSIONS = {
    "relies": "relies on the end",
    "fell": "AUC fell when the end was hidden; cannot be separated from an unfamiliar input",
    "none": "no evidence of reliance on the end",
}


def fail(msg):
    raise SystemExit(f"VALIDATION FAILED: {msg}")


def cut_points(lengths):
    """As in analyze.py: tertiles and quintiles of all 487 recording lengths, without the outcome."""
    return np.quantile(lengths, [1 / 3, 2 / 3]), np.quantile(lengths, [0.2, 0.4, 0.6, 0.8])


def conclusion(supported: bool, opposite: bool, aauc_above_0: bool) -> str:
    if supported and aauc_above_0:
        return CONCLUSIONS["relies"]
    if supported:
        return CONCLUSIONS["fell"]
    return CONCLUSIONS["none"] + (" (opposite direction)" if opposite else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/shortcut_e5")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    d = Path(args.dir)
    out = d / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    coh = pd.read_csv(d / "cohort.csv", dtype={"record_id": str}, float_precision="round_trip").set_index("record_id")
    coh_e4 = pd.read_csv(E2_DIR / "cohort.csv", dtype={"record_id": str}, float_precision="round_trip")

    # E5 predictions
    raw5 = pd.read_csv(d / "predictions.csv", dtype={"record_id": str}, float_precision="round_trip")
    if not np.isfinite(raw5.y_proba).all():
        fail("non-finite E5 predicted probabilities")
    avg5 = averaged(raw5)                          # also checks seeds x repeats predictions per record
    if not args.smoke:
        for f, g in avg5.groupby("eval_format"):
            if not (g.n_pred == N_PRED).all():
                fail(f"E5/{f}: not every record has {N_PRED} outer-test predictions")
            if set(g.record_id) != set(coh.index):
                fail(f"E5/{f}: outer-test records do not cover the cohort")
    w5 = avg5.pivot_table(index="record_id", columns="eval_format", values="y_proba")[["F5", "F5x"]]

    # E2 predictions (read-only), averaged the same way
    raw2 = pd.read_csv(E2_DIR / "predictions.csv", dtype={"record_id": str}, float_precision="round_trip")
    avg2 = averaged(raw2)
    avg2 = avg2[(avg2.experiment == "E2") & avg2.eval_format.isin(["F1", "F1x"])]
    if not (avg2.n_pred == N_PRED).all():
        fail(f"E2: not every record has {N_PRED} outer-test predictions")
    w2 = avg2.pivot_table(index="record_id", columns="eval_format", values="y_proba")[["F1", "F1x"]]

    wide = w5.join(w2, how="inner").join(coh[["y", "length_min", "in_primary60"]], how="inner")
    if not args.smoke and len(wide) != len(coh):
        fail(f"{len(wide)} records after joining, expected {len(coh)}")
    if not np.isfinite(wide[["F5", "F5x", "F1", "F1x"]].to_numpy()).all():
        fail("non-finite averaged predictions")

    # check: recomputed E2 AUCs in all 487 recordings
    w2c = w2.join(coh[["y"]], how="inner")
    for f, expected in E2_EXPECTED.items():
        got = roc_auc_score(w2c["y"], w2c[f])
        if abs(got - expected) > 1e-4:
            fail(f"E2 AUC({f}) = {got:.6f}, expected {expected} +- 0.0001")
        print(f"check passed: E2 AUC({f}) = {got:.6f} (expected {expected})")

    # cut-points: from all 487 lengths, exactly as in analyze.py; must equal E4's
    tert, quint = cut_points(coh["length_min"].to_numpy(float))
    tert_e4, quint_e4 = cut_points(coh_e4["length_min"].to_numpy(float))
    if not (np.array_equal(tert, tert_e4) and np.array_equal(quint, quint_e4)):
        fail(f"cut-points {tert} differ from E4 {tert_e4}")
    print(f"check passed: tertile cut-points {tert.tolist()} equal E4; quintiles {quint.tolist()}")

    cohorts = {"all_487": wide, "primary60": wide[wide.in_primary60 == 1]}
    auc_rows, test_rows, aauc_rows, sec_rows = [], [], [], []
    for cname, w in cohorts.items():
        y = w["y"].to_numpy(int)
        p5, p5x = w["F5"].to_numpy(), w["F5x"].to_numpy()
        for c in ("F5", "F5x"):
            auc_rows.append({"cohort": cname, "score": c, "n": len(y), "cases": int(y.sum()),
                             **auc_pr_ci(y, w[c].to_numpy(), BOOT_ITERS, SEED)})
        # section 6, item 1 (and item 5 in primary60): AUC - AAUC, tertiles (primary) and quintiles (robustness)
        tert_f5 = None
        for label, cuts in (("tertiles (primary)", tert), ("quintiles (robustness)", quint)):
            strata = np.digitize(w["length_min"].to_numpy(), cuts)
            for c in ("F5", "F5x"):
                r = aauc_with_ci(y, w[c].to_numpy(), strata, BOOT_ITERS, SEED)
                aauc_rows.append({"cohort": cname, "strata": label, "score": c, **r})
                if c == "F5" and label.startswith("tertiles"):
                    tert_f5 = r
        # section 5: primary test
        dl = delong_test(y, p5, p5x)
        ci = paired_delta_ci(y, p5, p5x, BOOT_ITERS, SEED)
        est, p = dl["auc_diff"], dl["p_value"]
        supported = bool(p < 0.05 and est > 0)
        opposite = bool(p < 0.05 and est < 0)
        above0 = bool(tert_f5["diff_ci_low"] > 0)
        test_rows.append({"cohort": cname, "n": len(y), "cases": int(y.sum()), "comparison": COMPARISON,
                          "estimate": est, "ci_low": ci["delta_ci_low"], "ci_high": ci["delta_ci_high"], "p_value": p,
                          "primary_supported": supported, "opposite_direction": opposite,
                          "aauc_tertile_ci_above_0": above0,
                          "f5_auc_minus_aauc_tertiles": tert_f5["auc_minus_aauc"],
                          "f5_auc_minus_aauc_ci_low": tert_f5["diff_ci_low"],
                          "f5_auc_minus_aauc_ci_high": tert_f5["diff_ci_high"],
                          "conclusion": conclusion(supported, opposite, above0)})
        # section 6, item 2 (both cohorts): size of the shift
        sec_rows.append({"cohort": cname, "item": "mean |p(F5) - p(F5x)|", "value": float(np.abs(p5 - p5x).mean())})
        # section 6, item 4 (all 487 only; item 5 repeats only the primary test and items 1-2)
        if cname == "all_487":
            for a, b in (("F5", "F1"), ("F5x", "F1x")):
                r = paired_delta_ci(y, w[a].to_numpy(), w[b].to_numpy(), BOOT_ITERS, SEED)
                sec_rows.append({"cohort": cname, "item": f"AUC({a}) - AUC({b})", "value": r["delta_auc"],
                                 "ci_low": r["delta_ci_low"], "ci_high": r["delta_ci_high"]})

    # section 6, item 3: training, E5 next to E2
    log5 = pd.read_csv(d / "training_log.csv")
    log2 = pd.read_csv(E2_DIR / "training_log.csv")
    log2 = log2[log2.experiment == "E2"]
    tr_rows = []
    for exp, crit, g in (("E5", "lowest validation loss", log5), ("E2", "highest validation PR-AUC", log2)):
        tr_rows.append({"experiment": exp, "early_stopping": crit, "n_folds": len(g),
                        "best_epoch_median": float(g.best_epoch.median()), "best_epoch_min": int(g.best_epoch.min()),
                        "best_epoch_max": int(g.best_epoch.max()), "n_folds_best_epoch_1": int((g.best_epoch == 1).sum())})
    if not args.smoke and len(log5) != 75:
        fail(f"E5 training log has {len(log5)} folds, expected 75")
    if not np.isfinite(log5.best_val_loss.astype(float)).all():
        fail("non-finite best validation loss in the E5 training log")

    pd.DataFrame(auc_rows).to_csv(out / "auc.csv", index=False)
    pd.DataFrame(test_rows).to_csv(out / "primary_test.csv", index=False)
    pd.DataFrame(aauc_rows).to_csv(out / "aauc.csv", index=False)
    pd.DataFrame(sec_rows).to_csv(out / "secondary.csv", index=False)
    pd.DataFrame(tr_rows).to_csv(out / "training_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(pd.DataFrame(test_rows).to_string())
    print(f"all validation checks passed; wrote {out}")


if __name__ == "__main__":
    main()
