"""Statistics for the pre-registered shortcut experiments (analysis plan, sections 5-7).

    python -m src.shortcut.analyze [--dir results/shortcut]

Reads predictions.csv, cohort.csv, occlusion.csv and reference/ written by run_shortcut.py.
Writes <dir>/analysis/: auc.csv, primary_tests.csv, aauc.csv, secondary.csv, occlusion_profile.csv.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.analysis.significance import _structural_components, delong_test
from src.revision.stats_utils import BOOT_ITERS, SEED, auc_pr_ci, holm, paired_delta_ci, strat_boot_index

E4_PLANNED = {"F2", "F1", "F3"}   # plan section 5 (plus the two references); F1x, F1c60, F3s, length: section 7 extras


def averaged(preds: pd.DataFrame) -> pd.DataFrame:
    """Outer-test predictions averaged over seeds and repeats, one row per record and format."""
    t = preds[preds.role == "outer_test"]
    n_expected = t.groupby(["experiment", "eval_format"]).apply(lambda g: g.seed.nunique() * g.repeat.nunique())
    a = (t.groupby(["experiment", "eval_format", "record_id"], as_index=False)
         .agg(y_true=("y_true", "first"), y_proba=("y_proba", "mean"), n_pred=("y_proba", "size")))
    for (e, f), n in n_expected.items():
        got = a[(a.experiment == e) & (a.eval_format == f)].n_pred
        if not (got == n).all():
            raise SystemExit(f"incomplete predictions for {e}/{f}: expected {n} per record")
    return a


def single_auc_test(y, p):
    """Two-sided z-test of AUC = 0.5 with the DeLong variance."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    V10, V01 = _structural_components(p[y == 1][None, :], p[y == 0][None, :])
    auc = float(V10.mean())
    var = V10[0].var(ddof=1) / V10.shape[1] + V01[0].var(ddof=1) / V01.shape[1]
    z = (auc - 0.5) / math.sqrt(var) if var > 0 else float("inf")
    return {"auc": auc, "z": z, "p_value": 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))}


def aauc(y, p, strata):
    """Janes-Pepe covariate-adjusted AUC: case-control pairs compared within strata only."""
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        n1, n0 = int((y[m] == 1).sum()), int((y[m] == 0).sum())
        if n1 and n0:
            num += n1 * n0 * roc_auc_score(y[m], p[m])
            den += n1 * n0
    return num / den if den else float("nan")


def aauc_with_ci(y, p, strata, iters=BOOT_ITERS, seed=SEED):
    idx = strat_boot_index(y, iters, seed)
    a = aauc(y, p, strata)
    auc = roc_auc_score(y, p)
    boots = np.array([aauc(y[i], p[i], strata[i]) for i in idx])
    diffs = np.array([roc_auc_score(y[i], p[i]) for i in idx]) - boots
    return {"auc": auc, "aauc": a, "aauc_ci_low": np.nanpercentile(boots, 2.5), "aauc_ci_high": np.nanpercentile(boots, 97.5),
            "auc_minus_aauc": auc - a, "diff_ci_low": np.nanpercentile(diffs, 2.5), "diff_ci_high": np.nanpercentile(diffs, 97.5)}


def reference_scores(d: Path, ids: set) -> dict:
    """Existing leak-free predictions at 60 min (F0 deep CTG-only; LR CTG-only), averaged like the main ones."""
    out = {}
    f = d / "reference" / "deep_oof_t60.csv"
    if f.exists():
        r = pd.read_csv(f, dtype={"record_id": str})
        r = r[(r.model == "ctg_only") & r.record_id.isin(ids)]
        out["F0 deep CTG-only (leak-free, 60 min)"] = r.groupby("record_id").y_proba.mean()
    f = d / "reference" / "classical_oof_unfiltered_t60.csv"
    if f.exists():
        r = pd.read_csv(f, dtype={"record_id": str})
        r = r[(r.model == "logistic_regression") & (r.feature_set == "ctg_only") & r.record_id.isin(ids)]
        out["LR CTG-only (leak-free, 60 min)"] = r.groupby("record_id").y_proba.mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/shortcut")
    args = ap.parse_args()
    d = Path(args.dir)
    out = d / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    coh = pd.read_csv(d / "cohort.csv", dtype={"record_id": str}).set_index("record_id")
    avg = averaged(pd.read_csv(d / "predictions.csv", dtype={"record_id": str}))
    wide = avg.pivot_table(index="record_id", columns="eval_format", values="y_proba")
    ref = reference_scores(d, set(coh.index))
    for k, s in ref.items():
        wide[k] = s
    wide["Recording length alone"] = coh["length_min"]
    wide = wide.join(coh[["y", "length_min", "in_primary60"]], how="inner")

    cohorts = {"all_487": wide, "primary60": wide[wide.in_primary60 == 1]}
    tert = np.quantile(coh["length_min"], [1 / 3, 2 / 3])
    quint = np.quantile(coh["length_min"], [0.2, 0.4, 0.6, 0.8])
    auc_rows, test_rows, aauc_rows, sec_rows = [], [], [], []
    score_cols = [c for c in wide.columns if c not in ("y", "length_min", "in_primary60")]
    for cname, w in cohorts.items():
        y = w["y"].to_numpy(int)
        for c in score_cols:
            if w[c].notna().all():
                auc_rows.append({"cohort": cname, "score": c, "n": len(y), "cases": int(y.sum()), **auc_pr_ci(y, w[c].to_numpy())})
        # primary tests
        tests = []
        if "F2" in w:
            r = single_auc_test(y, w["F2"].to_numpy())
            ci = auc_pr_ci(y, w["F2"].to_numpy())
            tests.append({"experiment": "E1", "comparison": "AUC(F2 envelope only) - 0.5", "estimate": r["auc"] - 0.5,
                          "ci_low": ci["roc_auc_ci_low"] - 0.5, "ci_high": ci["roc_auc_ci_high"] - 0.5, "p_value": r["p_value"]})
        for e, a, b in (("E2", "F1", "F1x"), ("E3", "F3", "F3s")):
            if a in w and b in w:
                dl = delong_test(y, w[a].to_numpy(), w[b].to_numpy())
                ci = paired_delta_ci(y, w[a].to_numpy(), w[b].to_numpy())
                tests.append({"experiment": e, "comparison": f"AUC({a}) - AUC({b})", "estimate": dl["auc_diff"],
                              "ci_low": ci["delta_ci_low"], "ci_high": ci["delta_ci_high"], "p_value": dl["p_value"]})
        ph = holm([t["p_value"] for t in tests])
        for t, h in zip(tests, ph):
            test_rows.append({"cohort": cname, **t, "holm_p": h, "supported": bool(h < 0.05 and t["estimate"] > 0)})
        # E4
        for label, cuts in (("tertiles (primary)", tert), ("quintiles (robustness)", quint)):
            strata = np.digitize(w["length_min"].to_numpy(), cuts)
            for c in score_cols:
                if w[c].notna().all():
                    role = "pre-registered" if c in E4_PLANNED or c in ref else "descriptive extra"
                    aauc_rows.append({"cohort": cname, "strata": label, "score": c, "role": role,
                                      **aauc_with_ci(y, w[c].to_numpy(), strata)})
        # secondary
        if "F1" in w and "F1x" in w:
            sec_rows.append({"cohort": cname, "item": "mean |p(F1) - p(F1x)|", "value": float(np.abs(w["F1"] - w["F1x"]).mean())})
        if "F1c60" in w:
            for k in ref:
                ci = paired_delta_ci(y, w["F1c60"].to_numpy(), w[k].to_numpy())
                sec_rows.append({"cohort": cname, "item": f"AUC(F1c60) - AUC({k}) [descriptive]", "value": ci["delta_auc"],
                                 "ci_low": ci["delta_ci_low"], "ci_high": ci["delta_ci_high"]})
    pd.DataFrame(auc_rows).to_csv(out / "auc.csv", index=False)
    pd.DataFrame(test_rows).to_csv(out / "primary_tests.csv", index=False)
    pd.DataFrame(aauc_rows).to_csv(out / "aauc.csv", index=False)
    pd.DataFrame(sec_rows).to_csv(out / "secondary.csv", index=False)

    occ_f = d / "occlusion.csv"
    if occ_f.exists():
        o = pd.read_csv(occ_f, dtype={"record_id": str})
        o = o.groupby(["record_id", "block", "minute_start", "length_min"], as_index=False).abs_dlogit.mean()
        o["min_from_end"] = o["minute_start"] + 1.0 - o["length_min"]   # block centre relative to the recording's end
        o["bin"] = (np.floor(o["min_from_end"] / 2) * 2 + 1).clip(-89, 89)
        o = o.merge(coh[["y"]], left_on="record_id", right_index=True)
        prof = (o.groupby(["bin"]).abs_dlogit.agg(["mean", "sem", "size"]).reset_index()
                .rename(columns={"bin": "minutes_from_recording_end"}))
        prof.to_csv(out / "occlusion_profile.csv", index=False)
    print(pd.DataFrame(test_rows).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
