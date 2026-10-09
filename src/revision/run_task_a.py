"""Task A: leak-free, checkpoint-based quality filter (classical models and Firth tests).

PRIMARY analysis (DEVIATIONS A1): variant `ckfilter`, maternal-block Firth LRT (8 df),
Holm across 5 checkpoints. Sensitivity: `unfiltered`, `primary345_causal`.
POST HOC: parity (1 df). Pre-declared sensitivity: extended intrapartum block (A3b).

    python -m src.revision.run_task_a

Writes results/revision/A_*.csv; per-repeat OOF and validation predictions go to the
git-ignored results/revision/oof/ and results/revision/valpred/.
"""
from __future__ import annotations

import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from src.analysis.firth import firth_lrt
from src.analysis.significance import delong_test
from src.data.fixed_ctg_features import ctg_summary_features_v2
from src.revision.causal_features import FEATURE_NAMES, causal_feature_matrix
from src.revision.cohorts import (
    CHECKPOINTS, VARIANTS, cohort_flow, labelled_meta, load_splits, make_and_save_splits,
    maternal_matrix, members, quality_table,
)
from src.revision.predictors import INTRAPARTUM_EXTRA, MATERNAL_8, assert_allowed
from src.revision.raw_cache import RawData
from src.revision.stats_utils import auc_pr_ci, holm, paired_delta_ci
from src.training.train_cv import aggregate_repeated_oof_anytime, train_sklearn_quality_fold

OUT = Path("results/revision")
SEED = 42
MODELS = ["logistic_regression", "random_forest"]
FEATURE_SETS = ["ctg_only", "maternal_only", "combined"]
EXTENDED_COLS = MATERNAL_8 + ["pyrexia", "meconium", "presentation__2", "presentation__3", "liq_praecox"]
PREDICTOR_BLOCKS = {"maternal_8": MATERNAL_8, "parity": ["parity"], "extended_intrapartum": EXTENDED_COLS}
TEST_LABELS = {
    "maternal_8": "PRIMARY (on ckfilter) / sensitivity (other variants): maternal block, 8 df",
    "parity": "POST HOC: raw parity, 1 df",
    "extended_intrapartum": "pre-declared sensitivity analysis: maternal 8 + pyrexia, meconium, presentation (dummies), liq. praecox",
}
PUBLISHED_STEP1 = Path("outputs/quality_cohorts/primary_705_anytime_vr060/maternal_value/step1_lrt_maternal_block.csv")
PUBLISHED_FEATFIX = Path("outputs/quality_cohorts/primary_705_anytime_vr060_featfix/featfix_results.csv")


# ------------------------------------------------------------------ design matrices

def extended_block(cohort: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Maternal 8 (median-imputed) + intrapartum extras; presentation NaN -> mode (code 1)."""
    mat = SimpleImputer(strategy="median").fit_transform(maternal_matrix(cohort))
    pres = cohort["presentation"].fillna(1.0).to_numpy()
    extra = np.column_stack([
        cohort["pyrexia"].to_numpy(float), cohort["meconium"].to_numpy(float),
        (pres == 2).astype(float), (pres == 3).astype(float), cohort["liq_praecox"].to_numpy(float),
    ])
    assert_allowed(EXTENDED_COLS)
    return np.column_stack([mat, extra]), list(EXTENDED_COLS)


def ctg_block_whole_cohort(ctg_raw: np.ndarray) -> tuple[np.ndarray, list[str], list[str]]:
    """Whole-cohort median imputation, constant columns dropped (steps123 convention)."""
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(ctg_raw)
    keep = [j for j in range(imp.shape[1]) if len(np.unique(imp[:, j])) > 1]
    return imp[:, keep], [FEATURE_NAMES[j] for j in keep], [FEATURE_NAMES[j] for j in range(imp.shape[1]) if j not in keep]


def firth_block_test(ctg: np.ndarray, block: np.ndarray, block_names: list[str], y: np.ndarray) -> dict:
    keep = [j for j in range(block.shape[1]) if len(np.unique(block[:, j])) > 1]
    dropped = [block_names[j] for j in range(block.shape[1]) if j not in keep]
    block = block[:, keep]
    X = np.column_stack([np.ones(len(y)), ctg, block])
    test_idx = list(range(1 + ctg.shape[1], X.shape[1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = firth_lrt(X, y, test_idx=test_idx)
    return {
        "chi2": r.lr_stat, "df": r.df, "raw_p": r.p_value,
        "block_cols_dropped_constant": ";".join(dropped) or "(none)",
        "converged_full": r.full_fit.converged, "converged_restricted": r.restricted_fit.converged,
        "fit_warnings": "; ".join(r.full_fit.warnings + r.restricted_fit.warnings) or "(none)",
        "coef_first_tested": float(r.full_fit.beta[test_idx[0]]) if len(test_idx) == 1 else np.nan,
    }


# ------------------------------------------------------------------ self-verification

def self_verify(raw, meta, q) -> pd.DataFrame:
    """Old features on the published arrays + published folds must reproduce published values."""
    from src.training.train_cv import load_cohort_arrays
    X_ctg, X_mat, y, _, _ = load_cohort_arrays(Path("data/processed"), "_705_anytime_vr060")
    coh = members(meta, q, "primary345_causal", 60.0)
    assert (coh["y"].to_numpy() == y).all()
    rows = []
    step1 = pd.read_csv(PUBLISHED_STEP1)
    featfix = pd.read_csv(PUBLISHED_FEATFIX)
    for t in CHECKPOINTS:
        make_and_save_splits(coh, "primary345_causal", t)
        splits = load_splits(coh, "primary345_causal", t)
        v2 = ctg_summary_features_v2(X_ctg, checkpoint_minutes=t)
        ctg, _, _ = ctg_block_whole_cohort(v2)
        mat = SimpleImputer(strategy="median").fit_transform(X_mat)
        mine = firth_block_test(ctg, mat, MATERNAL_8, y)["chi2"]
        pub = float(step1.loc[step1.checkpoint_minutes == t, "primary_lr_stat"].iloc[0])
        rows.append({"check": "Firth maternal chi2, v2 features, published 345", "checkpoint_minutes": t,
                     "reproduced": round(mine, 4), "published": pub, "match_4dp": round(mine, 4) == round(pub, 4)})
        scratch = Path(tempfile.mkdtemp(prefix="revA_verify_"))
        try:
            feats = np.concatenate([v2, X_mat], axis=1)
            p = np.zeros(len(y))
            for sp in splits:
                _, _, tp, _ = train_sklearn_quality_fold("logistic_regression", feats, y, sp, SEED, scratch)
                p[np.asarray(sp["outer_test_idx"])] += tp / 3.0
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        mine_auc = roc_auc_score(y, p)
        pub_auc = float(featfix.loc[(featfix.checkpoint_minutes == t) & (featfix.feature_set == "both")
                                    & (featfix.model == "logistic_regression"), "roc_auc_risk_set"].iloc[0])
        rows.append({"check": "LR CTG+maternal AUC, v2 features, published 345, splits from disk",
                     "checkpoint_minutes": t, "reproduced": round(mine_auc, 4), "published": pub_auc,
                     "match_4dp": round(mine_auc, 4) == round(pub_auc, 4)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "A_self_verification.csv", index=False)
    if not df["match_4dp"].all():
        print(df.to_string(index=False))
        raise SystemExit("ABORT: self-verification against published values failed.")
    return df


# ------------------------------------------------------------------ main

def feature_sets(ctg_raw: np.ndarray, mat: np.ndarray) -> dict:
    return {"ctg_only": ctg_raw, "maternal_only": mat, "combined": np.concatenate([ctg_raw, mat], axis=1)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "oof").mkdir(exist_ok=True)
    (OUT / "valpred").mkdir(exist_ok=True)
    raw = RawData()
    meta = labelled_meta(raw)
    q = quality_table(raw)
    q.to_csv(OUT / "cache" / "quality_by_checkpoint.csv", index=False)

    ver = self_verify(raw, meta, q)
    print(f"self-verification passed ({len(ver)} checks)", flush=True)

    flow = cohort_flow(meta, q)
    flow.to_csv(OUT / "A_cohort_flow.csv", index=False)

    oof_rows, val_rows, summary_rows, coding_rows, firth_rows = [], [], [], [], []
    scratch = Path(tempfile.mkdtemp(prefix="revA_"))
    try:
        for variant in VARIANTS:
            for t in CHECKPOINTS:
                coh = members(meta, q, variant, t)
                make_and_save_splits(coh, variant, t)
                splits = load_splits(coh, variant, t)
                y = coh["y"].to_numpy()
                rid = coh["record_id"].to_numpy()
                ctg_raw = causal_feature_matrix(raw, rid, t)
                mat = maternal_matrix(coh)
                summary_rows.append({
                    "variant": variant, "checkpoint_minutes": t, "n": len(y), "cases": int(y.sum()),
                    "prevalence": round(float(y.mean()), 4),
                    "records_no_valid_fhr_in_window": int(np.isnan(ctg_raw[:, 0]).sum()),
                    "records_no_valid_uc_in_window": int(np.isnan(ctg_raw[:, 1]).sum()),
                    "cases_nulliparous": int(((coh["parity"] == 0) & (y == 1)).sum()),
                    "controls_nulliparous": int(((coh["parity"] == 0) & (y == 0)).sum()),
                    "median_quality_ratio": round(float(coh[f"q{int(t)}"].median()), 4),
                })
                # ---- CV, classical models
                for fs, X in feature_sets(ctg_raw, mat).items():
                    for model in MODELS:
                        for sp in splits:
                            _, _, tp, vp = train_sklearn_quality_fold(model, X, y, sp, SEED, scratch)
                            base = {"variant": variant, "checkpoint_minutes": t, "model": model,
                                    "feature_set": fs, "repeat": sp["repeat"], "fold": sp["fold"]}
                            oof_rows += [{**base, "record_id": rid[i], "y_true": int(y[i]), "y_proba": float(p)}
                                         for i, p in zip(np.asarray(sp["outer_test_idx"]), tp)]
                            val_rows += [{**base, "record_id": rid[i], "y_true": int(y[i]), "y_proba": float(p)}
                                         for i, p in zip(np.asarray(sp["val_idx"]), vp)]
                    print(f"  CV done {variant} t={t:g} {fs}", flush=True)
                # ---- Firth tests (whole-cohort fit)
                ctg, ctg_names, dropped = ctg_block_whole_cohort(ctg_raw)
                blocks = {
                    "maternal_8": (SimpleImputer(strategy="median").fit_transform(mat), MATERNAL_8),
                    "parity": (coh[["parity"]].to_numpy(float), ["parity"]),
                    "extended_intrapartum": extended_block(coh),
                }
                for test, (B, names) in blocks.items():
                    assert_allowed(names)
                    r = firth_block_test(ctg, B, names, y)
                    firth_rows.append({"variant": variant, "checkpoint_minutes": t, "test": test,
                                       "label": TEST_LABELS[test], "n": len(y), "cases": int(y.sum()),
                                       "ctg_cols_kept": len(ctg_names),
                                       "ctg_cols_dropped_constant": ";".join(dropped) or "(none)",
                                       "events_per_variable_full_model": round(int(y.sum()) / (len(ctg_names) + r["df"]), 3),
                                       **r})
                # ---- extended-block coding / missingness
                for col in INTRAPARTUM_EXTRA:
                    vc = coh.groupby(["y", coh[col].fillna(-999)]).size()
                    for (yy, lvl), n in vc.items():
                        coding_rows.append({"variant": variant, "checkpoint_minutes": t, "field": col,
                                            "level": "missing" if lvl == -999 else lvl,
                                            "group": "case" if yy == 1 else "control", "n": int(n)})
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    pd.DataFrame(summary_rows).to_csv(OUT / "A_cohort_summary.csv", index=False)
    pd.DataFrame(coding_rows).to_csv(OUT / "A_extended_block_coding.csv", index=False)
    oof = pd.DataFrame(oof_rows)
    oof.to_csv(OUT / "oof" / "classical_oof.csv", index=False)
    pd.DataFrame(val_rows).to_csv(OUT / "valpred" / "classical_valpred.csv", index=False)

    firth = pd.DataFrame(firth_rows)
    firth["holm_p"] = np.nan
    for _, g in firth.groupby(["variant", "test"]):
        firth.loc[g.index, "holm_p"] = holm(g["raw_p"].tolist())
    firth.to_csv(OUT / "A_firth_lrt.csv", index=False)

    # ---- discrimination and combined-vs-CTG
    oof["cohort_name"] = oof["variant"] + "|" + oof["feature_set"]
    avg = aggregate_repeated_oof_anytime(oof)
    avg[["variant", "feature_set"]] = avg["cohort_name"].str.split("|", expand=True)
    avg.to_csv(OUT / "oof" / "classical_oof_repeat_averaged.csv", index=False)
    disc, comp = [], []
    for (variant, t, model), g in avg.groupby(["variant", "checkpoint_minutes", "model"]):
        by_fs = {fs: gg.sort_values("record_id") for fs, gg in g.groupby("feature_set")}
        for fs, gg in by_fs.items():
            yv = gg["y_true"].to_numpy()
            disc.append({"variant": variant, "checkpoint_minutes": t, "model": model, "feature_set": fs,
                         "n": len(yv), "cases": int(yv.sum()), **auc_pr_ci(yv, gg["y_proba"].to_numpy())})
        a, b = by_fs["combined"], by_fs["ctg_only"]
        assert (a["record_id"].to_numpy() == b["record_id"].to_numpy()).all()
        yv = a["y_true"].to_numpy()
        d = delong_test(yv, a["y_proba"].to_numpy(), b["y_proba"].to_numpy())
        comp.append({"variant": variant, "checkpoint_minutes": t, "model": model, "n": len(yv), "cases": int(yv.sum()),
                     "auc_combined": d["auc_a"], "auc_ctg_only": d["auc_b"], "delong_p": d["p_value"],
                     **paired_delta_ci(yv, a["y_proba"].to_numpy(), b["y_proba"].to_numpy())})
    disc = pd.DataFrame(disc)
    comp = pd.DataFrame(comp)
    comp["delong_holm_p"] = np.nan
    comp["boot_holm_p"] = np.nan
    for _, g in comp.groupby(["variant", "model"]):
        comp.loc[g.index, "delong_holm_p"] = holm(g["delong_p"].tolist())
        comp.loc[g.index, "boot_holm_p"] = holm(g["boot_p_two_sided"].tolist())
    disc.round(4).to_csv(OUT / "A_discrimination.csv", index=False)
    comp.round(4).to_csv(OUT / "A_combined_vs_ctg.csv", index=False)
    print("Task A (classical + Firth) complete.", flush=True)


if __name__ == "__main__":
    main()
