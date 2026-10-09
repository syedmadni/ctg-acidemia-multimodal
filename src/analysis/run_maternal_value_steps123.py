"""Maternal-value investigation, Steps 1-3 (HANDOFF.md section 7a).

Spec: outputs/quality_cohorts/primary_705_anytime_vr060/maternal_value/{HANDOFF.md,
FEATURE_FIX_SPEC.md}, refined by the user's decisions of 2026-09-20.

Gate passed before this ran: src/analysis/firth_ref/validate_firth.py (PASS, 3 dp,
against logistf 1.26.1 on sex2; see validation_report.txt).

Step 1: LRT M0 (CTG, fixed features) vs M1 (CTG + 8 maternal), df=8, Firth, at
checkpoints 10-60. Sensitivity: standard MLE with pre-eclampsia removed, df=7.
Holm across the 5 checkpoints, each family separately. EPV reported both ways
(full-cohort and per-fold).

Step 2: parity, labelled POST HOC everywhere (selected after CLAUDE.md's 2026-08-12
"22/23 positives nulliparous" observation, not pre-specified). 2x2 (nulliparous vs
parous x label) with Fisher exact + OR; LRT M0 vs M2 (df=1, raw parity added to CTG);
CV discrimination (CTG+parity vs CTG-only) on the same folds with paired DeLong and
permutation; parity-age Pearson correlation.

Step 3: bootstrap CI (percentile + BCa, 2000 iterations, seed 42) for the AUC
difference between the "both" (CTG+maternal) and "ctg_only_features" logistic
regression models already produced by run_featfix_cohort.py -- retrained here only
to recover per-repeat OOF (the saved file already pools all 3 repeats), and
self-verified against the published featfix_results.csv before use.

Risk set: checkpoints 10-60 all use the full cohort (n=345/23) -- see
FEATURE_FIX_SPEC.md section 2 and Amendment 1. 90 min is excluded throughout.

Outputs: outputs/quality_cohorts/primary_705_anytime_vr060/maternal_value/step123_*.csv
"""
from __future__ import annotations

import sys
import tempfile
import warnings as _warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.impute import SimpleImputer
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.firth import fit_firth, firth_lrt  # noqa: E402
from src.analysis.significance import delong_test, paired_permutation_test  # noqa: E402
from src.data.fixed_ctg_features import FEATURE_NAMES, ctg_summary_features_v2  # noqa: E402
from src.training.train_cv import (  # noqa: E402
    aggregate_repeated_oof_anytime,
    load_cohort_arrays,
    precompute_quality_splits,
    train_sklearn_quality_fold,
)

SUFFIX = "_705_anytime_vr060"
OUT_DIR = Path("outputs/quality_cohorts/primary_705_anytime_vr060/maternal_value")
FEATFIX_RESULTS = Path("outputs/quality_cohorts/primary_705_anytime_vr060_featfix/featfix_results.csv")
MATERNAL_NAMES = ["age", "gravidity", "parity", "gest_weeks", "diabetes", "hypertension", "preeclampsia", "induced"]
CHECKPOINTS = [10.0, 20.0, 30.0, 45.0, 60.0]  # 90 excluded per Amendment 1
SEED = 42
BOOT_ITERS = 2000
PERM_ITERS = 2000
REPORT_DECIMALS_FILE = 4
REPORT_DECIMALS_SUMMARY = 3

warnings_log: list[dict] = []


def log_warning(step: str, detail: str) -> None:
    warnings_log.append({"step": step, "detail": detail})


# ---------------------------------------------------------------------------
# Data loading and feature construction
# ---------------------------------------------------------------------------

def load_data():
    X_ctg, X_maternal, y, record_ids, _ = load_cohort_arrays(Path("data/processed"), SUFFIX)
    return X_ctg, X_maternal, y, record_ids


def build_ctg_block(X_ctg: np.ndarray, ck: float) -> tuple[np.ndarray, list[str], list[str]]:
    """Fixed CTG features at checkpoint ck, median-imputed on the FULL cohort
    (this is a single global fit for the LRT/parity analyses in this script --
    a DIFFERENT convention from the per-fold CV imputation used elsewhere in
    the project, and deliberately so: Steps 1-2 fit one model on the whole
    sample, not a cross-validated ensemble). Constant columns (nunique==1
    after imputation) are dropped and listed, never silently kept.
    """
    raw = ctg_summary_features_v2(X_ctg, checkpoint_minutes=ck)
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(raw)
    keep_idx, keep_names, dropped_names = [], [], []
    for j, name in enumerate(FEATURE_NAMES):
        if len(np.unique(imputed[:, j])) <= 1:
            dropped_names.append(name)
        else:
            keep_idx.append(j)
            keep_names.append(name)
    return imputed[:, keep_idx], keep_names, dropped_names


def impute_maternal(X_maternal: np.ndarray) -> np.ndarray:
    return SimpleImputer(strategy="median").fit_transform(X_maternal)


def build_ctg_block_raw(X_ctg: np.ndarray, ck: float) -> np.ndarray:
    """Raw (un-imputed) fixed CTG features -- for CV-fold contexts (Step 2c, Step 3)
    ONLY. These must reproduce run_featfix_cohort.py's convention exactly: NaNs are
    left in place and imputed by each fold's own sklearn Pipeline
    (imputer -> scaler -> model), fit on that fold's inner_train_idx alone. Using the
    whole-cohort-imputed block from build_ctg_block() here would leak test-fold
    information into the imputer and silently change every downstream AUC -- this is
    exactly the bug caught by Step 3's self-verification against featfix_results.csv.
    """
    return ctg_summary_features_v2(X_ctg, checkpoint_minutes=ck)


def holm(pvals: list[float]) -> list[float]:
    return multipletests(pvals, method="holm")[1].tolist()


# ---------------------------------------------------------------------------
# Step 1: LRT (maternal block), Firth primary + standard-MLE sensitivity
# ---------------------------------------------------------------------------

def step1(X_ctg, X_maternal, y, splits) -> pd.DataFrame:
    Xmat_imp = impute_maternal(X_maternal)
    pe_idx = MATERNAL_NAMES.index("preeclampsia")
    rows = []

    for ck in CHECKPOINTS:
        ctg, ctg_names, dropped = build_ctg_block(X_ctg, ck)
        if dropped:
            log_warning("step1", f"checkpoint {ck}: constant CTG columns dropped: {dropped}")
        n_ctg = ctg.shape[0]

        # ---- Primary: Firth, full 8-variable maternal block ----
        X1 = np.column_stack([np.ones(n_ctg), ctg, Xmat_imp])
        maternal_idx = list(range(1 + len(ctg_names), 1 + len(ctg_names) + 8))
        lrt = firth_lrt(X1, y, test_idx=maternal_idx)
        if not lrt.full_fit.converged:
            log_warning("step1_firth_full", f"checkpoint {ck}: full model did not converge")
        if not lrt.restricted_fit.converged:
            log_warning("step1_firth_full", f"checkpoint {ck}: restricted model (M0) did not converge")
        if lrt.full_fit.warnings:
            log_warning("step1_firth_full", f"checkpoint {ck} full: {lrt.full_fit.warnings}")
        if lrt.restricted_fit.warnings:
            log_warning("step1_firth_full", f"checkpoint {ck} restricted: {lrt.restricted_fit.warnings}")

        # ---- Sensitivity: standard MLE, pre-eclampsia removed (df=7) ----
        keep_mat = [i for i in range(8) if i != pe_idx]
        X1_sens = np.column_stack([np.ones(n_ctg), ctg, Xmat_imp[:, keep_mat]])
        n_ctg_cols = len(ctg_names)
        full_cols = slice(0, 1 + n_ctg_cols + 7)
        restricted_cols = slice(0, 1 + n_ctg_cols)
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            full_mle = sm.Logit(y, X1_sens[:, full_cols]).fit(disp=0, maxiter=500)
            restricted_mle = sm.Logit(y, X1_sens[:, restricted_cols]).fit(disp=0, maxiter=500)
            for w in caught:
                log_warning("step1_standard_mle", f"checkpoint {ck}: {w.category.__name__}: {w.message}")
        if not full_mle.mle_retvals.get("converged", True):
            log_warning("step1_standard_mle", f"checkpoint {ck}: full model (sensitivity) did not converge")
        if not restricted_mle.mle_retvals.get("converged", True):
            log_warning("step1_standard_mle", f"checkpoint {ck}: restricted model (sensitivity) did not converge")
        sens_lr = 2.0 * (full_mle.llf - restricted_mle.llf)
        sens_df = 7
        sens_p = float(stats.chi2.sf(sens_lr, sens_df))

        n_pos = int(y.sum())
        m0_cols = len(ctg_names)
        m1_cols = m0_cols + 8

        train_pos, train_n = [], []
        for sp in splits:
            idx = np.asarray(sp["inner_train_idx"], dtype=int)
            train_pos.append(int(y[idx].sum()))
            train_n.append(len(idx))
        mean_train_pos = float(np.mean(train_pos))

        rows.append({
            "checkpoint_minutes": ck,
            "n_ctg_cols_kept": m0_cols, "ctg_cols_dropped": ";".join(dropped) if dropped else "(none)",
            "primary_test": "Firth LRT, M0(CTG) vs M1(CTG+8 maternal)",
            "primary_lr_stat": round(lrt.lr_stat, REPORT_DECIMALS_FILE),
            "primary_df": lrt.df,
            "primary_raw_p": lrt.p_value,
            "primary_loglik_full": round(lrt.loglik_full, REPORT_DECIMALS_FILE),
            "primary_loglik_restricted": round(lrt.loglik_restricted, REPORT_DECIMALS_FILE),
            "sensitivity_test": "Standard MLE LRT, pre-eclampsia removed, M0(CTG) vs M1minusPE(CTG+7 maternal)",
            "sensitivity_lr_stat": round(sens_lr, REPORT_DECIMALS_FILE),
            "sensitivity_df": sens_df,
            "sensitivity_raw_p": sens_p,
            "epv_full_cohort_m0": round(n_pos / m0_cols, REPORT_DECIMALS_FILE),
            "epv_full_cohort_m1": round(n_pos / m1_cols, REPORT_DECIMALS_FILE),
            "epv_per_fold_m0": round(mean_train_pos / m0_cols, REPORT_DECIMALS_FILE),
            "epv_per_fold_m1": round(mean_train_pos / m1_cols, REPORT_DECIMALS_FILE),
            "mean_train_positives_per_fold": round(mean_train_pos, REPORT_DECIMALS_FILE),
        })

    df = pd.DataFrame(rows)
    df["primary_holm_p"] = holm(df["primary_raw_p"].tolist())
    df["sensitivity_holm_p"] = holm(df["sensitivity_raw_p"].tolist())
    return df


# ---------------------------------------------------------------------------
# Step 2: parity
# ---------------------------------------------------------------------------

def odds_ratio_2x2(a, b, c, d):
    """a=nulliparous&pos, b=nulliparous&neg, c=parous&pos, d=parous&neg (see caller).
    Returns (OR, log_or_se, used_haldane_anscombe)."""
    cells = [a, b, c, d]
    ha = any(x == 0 for x in cells)
    if ha:
        a, b, c, d = (x + 0.5 for x in cells)
    or_ = (a * d) / (b * c)
    log_or_se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return or_, log_or_se, ha


def step2(X_ctg, X_maternal, y, record_ids, splits) -> dict:
    Xmat_imp = impute_maternal(X_maternal)
    parity_idx = MATERNAL_NAMES.index("parity")
    age_idx = MATERNAL_NAMES.index("age")
    parity_raw = Xmat_imp[:, parity_idx]
    age_raw = Xmat_imp[:, age_idx]

    # ---- 2a. 2x2, whole cohort, POST HOC nulliparous indicator ----
    nulli = (parity_raw == 0).astype(int)
    # rows=nulliparous(1)/parous(0), cols=label(0/1)
    a = int(((nulli == 1) & (y == 1)).sum())  # nulliparous, positive
    b = int(((nulli == 1) & (y == 0)).sum())  # nulliparous, negative
    c = int(((nulli == 0) & (y == 1)).sum())  # parous, positive
    d = int(((nulli == 0) & (y == 0)).sum())  # parous, negative
    table = np.array([[a, b], [c, d]])
    fisher_odds, fisher_p = stats.fisher_exact(table)
    or_, log_or_se, used_ha = odds_ratio_2x2(a, b, c, d)
    ci_low = float(np.exp(np.log(or_) - 1.96 * log_or_se))
    ci_high = float(np.exp(np.log(or_) + 1.96 * log_or_se))
    twobytwo = {
        "label": "POST HOC -- nulliparous indicator selected after observing 22/23 positives "
                 "nulliparous (CLAUDE.md, 2026-08-12 diagnostics); not pre-specified.",
        "n_nulliparous_positive": a, "n_nulliparous_negative": b,
        "n_parous_positive": c, "n_parous_negative": d,
        "fisher_exact_odds_ratio": round(float(fisher_odds), REPORT_DECIMALS_FILE),
        "fisher_exact_p": float(fisher_p),
        "odds_ratio": round(float(or_), REPORT_DECIMALS_FILE),
        "odds_ratio_95ci_low": round(ci_low, REPORT_DECIMALS_FILE),
        "odds_ratio_95ci_high": round(ci_high, REPORT_DECIMALS_FILE),
        "haldane_anscombe_correction_used": used_ha,
    }

    # ---- 2b. LRT M0 vs M2 (raw parity added), df=1, Firth, per checkpoint ----
    lrt_rows = []
    for ck in CHECKPOINTS:
        ctg, ctg_names, dropped = build_ctg_block(X_ctg, ck)
        n_ctg = ctg.shape[0]
        X2 = np.column_stack([np.ones(n_ctg), ctg, parity_raw])
        parity_col_idx = X2.shape[1] - 1
        lrt = firth_lrt(X2, y, test_idx=[parity_col_idx])
        if not lrt.full_fit.converged or not lrt.restricted_fit.converged:
            log_warning("step2b_parity_lrt", f"checkpoint {ck}: convergence issue "
                        f"(full={lrt.full_fit.converged}, restricted={lrt.restricted_fit.converged})")
        lrt_rows.append({
            "checkpoint_minutes": ck,
            "label": "POST HOC parity test",
            "test": "Firth LRT, M0(CTG) vs M2(CTG+raw parity)",
            "lr_stat": round(lrt.lr_stat, REPORT_DECIMALS_FILE),
            "df": lrt.df,
            "raw_p": lrt.p_value,
            "parity_coef": round(float(lrt.full_fit.beta[parity_col_idx]), REPORT_DECIMALS_FILE),
            "parity_se": round(float(lrt.full_fit.se[parity_col_idx]), REPORT_DECIMALS_FILE),
            "parity_odds_ratio": round(float(np.exp(lrt.full_fit.beta[parity_col_idx])), REPORT_DECIMALS_FILE),
        })
    lrt_df = pd.DataFrame(lrt_rows)
    lrt_df["holm_p"] = holm(lrt_df["raw_p"].tolist())

    # ---- 2c. CV discrimination: CTG+parity vs CTG-only, same folds ----
    # Raw (un-imputed) features, per-fold imputation via the sklearn Pipeline --
    # matches run_featfix_cohort.py's convention; see build_ctg_block_raw docstring.
    parity_col_raw = X_maternal[:, parity_idx]
    scratch = Path(tempfile.mkdtemp(prefix="maternal_value_"))
    oof_rows = []
    try:
        for ck in CHECKPOINTS:
            ctg_raw = build_ctg_block_raw(X_ctg, ck)
            feats = {
                "ctg_only_features": ctg_raw,
                "ctg_plus_parity": np.column_stack([ctg_raw, parity_col_raw]),
            }
            for fs_name, X_fs in feats.items():
                for sp in splits:
                    _, _, tp, _ = train_sklearn_quality_fold("logistic_regression", X_fs, y, sp, SEED, scratch)
                    outer_idx = np.asarray(sp["outer_test_idx"], dtype=int)
                    for j, ix in enumerate(outer_idx):
                        oof_rows.append({
                            "checkpoint_minutes": ck, "cohort_name": fs_name, "model": "logistic_regression",
                            "record_id": record_ids.iloc[ix]["record_id"], "y_true": int(y[ix]),
                            "y_proba": float(tp[j]),
                        })
    finally:
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)

    avg = aggregate_repeated_oof_anytime(pd.DataFrame(oof_rows))
    cv_rows = []
    for ck in CHECKPOINTS:
        a_g = avg[(avg.checkpoint_minutes == ck) & (avg.cohort_name == "ctg_plus_parity")].sort_values("record_id")
        b_g = avg[(avg.checkpoint_minutes == ck) & (avg.cohort_name == "ctg_only_features")].sort_values("record_id")
        assert (a_g["record_id"].to_numpy() == b_g["record_id"].to_numpy()).all()
        yt = a_g["y_true"].to_numpy()
        from sklearn.metrics import roc_auc_score
        auc_a = roc_auc_score(yt, a_g["y_proba"].to_numpy())
        auc_b = roc_auc_score(yt, b_g["y_proba"].to_numpy())
        d = delong_test(yt, a_g["y_proba"].to_numpy(), b_g["y_proba"].to_numpy())
        pm = paired_permutation_test(yt, a_g["y_proba"].to_numpy(), b_g["y_proba"].to_numpy(),
                                     n_permutations=PERM_ITERS, seed=SEED)
        cv_rows.append({
            "checkpoint_minutes": ck,
            "auc_ctg_plus_parity": round(auc_a, REPORT_DECIMALS_FILE),
            "auc_ctg_only": round(auc_b, REPORT_DECIMALS_FILE),
            "gain": round(auc_a - auc_b, REPORT_DECIMALS_FILE),
            "delong_p": float(d["p_value"]), "permutation_p": float(pm["p_value"]),
        })
    cv_df = pd.DataFrame(cv_rows)
    cv_df["delong_holm_p"] = holm(cv_df["delong_p"].tolist())
    cv_df["permutation_holm_p"] = holm(cv_df["permutation_p"].tolist())

    # ---- 2d. parity-age correlation ----
    r, p = stats.pearsonr(parity_raw, age_raw)
    corr = {"pearson_r": round(float(r), REPORT_DECIMALS_FILE), "p_value": float(p), "n": len(parity_raw)}

    return {"twobytwo": twobytwo, "lrt": lrt_df, "cv_discrimination": cv_df, "correlation": corr}


# ---------------------------------------------------------------------------
# Step 3: bootstrap CI for the AUC difference (both vs ctg_only_features, LR)
# ---------------------------------------------------------------------------

def bca_ci(observed_diff, boot_diffs, jackknife_diffs, alpha=0.05):
    boot_diffs = np.asarray(boot_diffs)
    n_less = np.sum(boot_diffs < observed_diff)
    n_total = len(boot_diffs)
    # guard against 0/n_total or n/n_total (z would be +/-inf)
    prop = np.clip(n_less / n_total, 1.0 / (2 * n_total), 1 - 1.0 / (2 * n_total))
    z0 = stats.norm.ppf(prop)

    jk = np.asarray(jackknife_diffs)
    jk_mean = jk.mean()
    num = np.sum((jk_mean - jk) ** 3)
    den = 6.0 * (np.sum((jk_mean - jk) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0

    z_lo, z_hi = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)

    def adj(z):
        return stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

    lo_pct, hi_pct = adj(z_lo) * 100, adj(z_hi) * 100
    lo_pct, hi_pct = np.clip([lo_pct, hi_pct], 0.01, 99.99)
    return float(np.percentile(boot_diffs, lo_pct)), float(np.percentile(boot_diffs, hi_pct))


def step3(X_ctg, X_maternal, y, record_ids, splits) -> pd.DataFrame:
    # Raw (un-imputed) features throughout -- see build_ctg_block_raw docstring.
    # This must reproduce run_featfix_cohort.py's OOF exactly (self-verified below).
    scratch = Path(tempfile.mkdtemp(prefix="maternal_value_step3_"))
    oof_rows = []
    try:
        for ck in CHECKPOINTS:
            ctg_raw = build_ctg_block_raw(X_ctg, ck)
            feats = {
                "both": np.column_stack([ctg_raw, X_maternal]),
                "ctg_only_features": ctg_raw,
            }
            for fs_name, X_fs in feats.items():
                for sp in splits:
                    _, _, tp, _ = train_sklearn_quality_fold("logistic_regression", X_fs, y, sp, SEED, scratch)
                    outer_idx = np.asarray(sp["outer_test_idx"], dtype=int)
                    for j, ix in enumerate(outer_idx):
                        oof_rows.append({
                            "checkpoint_minutes": ck, "cohort_name": fs_name, "model": "logistic_regression",
                            "repeat": int(sp["repeat"]),
                            "record_id": record_ids.iloc[ix]["record_id"], "y_true": int(y[ix]),
                            "y_proba": float(tp[j]),
                        })
    finally:
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)

    oof = pd.DataFrame(oof_rows)
    avg = aggregate_repeated_oof_anytime(oof)  # pools all 3 repeats, matches run_featfix_cohort.py exactly

    # ---- self-verification against the published featfix_results.csv ----
    from sklearn.metrics import roc_auc_score
    published = pd.read_csv(FEATFIX_RESULTS)
    verify_rows = []
    for ck in CHECKPOINTS:
        for fs in ["both", "ctg_only_features"]:
            g = avg[(avg.checkpoint_minutes == ck) & (avg.cohort_name == fs)]
            my_auc = roc_auc_score(g["y_true"], g["y_proba"])
            pub = published[(published.checkpoint_minutes == ck) & (published.feature_set == fs)
                            & (published.model == "logistic_regression")]
            pub_auc = float(pub.iloc[0]["roc_auc_risk_set"]) if len(pub) else float("nan")
            verify_rows.append({"checkpoint_minutes": ck, "feature_set": fs,
                                "my_auc": my_auc, "published_auc": pub_auc,
                                "match_4dp": round(my_auc, 4) == round(pub_auc, 4)})
    verify_df = pd.DataFrame(verify_rows)
    if not verify_df["match_4dp"].all():
        log_warning("step3_verification", f"MISMATCH vs published featfix_results.csv:\n{verify_df}")
        raise SystemExit("ABORT: Step 3 retrained OOF does not match published featfix_results.csv. "
                        "See warnings log / verify_df. Not proceeding with bootstrap on unverified data.")

    rng = np.random.default_rng(SEED)
    rows = []
    for ck in CHECKPOINTS:
        both = avg[(avg.checkpoint_minutes == ck) & (avg.cohort_name == "both")].sort_values("record_id").reset_index(drop=True)
        ctg_only = avg[(avg.checkpoint_minutes == ck) & (avg.cohort_name == "ctg_only_features")].sort_values("record_id").reset_index(drop=True)
        assert (both["record_id"].to_numpy() == ctg_only["record_id"].to_numpy()).all()
        yt = both["y_true"].to_numpy()
        p_both = both["y_proba"].to_numpy()
        p_ctg = ctg_only["y_proba"].to_numpy()
        n = len(yt)
        pos_idx = np.where(yt == 1)[0]
        neg_idx = np.where(yt == 0)[0]

        observed_diff = roc_auc_score(yt, p_both) - roc_auc_score(yt, p_ctg)

        boot_diffs = []
        for _ in range(BOOT_ITERS):
            samp_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            samp_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            samp = np.concatenate([samp_pos, samp_neg])
            d = roc_auc_score(yt[samp], p_both[samp]) - roc_auc_score(yt[samp], p_ctg[samp])
            boot_diffs.append(d)
        boot_diffs = np.asarray(boot_diffs)

        # jackknife (leave-one-record-out) for BCa acceleration
        jk_diffs = []
        for i in range(n):
            keep = np.delete(np.arange(n), i)
            if len(np.unique(yt[keep])) < 2:
                continue
            jk_diffs.append(roc_auc_score(yt[keep], p_both[keep]) - roc_auc_score(yt[keep], p_ctg[keep]))

        pct_lo, pct_hi = np.percentile(boot_diffs, [2.5, 97.5])
        bca_lo, bca_hi = bca_ci(observed_diff, boot_diffs, jk_diffs)
        prop_gt0 = float(np.mean(boot_diffs > 0))

        # per-repeat point estimates (each record appears exactly once per repeat)
        repeat_diffs = []
        for r in sorted(oof["repeat"].unique()):
            b_r = oof[(oof.checkpoint_minutes == ck) & (oof.cohort_name == "both") & (oof.repeat == r)].sort_values("record_id")
            c_r = oof[(oof.checkpoint_minutes == ck) & (oof.cohort_name == "ctg_only_features") & (oof.repeat == r)].sort_values("record_id")
            assert (b_r["record_id"].to_numpy() == c_r["record_id"].to_numpy()).all()
            yt_r = b_r["y_true"].to_numpy()
            repeat_diffs.append(roc_auc_score(yt_r, b_r["y_proba"]) - roc_auc_score(yt_r, c_r["y_proba"]))

        rows.append({
            "checkpoint_minutes": ck,
            "auc_both": round(float(roc_auc_score(yt, p_both)), REPORT_DECIMALS_FILE),
            "auc_ctg_only": round(float(roc_auc_score(yt, p_ctg)), REPORT_DECIMALS_FILE),
            "observed_diff": round(float(observed_diff), REPORT_DECIMALS_FILE),
            "percentile_ci_low": round(float(pct_lo), REPORT_DECIMALS_FILE),
            "percentile_ci_high": round(float(pct_hi), REPORT_DECIMALS_FILE),
            "bca_ci_low": round(float(bca_lo), REPORT_DECIMALS_FILE),
            "bca_ci_high": round(float(bca_hi), REPORT_DECIMALS_FILE),
            "proportion_boot_diff_gt_0": round(prop_gt0, REPORT_DECIMALS_FILE),
            "n_bootstrap_iters": BOOT_ITERS,
            "per_repeat_diff_min": round(float(min(repeat_diffs)), REPORT_DECIMALS_FILE),
            "per_repeat_diff_max": round(float(max(repeat_diffs)), REPORT_DECIMALS_FILE),
            "per_repeat_diffs": ";".join(f"{d:.4f}" for d in repeat_diffs),
            "caveat": "Treats the 15-fold-averaged OOF as fixed data; does not repropagate "
                    "fold/repeat resampling uncertainty, so this CI is narrower than the "
                    "true sampling uncertainty of the whole CV procedure.",
        })

    return pd.DataFrame(rows), verify_df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_ctg, X_maternal, y, record_ids = load_data()
    splits = precompute_quality_splits(y, 5, 3, SEED, 0.2)

    print("=== Step 1: LRT (maternal block) ===")
    step1_df = step1(X_ctg, X_maternal, y, splits)
    step1_df.to_csv(OUT_DIR / "step1_lrt_maternal_block.csv", index=False)
    print(step1_df.to_string(index=False))

    print("\n=== Step 2: parity ===")
    step2_res = step2(X_ctg, X_maternal, y, record_ids, splits)
    pd.DataFrame([step2_res["twobytwo"]]).to_csv(OUT_DIR / "step2a_parity_2x2.csv", index=False)
    step2_res["lrt"].to_csv(OUT_DIR / "step2b_parity_lrt.csv", index=False)
    step2_res["cv_discrimination"].to_csv(OUT_DIR / "step2c_parity_cv_discrimination.csv", index=False)
    pd.DataFrame([step2_res["correlation"]]).to_csv(OUT_DIR / "step2d_parity_age_correlation.csv", index=False)
    print("2x2:", step2_res["twobytwo"])
    print(step2_res["lrt"].to_string(index=False))
    print(step2_res["cv_discrimination"].to_string(index=False))
    print("correlation:", step2_res["correlation"])

    print("\n=== Step 3: bootstrap AUC difference ===")
    step3_df, verify_df = step3(X_ctg, X_maternal, y, record_ids, splits)
    step3_df.to_csv(OUT_DIR / "step3_bootstrap_auc_diff.csv", index=False)
    verify_df.to_csv(OUT_DIR / "step3_self_verification.csv", index=False)
    print(step3_df.to_string(index=False))
    print("\nself-verification vs published featfix_results.csv:")
    print(verify_df.to_string(index=False))

    warn_df = pd.DataFrame(warnings_log) if warnings_log else pd.DataFrame(columns=["step", "detail"])
    warn_df.to_csv(OUT_DIR / "step123_warnings.csv", index=False)
    print(f"\nWarnings logged: {len(warn_df)}")
    if len(warn_df):
        print(warn_df.to_string(index=False))

    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
