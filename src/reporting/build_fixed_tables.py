"""7f: regenerate every classical-model (LR/RF) table from the FIXED-feature results.

The published tables in outputs/report_figures/tables/ (T1, T3, T4, T6, and the ensemble
row of T2) were built from CTG summary features computed over zero padding (RESULTS_FACTS
section 1.1). This script rebuilds their classical content from the fixed-feature result
files into a NEW directory, outputs/report_figures/tables_fixed/. Nothing in tables/ is
modified.

Scope: classical models only, checkpoints 10-60 (90 min kept only as a flagged
supplementary block). Deep-model columns are deliberately absent: the paper reports the
7d retrained deep models, seed-averaged, and those tables belong to 7e. The ensemble is
not rebuilt (no longer a headline model; its deep half is the pre-7d CNN).

No new hypothesis tests are run here. The only new computation is the LR coefficient
rank in TF4 (the old rank came from padded-feature models), refit on the same folds.

Self-checks run before writing and abort on failure:
  * every gapped LR/RF ROC-AUC in TF1 equals featfix_results.csv (risk set) to 4 dp;
  * every TF2 gain equals featfix_ablation_significance.csv to 4 dp.

Usage: python -m src.reporting.build_fixed_tables
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from src.analysis.run_consecutive_label_sensitivity import MATERNAL_NAMES, build_feature_set
from src.data.fixed_ctg_features import FEATURE_NAMES
from src.training.train_cv import load_cohort_arrays, precompute_quality_splits, train_sklearn_quality_fold

QC = Path("outputs/quality_cohorts")
MV = QC / "primary_705_anytime_vr060" / "maternal_value"
FEATFIX = QC / "primary_705_anytime_vr060_featfix"
CONSEC = QC / "primary_705_consec_vr060"
VRSENS = QC / "quality_threshold_sensitivity"
OUT = Path("outputs/report_figures/tables_fixed")
CHECKPOINTS = [10.0, 20.0, 30.0, 45.0, 60.0]
DEC = 4

MODEL_LABELS = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest"}
FS_LABELS = {
    "ctg_only_features": "CTG only",
    "both": "CTG + maternal",
    "maternal_only_features": "maternal only",
    "ctg_plus_parity": "CTG + parity (POST HOC)",
}


def md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def write(df: pd.DataFrame, name: str, note: str, manifest: list, sources: list, display: pd.DataFrame | None = None):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    body = f"# {name}\n\n{note}\n\nSources: " + ", ".join(f"`{s}`" for s in sources) + "\n\n"
    (OUT / f"{name}.md").write_text(body + md(display if display is not None else df), encoding="utf-8")
    manifest.append({"table": name, "rows": len(df), "sources": "; ".join(sources), "note": note})
    print(f"  wrote {name} ({len(df)} rows)")


def ci(v, lo, hi):
    return f"{v:.4f} [{lo:.4f}, {hi:.4f}]"


def holm(p):
    return multipletests(p, method="holm")[1]


# ---------------------------------------------------------------- TF1 discrimination
def tf1(manifest):
    cv = pd.read_csv(CONSEC / "consec_cv_results.csv")
    pub = pd.read_csv(FEATFIX / "featfix_results.csv")
    g = cv[cv.label_design == "gapped"].merge(
        pub, on=["checkpoint_minutes", "feature_set", "model"], how="inner")
    bad = g[g.roc_auc.round(4) != g.roc_auc_risk_set.round(4)]
    if len(g) != 30 or len(bad):
        raise SystemExit(f"ABORT TF1 self-check: {len(g)} rows matched, {len(bad)} mismatches")
    print(f"  TF1 self-check: gapped ROC-AUC matches featfix_results.csv at {len(g)}/30 rows (4 dp)")

    df = cv.copy()
    df["feature_set_label"] = df.feature_set.map(FS_LABELS)
    df["model_label"] = df.model.map(MODEL_LABELS)
    df["post_hoc"] = df.feature_set.eq("ctg_plus_parity")
    df = df.sort_values(["label_design", "checkpoint_minutes", "model", "feature_set"],
                        key=lambda s: s.map({"gapped": 0, "consecutive": 1}) if s.name == "label_design" else s)
    disp = df.assign(ROC=lambda d: [ci(*r) for r in d[["roc_auc", "roc_ci_low", "roc_ci_high"]].to_numpy()],
                     PR=lambda d: [ci(*r) for r in d[["pr_auc", "pr_ci_low", "pr_ci_high"]].to_numpy()])[
        ["label_design", "checkpoint_minutes", "model_label", "feature_set_label", "n_risk_set",
         "positives_risk_set", "ROC", "PR"]]
    write(df, "TF1_classical_discrimination",
          "Classical-model discrimination, fixed CTG features, risk-set scoring, checkpoints 10-60. "
          "ROC/PR-AUC with 1000-iteration bootstrap 95% CI on repeat-averaged OOF (5x3 CV, seed 42). "
          "Gapped = primary label; consecutive = sensitivity. 'CTG + parity' is POST HOC. "
          "Replaces the LR/RF rows of T1. The CI procedure matches 7b (consec_cv_results.csv); "
          "fixed_full_table.csv carries slightly different CIs for the same point estimates "
          "(e.g. 60 min LR CTG+maternal [0.7289, 0.8882] there vs [0.7263, 0.8910] here), from a "
          "separate bootstrap run.",
          manifest, [str(CONSEC / "consec_cv_results.csv"), str(FEATFIX / "featfix_results.csv")], disp)

    sup = pub[pub.checkpoint_minutes == 90.0][["checkpoint_minutes", "feature_set", "model", "n_risk_set",
                                                "positives_risk_set", "roc_auc_risk_set", "pr_auc_risk_set",
                                                "reliably_evaluable"]].copy()
    sup["feature_set"] = sup.feature_set.map(FS_LABELS)
    sup["model"] = sup.model.map(MODEL_LABELS)
    write(sup, "TF1s_supplementary_90min",
          "SUPPLEMENTARY ONLY. 90-min checkpoint, gapped label, fixed features, risk set = records with "
          "duration >= 90 min. NOT reliably evaluable (4 positives); excluded from all inference.",
          manifest, [str(FEATFIX / "featfix_results.csv")])


# ---------------------------------------------------------------- TF2 modality ablation
def tf2(manifest):
    sig = pd.read_csv(FEATFIX / "featfix_ablation_significance.csv")
    sig = sig[sig.checkpoint_minutes.isin(CHECKPOINTS)].copy()
    cv = pd.read_csv(CONSEC / "consec_cv_results.csv")
    wide = cv.pivot_table(index=["label_design", "model", "checkpoint_minutes"], columns="feature_set",
                          values="roc_auc").reset_index()
    wide["gain"] = (wide["both"] - wide["ctg_only_features"]).round(DEC)

    gp = wide[wide.label_design == "gapped"].merge(sig, on=["model", "checkpoint_minutes"],
                                                   suffixes=("_mine", "_pub"))
    if len(gp) != 10 or (gp.gain_mine - gp.gain_pub).abs().max() > 1.5e-4:
        raise SystemExit("ABORT TF2 self-check: gains differ from featfix_ablation_significance.csv")
    print(f"  TF2 self-check: {len(gp)}/10 gapped gains match featfix_ablation_significance.csv")

    for m in MODEL_LABELS:
        sel = sig.model == m
        sig.loc[sel, "delong_holm_p"] = holm(sig.loc[sel, "delong_p"].to_numpy())
        sig.loc[sel, "permutation_holm_p"] = holm(sig.loc[sel, "permutation_p"].to_numpy())

    g = wide[wide.label_design == "gapped"].merge(
        sig[["model", "checkpoint_minutes", "delong_p", "permutation_p", "delong_holm_p", "permutation_holm_p"]],
        on=["model", "checkpoint_minutes"])
    c = wide[wide.label_design == "consecutive"][["model", "checkpoint_minutes", "ctg_only_features", "both", "gain"]]
    c = c.rename(columns={"ctg_only_features": "consec_ctg_only", "both": "consec_both", "gain": "consec_gain"})
    out = g.rename(columns={"ctg_only_features": "gapped_ctg_only", "maternal_only_features": "gapped_maternal_only",
                            "both": "gapped_both", "gain": "gapped_gain"}).merge(c, on=["model", "checkpoint_minutes"])
    out["model"] = out.model.map(MODEL_LABELS)
    out = out[["model", "checkpoint_minutes", "gapped_ctg_only", "gapped_maternal_only", "gapped_both",
               "gapped_gain", "delong_p", "delong_holm_p", "permutation_p", "permutation_holm_p",
               "consec_ctg_only", "consec_both", "consec_gain"]].sort_values(["model", "checkpoint_minutes"])
    out = out.round(DEC)
    write(out, "TF2_modality_ablation",
          "Modality ablation, ROC-AUC, fixed features, identical folds within each label. gain = CTG+maternal "
          "minus CTG-only. Paired DeLong/permutation p (gapped only) with Holm across the 5 checkpoints per "
          "model. No ablation test was run for the consecutive label (gains descriptive). Replaces T6. The "
          "primary test of the maternal block is the Firth LRT in TF3, not these AUC comparisons. The gain is "
          "threshold-dependent (TF5).",
          manifest, [str(FEATFIX / "featfix_ablation_significance.csv"), str(CONSEC / "consec_cv_results.csv")])


# ---------------------------------------------------------------- TF3 maternal-block tests
def tf3(manifest):
    f = pd.read_csv(CONSEC / "consec_firth_lrt.csv")
    s1 = pd.read_csv(MV / "step1_lrt_maternal_block.csv")[
        ["checkpoint_minutes", "sensitivity_lr_stat", "sensitivity_df", "sensitivity_raw_p", "sensitivity_holm_p"]]
    s1["label_design"] = "gapped"
    out = f.merge(s1, on=["label_design", "checkpoint_minutes"], how="left")
    out = out[["label_design", "checkpoint_minutes", "maternal_lr_stat", "maternal_df", "maternal_raw_p",
               "maternal_holm_p", "sensitivity_lr_stat", "sensitivity_df", "sensitivity_raw_p", "sensitivity_holm_p",
               "parity_lr_stat", "parity_df", "parity_raw_p", "parity_holm_p", "parity_or_per_additional_birth",
               "parity_or_95ci_low", "parity_or_95ci_high", "all_converged"]]
    out = out.rename(columns={
        "maternal_lr_stat": "firth_maternal_chi2", "maternal_df": "firth_maternal_df",
        "maternal_raw_p": "firth_maternal_p", "maternal_holm_p": "firth_maternal_holm_p",
        "sensitivity_lr_stat": "mle_noPE_chi2", "sensitivity_df": "mle_noPE_df",
        "sensitivity_raw_p": "mle_noPE_p", "sensitivity_holm_p": "mle_noPE_holm_p",
        "parity_lr_stat": "POSTHOC_parity_chi2", "parity_df": "POSTHOC_parity_df",
        "parity_raw_p": "POSTHOC_parity_p", "parity_holm_p": "POSTHOC_parity_holm_p",
        "parity_or_per_additional_birth": "POSTHOC_parity_OR_per_birth",
        "parity_or_95ci_low": "POSTHOC_parity_OR_low", "parity_or_95ci_high": "POSTHOC_parity_OR_high"})
    out = out.sort_values(["label_design", "checkpoint_minutes"], ascending=[False, True])
    num = out.select_dtypes("number").columns
    out[num] = out[num].apply(lambda c: c.round(DEC) if not c.name.endswith("_p") else c.map(lambda v: float(f"{v:.3g}") if pd.notna(v) else v))
    write(out, "TF3_maternal_block_tests",
          "Single-fit likelihood-ratio tests, whole-cohort median imputation, Holm across 5 checkpoints within "
          "each test and label. PRIMARY (pre-specified): Firth LRT, CTG (10) vs CTG + 8 maternal (18), df 8. "
          "Sensitivity (gapped only): standard MLE with pre-eclampsia removed, df 7. POSTHOC: Firth LRT adding "
          "raw parity count, df 1; OR per additional birth (<1 = protective). Replaces the significance column "
          "of T6 as the maternal-contribution test.",
          manifest, [str(CONSEC / "consec_firth_lrt.csv"), str(MV / "step1_lrt_maternal_block.csv")])


# ---------------------------------------------------------------- TF4 maternal diagnostics
def lr_coef_ranks():
    """Refit LR (CTG + maternal, fixed features, gapped) on each fold's inner-train and rank
    |standardised coefficient| (1 = largest of 18). Same Pipeline and folds as TF1."""
    X_ctg, X_mat, y, _, _ = load_cohort_arrays(Path("data/processed"), "_705_anytime_vr060")
    splits = precompute_quality_splits(y, 5, 3, 42, 0.2)
    names = FEATURE_NAMES + MATERNAL_NAMES
    rows = []
    scratch = Path(tempfile.mkdtemp(prefix="7f_rank_"))
    try:
        for ck in CHECKPOINTS:
            X = build_feature_set("both", X_ctg, X_mat, ck)
            for sp in splits:
                model, _, _, _ = train_sklearn_quality_fold("logistic_regression", X, y, sp, 42, scratch)
                coef = model.named_steps["model"].coef_.ravel()
                rank = pd.Series(np.abs(coef)).rank(ascending=False, method="min").to_numpy()
                for nm, c, r in zip(names, coef, rank):
                    rows.append({"checkpoint_minutes": ck, "fold_name": sp["fold_name"], "feature": nm,
                                 "coef": float(c), "abs_coef": float(abs(c)), "rank": int(r)})
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return pd.DataFrame(rows)


def tf4(manifest):
    ranks = lr_coef_ranks()
    per_ck = ranks.groupby(["feature", "checkpoint_minutes"]).agg(
        mean_rank=("rank", "mean"), mean_coef=("coef", "mean"), mean_abs_coef=("abs_coef", "mean")).reset_index()
    wide = per_ck.pivot(index="feature", columns="checkpoint_minutes", values="mean_rank")
    wide.columns = [f"lr_mean_rank_{int(c)}min" for c in wide.columns]
    overall = ranks.groupby("feature").agg(lr_mean_rank_10_60=("rank", "mean"),
                                           lr_mean_signed_coef_10_60=("coef", "mean")).reset_index()
    diag = pd.read_csv(Path("outputs/quality_cohorts/primary_705_anytime_vr060/maternal_variable_diagnostics.csv"))
    out = diag[["variable", "kind", "n_flag1", "acidemia_rate_flag1", "acidemia_rate_flag0", "odds_ratio",
                "fisher_p", "pos_mean", "neg_mean"]].merge(
        overall.rename(columns={"feature": "variable"}), on="variable", how="left").merge(
        wide.reset_index().rename(columns={"feature": "variable"}), on="variable", how="left").round(DEC)
    write(out, "TF4_maternal_diagnostics",
          "Per-variable diagnostics (gapped cohort, label-level, unaffected by the feature fix) plus LR "
          "|standardised coefficient| rank of 18 features (1 = largest), refit with FIXED CTG features on the "
          "inner-train split of all 15 folds at each checkpoint 10-60 and averaged. Replaces T4, whose rank "
          "came from padded-feature models, a single best fold, and checkpoints 10-90. Parity was chosen for "
          "study POST HOC; the rank is descriptive.",
          manifest, ["outputs/quality_cohorts/primary_705_anytime_vr060/maternal_variable_diagnostics.csv",
                     "refit: train_sklearn_quality_fold(logistic_regression), fixed features"])
    long = ranks.groupby(["checkpoint_minutes", "feature"]).agg(
        mean_rank=("rank", "mean"), min_rank=("rank", "min"), max_rank=("rank", "max"),
        mean_coef=("coef", "mean"), sd_coef=("coef", "std")).reset_index().round(DEC)
    write(long.sort_values(["checkpoint_minutes", "mean_rank"]), "TF4b_lr_coefficient_ranks_by_checkpoint",
          "Long form of TF4's LR rank: per checkpoint and feature, mean/min/max rank over 15 folds and mean/SD "
          "of the standardised coefficient.", manifest, ["refit as TF4"])
    return per_ck


# ---------------------------------------------------------------- TF5 quality thresholds
def tf5(manifest):
    cv = pd.read_csv(VRSENS / "vrsens_cv_results.csv")
    cv = cv[cv.checkpoint_minutes.isin(CHECKPOINTS)].copy()
    cv["feature_set"] = cv.feature_set.map(FS_LABELS)
    cv["model"] = cv.model.map(MODEL_LABELS)
    disp = cv[cv.checkpoint_minutes == 60.0].assign(
        ROC=lambda d: [ci(*r) for r in d[["roc_auc", "roc_ci_low", "roc_ci_high"]].to_numpy()]).pivot_table(
        index=["model", "feature_set"], columns="threshold", values="ROC", aggfunc="first").reset_index()
    write(cv, "TF5_quality_threshold_discrimination",
          "Quality-threshold sensitivity (7c), gapped label, fixed features, LR/RF ROC/PR-AUC with bootstrap CI "
          "at checkpoints 10-60 (CSV); display = 60 min ROC-AUC [95% CI]. Descriptive only. vr040 is not a "
          "predefined threshold. 0.60 was selected after exploratory cross-threshold runs (RESULTS_FACTS 10.16).",
          manifest, [str(VRSENS / "vrsens_cv_results.csv")], disp)
    f = pd.read_csv(VRSENS / "vrsens_firth_lrt.csv")
    f = f[["threshold", "checkpoint_minutes", "maternal_lr_stat", "maternal_df", "maternal_raw_p", "maternal_holm_p",
           "parity_lr_stat", "parity_raw_p", "parity_holm_p", "parity_or_per_additional_birth",
           "parity_or_95ci_low", "parity_or_95ci_high", "all_converged"]].round(DEC)
    write(f, "TF5b_quality_threshold_firth",
          "Firth LRT by quality threshold, Holm across 5 checkpoints within each threshold. Maternal block df 8 "
          "(primary); parity df 1 (POST HOC).", manifest, [str(VRSENS / "vrsens_firth_lrt.csv")])


# ---------------------------------------------------------------- TF6 classical vs deep
def tf6(manifest):
    s = pd.read_csv(MV / "fixed_significance_tests.csv")
    s = s[s.comparison_type == "within_checkpoint_baseline"].copy()
    s = s[s.checkpoint_minutes_a.isin(CHECKPOINTS)]
    for m in MODEL_LABELS:
        sel = s.model_b == m
        s.loc[sel, "delong_holm_p"] = holm(s.loc[sel, "delong_p"].to_numpy())
        s.loc[sel, "permutation_holm_p"] = holm(s.loc[sel, "permutation_p"].to_numpy())
    s["classical_minus_deep"] = -s.auc_diff
    out = s[["checkpoint_minutes_a", "model_a", "model_b", "n_paired_records", "n_pos_paired", "auc_a", "auc_b",
             "classical_minus_deep", "delong_p", "delong_holm_p", "permutation_p", "permutation_holm_p"]].rename(
        columns={"checkpoint_minutes_a": "checkpoint_minutes", "model_a": "deep_model",
                 "model_b": "classical_model", "auc_a": "auc_deep_ORIGINAL", "auc_b": "auc_classical_fixed"})
    out["classical_model"] = out.classical_model.map(MODEL_LABELS)
    write(out.round(DEC), "TF6_classical_vs_deep_PROVISIONAL",
          "PROVISIONAL. Fixed LR/RF (CTG + maternal) vs deep ctg_only, paired DeLong/permutation, gapped, Holm "
          "across 5 checkpoints per classical model. The deep side is the ORIGINAL (pre-7d, not retrained, "
          "single-seed) model; this table is superseded when 7e produces the retrained, seed-averaged deep OOF. "
          "Replaces the classical rows of T3.",
          manifest, [str(MV / "fixed_significance_tests.csv")])


# ---------------------------------------------------------------- TF7 calibration
def tf7(manifest):
    m = pd.read_csv(MV / "step7f_calibration_metrics.csv")
    keep = ["label_design", "checkpoint_minutes", "model", "version", "oof", "n", "positives", "prevalence",
            "mean_predicted", "observed_over_expected", "brier", "brier_boot_low", "brier_boot_high",
            "brier_reference", "scaled_brier", "cal_intercept", "cal_intercept_boot_low", "cal_intercept_boot_high",
            "cal_slope", "cal_slope_boot_low", "cal_slope_boot_high", "cal_slope_wald_low", "cal_slope_wald_high",
            "roc_auc", "brier_repeat_min", "brier_repeat_max", "cal_intercept_repeat_min", "cal_intercept_repeat_max",
            "cal_slope_repeat_min", "cal_slope_repeat_max"]
    out = m[[c for c in keep if c in m.columns]]
    main = out[(out.version == "as_fitted") & (out.oof == "repeat-averaged")]
    disp = main.assign(
        Brier=lambda d: [ci(*r) for r in d[["brier", "brier_boot_low", "brier_boot_high"]].to_numpy()],
        intercept=lambda d: [ci(*r) for r in d[["cal_intercept", "cal_intercept_boot_low", "cal_intercept_boot_high"]].to_numpy()],
        slope=lambda d: [ci(*r) for r in d[["cal_slope", "cal_slope_boot_low", "cal_slope_boot_high"]].to_numpy()],
    )[["label_design", "model", "prevalence", "mean_predicted", "Brier", "brier_reference", "intercept", "slope"]]
    write(out, "TF7_calibration_lr_60min",
          "Calibration, LR at 60 min, fixed features, both labels. Primary rows: version=as_fitted, "
          "oof=repeat-averaged (display). Also in CSV: single-repeat mean/min/max, and a POST HOC prior-corrected "
          "version removing the class_weight='balanced' logit shift. CIs: 2000-iteration stratified bootstrap, "
          "percentile (and Wald for the slope); both treat OOF as fixed and so are too narrow. Ideal: intercept "
          "0, slope 1. Figure: outputs/report_figures/calibration_lr_60min_fixed.png.",
          manifest, [str(MV / "step7f_calibration_metrics.csv")], disp)


def main():
    manifest = []
    print("Building fixed classical tables ->", OUT)
    tf1(manifest)
    tf2(manifest)
    tf3(manifest)
    per_ck = tf4(manifest)
    tf5(manifest)
    tf6(manifest)
    tf7(manifest)
    man = pd.DataFrame(manifest)
    readme = ("# Fixed classical-model tables (7f)\n\n"
              "Built by `python -m src.reporting.build_fixed_tables`. Classical (LR/RF) content only, fixed CTG "
              "features, checkpoints 10-60. The original tables in `../tables/` are untouched and are "
              "SUPERSEDED for all classical content: T1 LR/RF rows -> TF1; T3 classical rows -> TF6; T4 -> TF4; "
              "T6 -> TF2 + TF3. T2's ensemble row (0.777) is superseded but not rebuilt (ensemble is no longer a "
              "headline model; fixed value 0.8187 at 60 min is in maternal_value/fixed_full_table.csv). "
              "Deep-model tables are 7e.\n\n" + md(man))
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    parity = per_ck[per_ck.feature == "parity"]
    print("  parity mean LR rank by checkpoint:", dict(zip(parity.checkpoint_minutes, parity.mean_rank.round(2))))
    print("Done.")


if __name__ == "__main__":
    main()
