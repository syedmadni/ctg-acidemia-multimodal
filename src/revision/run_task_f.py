"""Task F: regenerate the manuscript tables and figures on the NEW PRIMARY cohort (ckfilter,
causal valid-only features). Layout follows the EJOG manuscript
(Raza_FetalAcidemia_EJOG_JournalLayout.docx): Tables 1-5, Figs 1-6.

- Table 1 maternal characteristics (+ newborn sex, A3c), ckfilter cohort at t = 60 min
- Table 2 Firth LRTs: maternal block (PRIMARY) and parity (POST HOC), OR per birth
- Table 3 combined minus CTG-only AUC by fusion strategy
- Table 4 gated-fusion gate weights -> REMOVED (gated fusion not rerun, DEVIATIONS A2)
- Table 5 calibration of LR at 60 min, OUT-OF-FOLD, class-weighted and unweighted LR,
  primary (ckfilter) and sensitivity (unfiltered) cohorts
- Fig 1 AUC by checkpoint, all models; Fig 2 combined models; Fig 3 ROC curves at 60 min;
  Fig 4 modality contribution; Fig 5 -> REMOVED (gate); Fig 6 calibration.

Deep models: reduced encoder only, seeds 42-46 averaged (the full-size encoder was cancelled).

    python -m src.revision.run_task_f
"""
from __future__ import annotations

import shutil
import tempfile
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from statsmodels.nonparametric.smoothers_lowess import lowess  # noqa: E402

from src.analysis.firth import fit_firth  # noqa: E402
from src.analysis.run_7f_calibration import bootstrap as cal_bootstrap  # noqa: E402
from src.analysis.run_7f_calibration import point_metrics, reliability_bins  # noqa: E402
from src.analysis.run_maternal_value_steps123 import odds_ratio_2x2  # noqa: E402
from src.revision import figures as F  # noqa: E402
from src.revision.causal_features import causal_feature_matrix  # noqa: E402
from src.revision.cohorts import CHECKPOINTS, labelled_meta, load_splits, maternal_matrix, members, quality_table  # noqa: E402
from src.revision.raw_cache import RawData  # noqa: E402
from src.revision.run_task_a import ctg_block_whole_cohort  # noqa: E402
from src.training.train_cv import aggregate_repeated_oof_anytime, train_sklearn_quality_fold  # noqa: E402

OUT = Path("results/revision")
TF = OUT / "tables_final"
PRIMARY = "ckfilter"
SENS = "unfiltered"
LEAD = 60.0
SEED = 42


def fmt_ci(v, lo, hi, d=3):
    return f"{v:.{d}f} ({lo:.{d}f} to {hi:.{d}f})"


# ------------------------------------------------------------------ Table 1

def table1(meta, q) -> pd.DataFrame:
    coh = members(meta, q, PRIMARY, LEAD)
    y = coh["y"].to_numpy()
    rows = []
    cont = [("Maternal age (years), mean", "age"), ("Gravidity, mean", "gravidity"),
            ("Parity, mean", "parity"), ("Gestational age (weeks), mean", "gest_weeks")]
    for lab, c in cont:
        rows.append({"variable": lab, "cases": round(float(coh.loc[y == 1, c].mean()), 2),
                     "controls": round(float(coh.loc[y == 0, c].mean()), 2), "p_value": "-",
                     "missing": int(coh[c].isna().sum())})
    binary = [("Nulliparous", coh["parity"] == 0), ("Diabetes", coh["diabetes"] == 1),
              ("Hypertension", coh["hypertension"] == 1), ("Pre-eclampsia", coh["preeclampsia"] == 1),
              ("Induced labour", coh["induced"] == 1),
              ("Newborn sex code 1", coh["sex"] == 1), ("Newborn sex code 2", coh["sex"] == 2)]
    for lab, m in binary:
        m = m.to_numpy()
        a, b = int((m & (y == 1)).sum()), int((m & (y == 0)).sum())
        c, d = int(y.sum()) - a, int((y == 0).sum()) - b
        p = stats.fisher_exact([[a, b], [c, d]])[1]
        rows.append({"variable": f"{lab}, n (%)", "cases": f"{a} ({100 * a / y.sum():.1f} %)",
                     "controls": f"{b} ({100 * b / (y == 0).sum():.1f} %)", "p_value": round(p, 3), "missing": 0})
    nulli = (coh["parity"] == 0).to_numpy()
    or_, se, _ = odds_ratio_2x2(int((nulli & (y == 1)).sum()), int((nulli & (y == 0)).sum()),
                                int((~nulli & (y == 1)).sum()), int((~nulli & (y == 0)).sum()))
    t = pd.DataFrame(rows)
    t.attrs["footnote"] = (
        f"Primary cohort (checkpoint quality filter) at 60 min: n = {len(y)}, {int(y.sum())} cases, "
        f"{int((y == 0).sum())} controls. The cohort changes with the checkpoint (377/28 at 10 min to "
        f"474/33 at 60 min; A_cohort_summary.csv). Categorical: Fisher exact test; continuous: not "
        f"tested (-), as in the manuscript. Nulliparity OR = {or_:.2f} "
        f"({np.exp(np.log(or_) - 1.96 * se):.2f}-{np.exp(np.log(or_) + 1.96 * se):.2f}), Woolf CI. "
        "Newborn sex is shown as the header codes 1 and 2: neither the PhysioNet database page nor "
        "Chudacek et al. 2014 (BMC Pregnancy Childbirth 14:16) defines which code is male or female. "
        "Sex is descriptive only, never a predictor.")
    return t


# ------------------------------------------------------------------ Table 2

def table2(raw, meta, q) -> pd.DataFrame:
    a = pd.read_csv(OUT / "A_firth_lrt.csv")
    a = a[a.variant == PRIMARY]
    rows = []
    for t in CHECKPOINTS:
        coh = members(meta, q, PRIMARY, t)
        y = coh["y"].to_numpy()
        ctg, _, _ = ctg_block_whole_cohort(causal_feature_matrix(raw, coh["record_id"].to_numpy(), t))
        X = np.column_stack([np.ones(len(y)), ctg, coh["parity"].to_numpy(float)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = fit_firth(X, y)
        beta, se = f.beta[-1], f.se[-1]
        m = a[(a.checkpoint_minutes == t) & (a.test == "maternal_8")].iloc[0]
        p = a[(a.checkpoint_minutes == t) & (a.test == "parity")].iloc[0]
        assert np.isclose(beta, p["coef_first_tested"], atol=1e-4)
        rows.append({"checkpoint_minutes": t, "n": int(m["n"]), "cases": int(m["cases"]),
                     "maternal_chi2_8df": round(m["chi2"], 2), "maternal_holm_p": round(m["holm_p"], 2),
                     "POSTHOC_parity_chi2_1df": round(p["chi2"], 2), "POSTHOC_parity_holm_p": round(p["holm_p"], 3),
                     "POSTHOC_parity_OR_per_birth": round(float(np.exp(beta)), 2),
                     "OR_ci_low": round(float(np.exp(beta - 1.96 * se)), 2),
                     "OR_ci_high": round(float(np.exp(beta + 1.96 * se)), 2)})
    t2 = pd.DataFrame(rows)
    t2.attrs["footnote"] = (
        "Primary cohort at each checkpoint. Firth penalised LRT (logistftest convention), CTG features "
        "+ tested variables vs CTG features only; whole-cohort median imputation; Holm across the five "
        "checkpoints. OR per additional previous birth, Wald CI with the Firth standard error (as in the "
        "published table). Parity is POST HOC. After adjustment for time to delivery (second-stage "
        "duration), the parity association is no longer significant (Holm p 0.43-0.74; Task B).")
    return t2


# ------------------------------------------------------------------ Table 3

def table3() -> pd.DataFrame:
    c = pd.read_csv(OUT / "A_combined_vs_ctg.csv")
    d = pd.read_csv(OUT / "A_deep_fusion_vs_ctg.csv")
    rows = []
    for name, g in (("Logistic regression", c[(c.variant == PRIMARY) & (c.model == "logistic_regression")]),
                    ("Random forest", c[(c.variant == PRIMARY) & (c.model == "random_forest")]),
                    ("Deep fusion, reduced encoder (5 seeds)", d[d.variant == PRIMARY])):
        row = {"fusion_strategy": name}
        for _, r in g.sort_values("checkpoint_minutes").iterrows():
            row[f"{int(r.checkpoint_minutes)} min"] = (f"{r.delta_auc:+.3f} ({r.delta_ci_low:.3f} to "
                                                       f"{r.delta_ci_high:.3f}) p = {r.delong_holm_p:.2f}")
        rows.append(row)
    for name in ("Deep fusion, full-size encoder", "Deep gated fusion (both encoder sizes)"):
        rows.append({"fusion_strategy": name, **{f"{int(t)} min": "not re-estimated (DEVIATIONS A2)" for t in CHECKPOINTS}})
    t3 = pd.DataFrame(rows)
    t3.attrs["footnote"] = (
        "Combined (CTG + maternal) minus CTG-only ROC-AUC, primary cohort. Paired stratified bootstrap "
        "95 % CI (2000); p = paired DeLong, Holm across the five checkpoints. Deep fusion is compared "
        "with the seed-averaged deep CTG-only network; both are trained on the unfiltered cohort and "
        "evaluated on the primary cohort.")
    return t3


# ------------------------------------------------------------------ Table 5 (calibration)

def unweighted_lr():
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                     ("model", LogisticRegression(max_iter=5000, random_state=SEED))])


def calibration_oof(raw, meta, q):
    """OOF at 60 min for class-weighted (Task A, recomputed identically) and unweighted LR."""
    rows = []
    scratch = Path(tempfile.mkdtemp(prefix="revF_"))
    try:
        for variant in (PRIMARY, SENS):
            coh = members(meta, q, variant, LEAD)
            y = coh["y"].to_numpy()
            rid = coh["record_id"].to_numpy()
            splits = load_splits(coh, variant, LEAD)
            ctg = causal_feature_matrix(raw, rid, LEAD)
            mat = maternal_matrix(coh)
            for fs, X in (("ctg_only", ctg), ("combined", np.column_stack([ctg, mat]))):
                for sp in splits:
                    _, _, tp, _ = train_sklearn_quality_fold("logistic_regression", X, y, sp, SEED, scratch)
                    m = unweighted_lr().fit(X[sp["inner_train_idx"]], y[sp["inner_train_idx"]])
                    tu = m.predict_proba(X[sp["outer_test_idx"]])[:, 1]
                    for w, p in (("class_weighted", tp), ("unweighted", tu)):
                        rows += [{"cohort_name": f"{variant}|{fs}|{w}", "checkpoint_minutes": LEAD,
                                  "model": "logistic_regression", "record_id": rid[i], "y_true": int(y[i]),
                                  "y_proba": float(v)} for i, v in zip(sp["outer_test_idx"], p)]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    avg = aggregate_repeated_oof_anytime(pd.DataFrame(rows))
    avg[["variant", "feature_set", "weighting"]] = avg["cohort_name"].str.split("|", expand=True)
    return avg


def table5(avg) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, bins, verify = [], [], []
    a_oof = pd.read_csv(OUT / "oof" / "classical_oof_repeat_averaged.csv", dtype={"record_id": str},
                        float_precision="round_trip")
    for (variant, fs, w), g in avg.groupby(["variant", "feature_set", "weighting"]):
        y, p = g["y_true"].to_numpy(), g["y_proba"].to_numpy()
        m = point_metrics(y, p)
        m.update(cal_bootstrap(y, p))
        rows.append({"cohort": variant, "model": fs, "lr_weighting": w, **m})
        for b in reliability_bins(y, p):
            bins.append({"cohort": variant, "model": fs, "lr_weighting": w, **b})
        if w == "class_weighted":
            ref = a_oof[(a_oof.variant == variant) & (a_oof.feature_set == fs) & (a_oof.checkpoint_minutes == LEAD)
                        & (a_oof.model == "logistic_regression")].set_index("record_id")["y_proba"]
            diff = float(np.max(np.abs(g.set_index("record_id")["y_proba"] - ref.loc[g["record_id"]].to_numpy())))
            verify.append({"check": f"class-weighted OOF == Task A OOF ({variant}, {fs})", "max_abs_diff": diff,
                           "pass": diff < 1e-12})
    return pd.DataFrame(rows), pd.DataFrame(bins), pd.DataFrame(verify)


# ------------------------------------------------------------------ figures

MODELS6 = [("LR CTG-only", "logistic_regression", "ctg_only"), ("LR CTG+maternal", "logistic_regression", "combined"),
           ("RF CTG-only", "random_forest", "ctg_only"), ("RF CTG+maternal", "random_forest", "combined"),
           ("Deep CTG-only (reduced, 5 seeds)", "deep_ctg_only", None), ("Deep fusion (reduced, 5 seeds)", "deep_fusion", None)]
COLS6 = F.C + ["#eda100", "#e87ba4", "#008300"]
MARK6 = ["o", "s", "^", "D", "v", "P"]


def disc_rows(variant=PRIMARY):
    c = pd.read_csv(OUT / "A_discrimination.csv")
    d = pd.read_csv(OUT / "A_deep_discrimination.csv")
    out = {}
    for name, m, fs in MODELS6:
        if fs is None:
            out[name] = d[(d.variant == variant) & (d.model == m)].sort_values("checkpoint_minutes")
        else:
            out[name] = c[(c.variant == variant) & (c.model == m) & (c.feature_set == fs)].sort_values("checkpoint_minutes")
    return out


def auc_panel(ax, rows, names, metric="roc_auc"):
    for i, name in enumerate(names):
        k = [n for n, _, _ in MODELS6].index(name)
        g = rows[name]
        x = g.checkpoint_minutes + (i - (len(names) - 1) / 2) * 0.9
        ax.errorbar(x, g[metric], yerr=[g[metric] - g[f"{metric}_ci_low"], g[f"{metric}_ci_high"] - g[metric]],
                    color=COLS6[k], marker=MARK6[k], lw=1.3, ms=4.5, elinewidth=0.9, label=name)
    ax.set_xticks(CHECKPOINTS)
    ax.set_xlabel("Elapsed recording time, t (min)")


def fig1():
    rows = disc_rows()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    auc_panel(ax, rows, [n for n, _, _ in MODELS6])
    ax.axhline(0.5, color=F.MUTED, lw=0.8, ls="--")
    ax.set_ylabel("ROC-AUC (95 % CI)")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.set_title("Fig. 1. Discrimination by checkpoint, primary cohort (n varies with t)", fontsize=9, loc="left", color=F.INK)
    F.save(fig, "Fig1_auc_by_checkpoint_all_models")


def fig2():
    rows = disc_rows()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    auc_panel(ax, rows, ["LR CTG+maternal", "RF CTG+maternal", "Deep fusion (reduced, 5 seeds)"])
    ax.axhline(0.5, color=F.MUTED, lw=0.8, ls="--")
    ax.set_ylabel("ROC-AUC (95 % CI)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Fig. 2. Combined CTG + maternal models, primary cohort", fontsize=9, loc="left", color=F.INK)
    F.save(fig, "Fig2_auc_combined_models")


def oof60(meta, q):
    ids = set(members(meta, q, PRIMARY, LEAD)["record_id"])
    c = pd.read_csv(OUT / "oof" / "classical_oof_repeat_averaged.csv", dtype={"record_id": str}, float_precision="round_trip")
    d = pd.read_csv(OUT / "oof" / "deep_oof.csv", dtype={"record_id": str}, float_precision="round_trip")
    d = d[d.checkpoint_minutes == LEAD].groupby(["model", "record_id"], as_index=False).agg(
        y_true=("y_true", "first"), y_proba=("y_proba", "mean"))
    out = {}
    for name, m, fs in MODELS6:
        if fs is None:
            g = d[(d.model == m.replace("deep_", "")) & d.record_id.isin(ids)]
        else:
            g = c[(c.variant == PRIMARY) & (c.checkpoint_minutes == LEAD) & (c.model == m) & (c.feature_set == fs)]
        assert set(g.record_id) == ids
        out[name] = g
    return out


def fig3(meta, q):
    rows = disc_rows()
    o = oof60(meta, q)
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], color=F.MUTED, lw=0.8, ls=":")
    lss = ["-", "--", "-", "--", "-", "--"]
    for k, (name, _, _) in enumerate(MODELS6):
        g = o[name]
        fpr, tpr, _ = roc_curve(g.y_true, g.y_proba)
        auc = rows[name].set_index("checkpoint_minutes").loc[LEAD, "roc_auc"]
        ax.plot(fpr, tpr, color=COLS6[k], ls=lss[k], lw=1.6, label=f"{name}, AUC {auc:.3f}")
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")
    ax.legend(fontsize=6.5, loc="lower right")
    n, cases = len(o[MODELS6[0][0]]), int(o[MODELS6[0][0]].y_true.sum())
    ax.set_title(f"Fig. 3. ROC at 60 min, primary cohort (n = {n}, {cases} cases)", fontsize=9, loc="left", color=F.INK)
    F.save(fig, "Fig3_roc_curves_60min")


def fig4():
    c = pd.read_csv(OUT / "A_discrimination.csv")
    d = pd.read_csv(OUT / "A_deep_discrimination.csv")
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.1), sharey=True)
    panels = [("Logistic regression", "logistic_regression"), ("Random forest", "random_forest"), ("Deep, reduced encoder", "deep")]
    for ax, (title, m) in zip(axes, panels):
        if m == "deep":
            series = [("CTG only", d[(d.variant == PRIMARY) & (d.model == "deep_ctg_only")]),
                      ("CTG + maternal", d[(d.variant == PRIMARY) & (d.model == "deep_fusion")])]
        else:
            series = [(F.FS_LABEL[fs], c[(c.variant == PRIMARY) & (c.model == m) & (c.feature_set == fs)])
                      for fs in ("ctg_only", "combined", "maternal_only")]
        for i, (lab, g) in enumerate(series):
            g = g.sort_values("checkpoint_minutes")
            ax.fill_between(g.checkpoint_minutes, g.roc_auc_ci_low, g.roc_auc_ci_high, color=F.C[i], alpha=0.13, lw=0)
            ax.plot(g.checkpoint_minutes, g.roc_auc, color=F.C[i], ls=F.LS[i], marker=F.MARK[i], lw=1.8, ms=4.5, label=lab)
        ax.axhline(0.5, color=F.MUTED, lw=0.8, ls="--")
        ax.set_xticks(CHECKPOINTS)
        ax.set_xlabel("t (min)")
        ax.set_title(title, fontsize=9, color=F.INK)
    axes[0].set_ylabel("ROC-AUC (95 % CI band)")
    axes[0].legend(fontsize=7, loc="lower right")
    axes[2].text(0.02, 0.02, "deep maternal-only not rerun", transform=axes[2].transAxes, fontsize=6.5, color=F.MUTED)
    fig.suptitle("Fig. 4. Contribution of each modality, primary cohort", x=0.02, y=1.04, ha="left", fontsize=10)
    F.save(fig, "Fig4_modality_contribution")


def fig6(avg, bins):
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0), sharex=True, sharey=True)
    for r, w in enumerate(("class_weighted", "unweighted")):
        for cidx, variant in enumerate((PRIMARY, SENS)):
            ax = axes[r, cidx]
            ax.plot([0, 1], [0, 1], color=F.MUTED, lw=0.9, ls=":")
            for i, fs in enumerate(("ctg_only", "combined")):
                g = avg[(avg.variant == variant) & (avg.feature_set == fs) & (avg.weighting == w)]
                sm = lowess(g.y_true, g.y_proba, frac=0.6, it=0, return_sorted=True)
                ax.plot(sm[:, 0], sm[:, 1], color=F.C[i], ls=F.LS[i], lw=1.8)
                b = bins[(bins.cohort == variant) & (bins.model == fs) & (bins.lr_weighting == w)]
                ax.errorbar(b.mean_predicted, b.observed_rate,
                            yerr=[b.observed_rate - b.observed_wilson_low, b.observed_wilson_high - b.observed_rate],
                            fmt=F.MARK[i], color=F.C[i], ms=5, elinewidth=1, label=F.FS_LABEL[fs])
            n, cases = len(g), int(g.y_true.sum())
            ax.set_title(f"{'Primary' if variant == PRIMARY else 'Unfiltered (sensitivity)'}, "
                         f"{'class-weighted' if w == 'class_weighted' else 'unweighted'} LR\n(n = {n}, {cases} cases)",
                         fontsize=8.5, color=F.INK)
            ax.set_xlim(-0.01, 1.0)
            ax.set_ylim(-0.02, 0.6)
            if r == 1:
                ax.set_xlabel("Predicted probability, 60 min (OOF)")
            if cidx == 0:
                ax.set_ylabel("Observed rate (Wilson 95 % CI)")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Fig. 6. Calibration of logistic regression at 60 min (bins + LOWESS; dotted = ideal)",
                 x=0.02, y=1.0, ha="left", fontsize=10)
    F.save(fig, "Fig6_calibration_lr_60min")


def write_table(df: pd.DataFrame, name: str, title: str):
    TF.mkdir(parents=True, exist_ok=True)
    df.to_csv(TF / f"{name}.csv", index=False)
    cols = [str(c) for c in df.columns]
    grid = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    grid += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    md = [f"# {title}", "", *grid, "", df.attrs.get("footnote", "")]
    (TF / f"{name}.md").write_text("\n".join(md), encoding="utf-8")


def main():
    raw = RawData()
    meta = labelled_meta(raw)
    q = quality_table(raw)
    write_table(table1(meta, q), "Table1_maternal_characteristics", "Table 1. Maternal characteristics of cases and controls (primary cohort, 60 min)")
    write_table(table2(raw, meta, q), "Table2_firth_maternal_parity", "Table 2. Maternal block and parity over CTG features (Firth LRT, primary cohort)")
    write_table(table3(), "Table3_delta_auc_fusion", "Table 3. Combined minus CTG-only AUC by fusion strategy (primary cohort)")
    t4 = pd.DataFrame([{"status": "REMOVED", "reason": "Gated fusion was not rerun on the leak-free inputs "
                        "(DEVIATIONS A2); the gate-weight table is removed from the main text."}])
    write_table(t4, "Table4_gate_REMOVED", "Table 4. Gate weights - removed")
    avg = calibration_oof(raw, meta, q)
    t5, bins, ver = table5(avg)
    if not ver["pass"].all():
        print(ver.to_string(index=False))
        raise SystemExit("ABORT: class-weighted calibration OOF does not reproduce Task A OOF")
    ver.to_csv(TF / "Table5_self_verification.csv", index=False)
    bins.to_csv(TF / "Table5_calibration_bins.csv", index=False)
    show = t5[["cohort", "model", "lr_weighting", "n", "positives", "mean_predicted", "prevalence", "brier",
               "brier_reference", "cal_intercept", "cal_intercept_boot_low", "cal_intercept_boot_high",
               "cal_slope", "cal_slope_boot_low", "cal_slope_boot_high", "roc_auc"]].round(3)
    show.attrs["footnote"] = ("Out-of-fold (repeat-averaged) predictions at 60 min. Intercept = calibration-in-the-large "
                              "(GLM with offset logit p), slope = GLM of y on logit p; 2000 stratified bootstrap "
                              "percentile CIs, treating OOF as fixed. Class-weighted = model as published "
                              "(class_weight='balanced', which shifts predicted risk upward by design); "
                              "unweighted = same pipeline without class weights. Ideal: intercept 0, slope 1.")
    write_table(show, "Table5_calibration_lr_60min", "Table 5. Calibration of logistic regression at 60 min")
    for fn in (fig1, fig2, fig4):
        fn()
    fig3(meta, q)
    fig6(avg, bins)
    (TF / "Fig5_gate_REMOVED.md").write_text("Fig. 5 (gate weights) removed: gated fusion not rerun (DEVIATIONS A2).\n",
                                             encoding="utf-8")
    print("Task F complete.")


if __name__ == "__main__":
    main()
