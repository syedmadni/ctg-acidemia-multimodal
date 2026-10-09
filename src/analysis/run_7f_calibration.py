"""7f: calibration of logistic regression at 60 min, fixed CTG features, both labels.

Models: LR on CTG-only features (10) and LR on CTG + 8 maternal (18), each through the
project's classical CV path (train_sklearn_quality_fold, RepeatedStratifiedKFold 5x3,
seed 42, per-fold imputation/scaling inside the Pipeline, fit on inner-train only).
Features and folds are built exactly as in run_consecutive_label_sensitivity.py (7b),
so the OOF predictions here are the ones behind every published fixed LR number.

SELF-VERIFICATION (runs first; aborts on failure):
  * averaged-OOF ROC-AUC must match consec_cv_results.csv (both labels) to 4 dp;
  * gapped averaged OOF must match primary_705_anytime_vr060_featfix/
    oof_predictions_averaged.csv record-for-record.

Calibration metrics (all on the risk set = full cohort at 60 min):
  * Brier score, reference Brier prev*(1-prev), scaled Brier 1 - B/B_ref;
  * mean predicted risk vs observed prevalence (O/E ratio);
  * calibration intercept = calibration-in-the-large: logistic GLM y ~ 1 + offset(logit p);
  * calibration slope: logistic GLM y ~ logit p;
  * Wald 95% CIs from the GLMs, plus 2000-iteration stratified bootstrap percentile CIs.
    Both treat the OOF predictions as fixed, so they understate CV uncertainty.
Primary = the repeat-AVERAGED OOF (project convention). Averaging over 3 repeats shrinks
predictions slightly, so each single repeat (a complete OOF set on its own) is also
reported, as mean / min / max.

POST HOC diagnostic: the project's LR uses class_weight="balanced", which by
construction raises every predicted probability (in logit terms by about
log(n_neg/n_pos) of the inner-train fold, ~2.6 here). A prior-corrected version
logit(p) - log(n_neg/n_pos) per fold is reported so the intercept can be read with and
without that design effect. The slope does not change under a constant logit shift
(it can change slightly after averaging across repeats with fold-specific shifts).

Reliability plot: 5 quantile bins (as in the earlier calibration analysis) with Wilson
95% CIs, plus a LOWESS smooth (it=0, no robustness iterations: with a rare binary outcome
those iterations treat positives as outliers and flatten the curve to zero).

Outputs:
  outputs/quality_cohorts/primary_705_anytime_vr060/maternal_value/
      step7f_calibration_metrics.csv, step7f_calibration_bins.csv,
      step7f_calibration_self_verification.csv, step7f_calibration_oof.csv
  outputs/report_figures/calibration_lr_60min_fixed.png

Usage: python -m src.analysis.run_7f_calibration
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from statsmodels.nonparametric.smoothers_lowess import lowess  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

from src.analysis.run_consecutive_label_sensitivity import build_feature_set  # noqa: E402
from src.training.train_cv import (  # noqa: E402
    load_cohort_arrays,
    precompute_quality_splits,
    train_sklearn_quality_fold,
)

CHECKPOINT = 60.0
SEED = 42
BOOT_ITERS = 2000
N_BINS = 5
EPS = 1e-6
DEC = 4
LABELS = {"gapped": "_705_anytime_vr060", "consecutive": "_705_consec_vr060"}
FEATURE_SETS = {"ctg_only_features": "LR CTG-only", "both": "LR CTG+maternal"}
OUT_DIR = Path("outputs/quality_cohorts/primary_705_anytime_vr060/maternal_value")
FIG_PATH = Path("outputs/report_figures/calibration_lr_60min_fixed.png")
CONSEC_RESULTS = Path("outputs/quality_cohorts/primary_705_consec_vr060/consec_cv_results.csv")
FEATFIX_OOF = Path("outputs/quality_cohorts/primary_705_anytime_vr060_featfix/oof_predictions_averaged.csv")

# dataviz reference palette, categorical slots 1-2 (light mode); series also differ by
# marker and line style so identity never rests on colour alone.
STYLE = {
    "ctg_only_features": {"color": "#2a78d6", "marker": "o", "ls": "--"},
    "both": {"color": "#eb6834", "marker": "s", "ls": "-"},
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def build_oof(label_key, suffix):
    X_ctg, X_maternal, y, record_ids, _ = load_cohort_arrays(Path("data/processed"), suffix)
    splits = precompute_quality_splits(y, 5, 3, SEED, 0.2)
    rows = []
    scratch = Path(tempfile.mkdtemp(prefix=f"7f_{label_key}_"))
    try:
        for fs in FEATURE_SETS:
            X = build_feature_set(fs, X_ctg, X_maternal, CHECKPOINT)
            for sp in splits:
                _, _, tp, _ = train_sklearn_quality_fold("logistic_regression", X, y, sp, SEED, scratch)
                inner = np.asarray(sp["inner_train_idx"], dtype=int)
                n_pos = int(y[inner].sum())
                shift = float(np.log((len(inner) - n_pos) / n_pos))
                for j, ix in enumerate(np.asarray(sp["outer_test_idx"], dtype=int)):
                    rows.append({
                        "label_design": label_key, "feature_set": fs, "repeat": int(sp["repeat"]),
                        "fold": int(sp["fold"]), "record_id": str(record_ids.iloc[ix]["record_id"]),
                        "y_true": int(y[ix]), "y_proba": float(tp[j]),
                        "y_proba_prior_corrected": float(expit(logit(tp[j]) - shift)),
                        "balanced_weight_logit_shift": shift,
                    })
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return pd.DataFrame(rows)


def average_oof(oof, col):
    return (oof.groupby(["label_design", "feature_set", "record_id"], as_index=False)
            .agg(y_true=("y_true", "first"), y_proba=(col, "mean")))


def glm_calibration(y, p):
    z = logit(p)
    citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=z).fit()
    slope = sm.GLM(y, sm.add_constant(z), family=sm.families.Binomial()).fit()
    ci_i, ci_s = citl.conf_int()[0], slope.conf_int()[1]
    return {
        "cal_intercept": float(citl.params[0]), "cal_intercept_wald_low": float(ci_i[0]),
        "cal_intercept_wald_high": float(ci_i[1]),
        "cal_slope": float(slope.params[1]), "cal_slope_wald_low": float(ci_s[0]),
        "cal_slope_wald_high": float(ci_s[1]),
    }


def point_metrics(y, p):
    prev = float(y.mean())
    brier = float(np.mean((p - y) ** 2))
    b_ref = prev * (1 - prev)
    out = {
        "n": int(len(y)), "positives": int(y.sum()), "prevalence": prev,
        "mean_predicted": float(p.mean()), "observed_over_expected": prev / float(p.mean()),
        "brier": brier, "brier_reference": b_ref, "scaled_brier": 1 - brier / b_ref,
        "roc_auc": float(roc_auc_score(y, p)),
    }
    out.update(glm_calibration(y, p))
    return out


def bootstrap(y, p, iters=BOOT_ITERS, seed=SEED):
    """Stratified bootstrap percentile CIs for Brier, intercept, slope."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = {"brier": [], "cal_intercept": [], "cal_slope": []}
    for _ in range(iters):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        yb, pb = y[idx], p[idx]
        vals["brier"].append(np.mean((pb - yb) ** 2))
        g = glm_calibration(yb, pb)
        vals["cal_intercept"].append(g["cal_intercept"])
        vals["cal_slope"].append(g["cal_slope"])
    out = {}
    for k, v in vals.items():
        out[f"{k}_boot_low"] = float(np.percentile(v, 2.5))
        out[f"{k}_boot_high"] = float(np.percentile(v, 97.5))
    return out


def reliability_bins(y, p):
    edges = np.quantile(p, np.linspace(0, 1, N_BINS + 1))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, N_BINS - 1)
    rows = []
    for k in range(N_BINS):
        m = b == k
        n, k_pos = int(m.sum()), int(y[m].sum())
        lo, hi = proportion_confint(k_pos, n, method="wilson")
        rows.append({"bin": k + 1, "n": n, "positives": k_pos, "mean_predicted": float(p[m].mean()),
                     "observed_rate": k_pos / n, "observed_wilson_low": float(lo),
                     "observed_wilson_high": float(hi), "p_min": float(p[m].min()), "p_max": float(p[m].max())})
    return rows


def self_verify(avg_all):
    pub = pd.read_csv(CONSEC_RESULTS)
    rows = []
    for (lab, fs), g in avg_all.groupby(["label_design", "feature_set"]):
        mine = roc_auc_score(g.y_true, g.y_proba)
        p = pub[(pub.label_design == lab) & (pub.checkpoint_minutes == CHECKPOINT)
                & (pub.feature_set == fs) & (pub.model == "logistic_regression")]
        rows.append({"check": f"ROC-AUC vs consec_cv_results.csv ({lab}, {fs})", "mine": round(mine, 6),
                     "published": float(p.iloc[0].roc_auc), "pass": bool(round(mine, 4) == float(p.iloc[0].roc_auc))})
    ff = pd.read_csv(FEATFIX_OOF)
    ff = ff[(ff.checkpoint_minutes == CHECKPOINT) & (ff.model == "logistic_regression")]
    for fs in FEATURE_SETS:
        a = avg_all[(avg_all.label_design == "gapped") & (avg_all.feature_set == fs)].set_index("record_id")
        b = ff[ff.cohort_name == fs].assign(record_id=lambda d: d.record_id.astype(str)).set_index("record_id")
        j = a.join(b, rsuffix="_pub", how="outer")
        diff = float(np.nanmax(np.abs(j.y_proba - j.y_proba_pub))) if j.y_proba_pub.notna().all() else np.inf
        rows.append({"check": f"gapped OOF vs featfix oof_predictions_averaged.csv ({fs}), max |diff|, n={len(j)}",
                     "mine": diff, "published": 0.0, "pass": bool(diff < 1e-12 and len(j) == 345)})
    return pd.DataFrame(rows)


def plot(bins_df, avg_all):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), sharey=True)
    for ax, lab in zip(axes, LABELS):
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":", zorder=1)
        for fs, name in FEATURE_SETS.items():
            st = STYLE[fs]
            g = avg_all[(avg_all.label_design == lab) & (avg_all.feature_set == fs)]
            sm_xy = lowess(g.y_true, g.y_proba, frac=0.6, it=0, return_sorted=True)  # it=0: robustness
            # iterations down-weight the rare positives as outliers and flatten the curve to 0
            ax.plot(sm_xy[:, 0], sm_xy[:, 1], color=st["color"], lw=2, ls=st["ls"], zorder=2)
            bd = bins_df[(bins_df.label_design == lab) & (bins_df.feature_set == fs) & (bins_df.version == "as_fitted")]
            ax.errorbar(bd.mean_predicted, bd.observed_rate,
                        yerr=[bd.observed_rate - bd.observed_wilson_low, bd.observed_wilson_high - bd.observed_rate],
                        fmt=st["marker"], ms=7, color=st["color"], mec="white", mew=1.5, elinewidth=1.2,
                        capsize=0, zorder=3, label=f"{name} (bins, Wilson 95% CI; line = LOWESS)")
            ax.plot(g.y_proba, np.full(len(g), -0.025 if fs == "both" else -0.05), "|",
                    color=st["color"], ms=5, alpha=0.35)
        n, pos = int(g.y_true.size), int(g.y_true.sum())
        ax.set_title(f"{lab.capitalize()} label (n={n}, {pos} positives)", color=INK, fontsize=10.5, loc="left")
        ax.set_xlim(-0.01, 1.0)
        ax.set_ylim(-0.07, 0.45)
        ax.set_xlabel("Predicted probability (60 min, repeat-averaged OOF)", color=INK, fontsize=9.5)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8.5)
    axes[0].set_ylabel("Observed acidemia rate", color=INK, fontsize=9.5)
    axes[0].legend(loc="upper left", fontsize=7.8, frameon=False, labelcolor=INK)
    fig.suptitle("Logistic regression calibration at 60 min (fixed CTG features). Dotted line = perfect "
                 "calibration; ticks below 0 = individual predictions.",
                 fontsize=8.5, color=MUTED, x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=300, facecolor="white")
    # PDF alongside the PNG, so the final/ figure set is consistent (2026-09-24).
    fig.savefig(FIG_PATH.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    oof = pd.concat([build_oof(k, s) for k, s in LABELS.items()], ignore_index=True)
    avg_all = average_oof(oof, "y_proba")

    ver = self_verify(avg_all)
    ver.to_csv(OUT_DIR / "step7f_calibration_self_verification.csv", index=False)
    print(ver.to_string(index=False))
    if not ver["pass"].all():
        raise SystemExit("ABORT: self-verification failed; calibration numbers not trusted.")

    avg_pc = average_oof(oof, "y_proba_prior_corrected")
    metric_rows, bin_rows = [], []
    for lab in LABELS:
        for fs, name in FEATURE_SETS.items():
            for version, src in [("as_fitted", avg_all), ("prior_corrected_POST_HOC", avg_pc)]:
                g = src[(src.label_design == lab) & (src.feature_set == fs)]
                y, p = g.y_true.to_numpy(), g.y_proba.to_numpy()
                row = {"label_design": lab, "checkpoint_minutes": CHECKPOINT, "model": name,
                       "feature_set": fs, "version": version, "oof": "repeat-averaged"}
                row.update(point_metrics(y, p))
                row.update(bootstrap(y, p))
                metric_rows.append(row)
                for b in reliability_bins(y, p):
                    bin_rows.append({"label_design": lab, "feature_set": fs, "model": name, "version": version, **b})
            # per-repeat (as fitted), each a complete OOF set
            reps = []
            for r, g in oof[(oof.label_design == lab) & (oof.feature_set == fs)].groupby("repeat"):
                reps.append(point_metrics(g.y_true.to_numpy(), g.y_proba.to_numpy()))
            reps = pd.DataFrame(reps)
            row = {"label_design": lab, "checkpoint_minutes": CHECKPOINT, "model": name, "feature_set": fs,
                   "version": "as_fitted", "oof": "single repeat: mean [min, max] over 3"}
            for k in ["brier", "cal_intercept", "cal_slope", "roc_auc", "mean_predicted"]:
                row[k] = float(reps[k].mean())
                row[f"{k}_repeat_min"] = float(reps[k].min())
                row[f"{k}_repeat_max"] = float(reps[k].max())
            for k in ["n", "positives", "prevalence"]:
                row[k] = reps[k].iloc[0]
            metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows).round(DEC)
    bins = pd.DataFrame(bin_rows).round(DEC)
    metrics.to_csv(OUT_DIR / "step7f_calibration_metrics.csv", index=False)
    bins.to_csv(OUT_DIR / "step7f_calibration_bins.csv", index=False)
    oof.to_csv(OUT_DIR / "step7f_calibration_oof.csv", index=False)
    plot(bins, avg_all)

    show = ["label_design", "model", "version", "oof", "mean_predicted", "prevalence", "brier", "scaled_brier",
            "cal_intercept", "cal_intercept_boot_low", "cal_intercept_boot_high",
            "cal_slope", "cal_slope_boot_low", "cal_slope_boot_high", "cal_slope_wald_low", "cal_slope_wald_high",
            "brier_boot_low", "brier_boot_high", "roc_auc"]
    print(metrics[[c for c in show if c in metrics.columns]].to_string(index=False))
    rep_cols = [c for c in metrics.columns if c.endswith("_repeat_min") or c.endswith("_repeat_max")]
    print(metrics[metrics.oof.str.startswith("single")][["label_design", "model"] + rep_cols].to_string(index=False))
    print(bins[bins.version == "as_fitted"].to_string(index=False))
    print(f"Wrote {OUT_DIR}/step7f_calibration_*.csv and {FIG_PATH}")


if __name__ == "__main__":
    main()
