"""Task C: clinical operating points at target FPR 10 % and 20 %.

PRIMARY (leak-free): in each outer fold, threshold = (1 - FPR) quantile (numpy, linear) of the
scores of the CONTROLS in that fold's validation split (the model is fit on inner-train only
and never sees the validation rows); a test record is called positive if score > threshold.
Deep models: seed-averaged scores (seeds 42-46, reduced encoder), same rule; when evaluated
on a subset (ckfilter at t, primary345), validation controls are restricted to that subset
(DEVIATIONS §4).

Aggregation over repeats: each record gets exactly one call per repeat (from the fold in
which it was a test record). Counts and metrics are computed per repeat and then averaged
over the 3 repeats; the per-repeat range is reported. Bootstrap: 2000 patient resamples
stratified by class; per-repeat metrics are recomputed on each resample and averaged.

SECONDARY (OPTIMISTIC, labelled): threshold = (1 - FPR) quantile of the controls' pooled
repeat-averaged OOF scores, applied to those same scores.

    python -m src.revision.run_task_c
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.revision import figures as F
from src.revision.cohorts import CHECKPOINTS, VARIANTS, labelled_meta, members, quality_table
from src.revision.raw_cache import RawData
from src.revision.stats_utils import BOOT_ITERS, SEED, strat_boot_index

OUT = Path("results/revision")
FPRS = [0.10, 0.20]
CLASSICAL = [("logistic_regression", "ctg_only"), ("logistic_regression", "combined"),
             ("random_forest", "ctg_only"), ("random_forest", "combined")]
MODEL_NAME = {("logistic_regression", "ctg_only"): "LR CTG-only", ("logistic_regression", "combined"): "LR CTG+maternal",
              ("random_forest", "ctg_only"): "RF CTG-only", ("random_forest", "combined"): "RF CTG+maternal",
              ("deep", "ctg_only"): "Deep CTG-only (reduced, 5 seeds)", ("deep", "fusion"): "Deep fusion (reduced, 5 seeds)"}


def threshold(val_scores_controls: np.ndarray, fpr: float) -> float:
    return float(np.quantile(val_scores_controls, 1.0 - fpr))


def metrics_from_counts(tp, fn, fp, tn):
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
                "ppv": np.where(tp + fp > 0, tp / (tp + fp), np.nan),
                "npv": np.where(tn + fn > 0, tn / (tn + fn), np.nan)}


def summarise(calls: np.ndarray, y: np.ndarray) -> dict:
    """calls: [R, n] binary, one row per repeat. Returns point + bootstrap CIs, averaged over repeats."""
    def counts(idx):
        yy = y[idx]  # [B, n] or [n]
        c = calls[:, idx]  # [R, B, n] or [R, n]
        tp = (c * yy).sum(-1)
        fn = ((1 - c) * yy).sum(-1)
        fp = (c * (1 - yy)).sum(-1)
        tn = ((1 - c) * (1 - yy)).sum(-1)
        return tp, fn, fp, tn

    tp, fn, fp, tn = counts(np.arange(len(y)))
    m = metrics_from_counts(tp.astype(float), fn, fp, tn)
    out = {"n": len(y), "cases": int(y.sum()),
           "TP": tp.mean(), "FN": fn.mean(), "FP": fp.mean(), "TN": tn.mean(),
           "cases_detected": f"{tp.mean():.1f} of {int(y.sum())}",
           "TP_per_repeat": ";".join(str(int(v)) for v in tp)}
    for k, v in m.items():
        out[k] = float(np.nanmean(v))
        out[f"{k}_repeat_min"] = float(np.nanmin(v)) if np.isfinite(v).any() else np.nan
        out[f"{k}_repeat_max"] = float(np.nanmax(v)) if np.isfinite(v).any() else np.nan
    idx = strat_boot_index(y, BOOT_ITERS, SEED)
    btp, bfn, bfp, btn = counts(idx)  # [R, B]
    bm = metrics_from_counts(btp.astype(float), bfn, bfp, btn)
    for k, v in bm.items():
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                avg = np.nanmean(v, axis=0)
        avg = avg[np.isfinite(avg)]
        out[f"{k}_ci_low"] = float(np.percentile(avg, 2.5)) if len(avg) else np.nan
        out[f"{k}_ci_high"] = float(np.percentile(avg, 97.5)) if len(avg) else np.nan
    return out


def leak_free_calls(test: pd.DataFrame, val: pd.DataFrame, ids: list[str], fpr: float):
    """test/val: rows with repeat, fold, record_id, y_true, y_proba. Returns calls [3, n], y, thresholds."""
    pos = {r: i for i, r in enumerate(ids)}
    y = np.zeros(len(ids), dtype=int)
    calls = np.full((3, len(ids)), -1, dtype=int)
    thr_rows = []
    for (rep, fold), g in test.groupby(["repeat", "fold"]):
        v = val[(val.repeat == rep) & (val.fold == fold) & (val.y_true == 0)]
        thr = threshold(v["y_proba"].to_numpy(), fpr)
        thr_rows.append({"repeat": rep, "fold": fold, "n_val_controls": len(v), "threshold": thr})
        for rid, yt, p in zip(g["record_id"], g["y_true"], g["y_proba"]):
            i = pos[rid]
            calls[rep - 1, i] = int(p > thr)
            y[i] = yt
    assert (calls >= 0).all(), "a record lacks a call in some repeat"
    return calls, y, pd.DataFrame(thr_rows)


def main() -> None:
    raw = RawData()
    meta = labelled_meta(raw)
    q = quality_table(raw)
    rf = dict(dtype={"record_id": str}, float_precision="round_trip")
    c_oof = pd.read_csv(OUT / "oof" / "classical_oof.csv", **rf)
    c_val = pd.read_csv(OUT / "valpred" / "classical_valpred.csv", **rf)
    d_oof = pd.read_csv(OUT / "oof" / "deep_oof.csv", **rf)
    d_val = pd.read_csv(OUT / "valpred" / "deep_valpred.csv", **rf)
    keys = ["model", "repeat", "fold", "checkpoint_minutes", "record_id"]
    d_oof = d_oof.groupby(keys, as_index=False).agg(y_true=("y_true", "first"), y_proba=("y_proba", "mean"))
    d_val = d_val.groupby(keys, as_index=False).agg(y_true=("y_true", "first"), y_proba=("y_proba", "mean"))

    rows, thr_all = [], []
    for variant in VARIANTS:
        for t in CHECKPOINTS:
            ids = members(meta, q, variant, t)["record_id"].tolist()
            idset = set(ids)
            sources = []
            for model, fs in CLASSICAL:
                sel = lambda d: d[(d.variant == variant) & (d.checkpoint_minutes == t) & (d.model == model) & (d.feature_set == fs)]
                sources.append(((model, fs), sel(c_oof), sel(c_val), "fit on this cohort"))
            for m in ["ctg_only", "fusion"]:
                sel = lambda d: d[(d.model == m) & (d.checkpoint_minutes == t) & d.record_id.isin(idset)]
                sources.append((("deep", m), sel(d_oof), sel(d_val), "trained on unfiltered; test and validation restricted to this cohort"))
            for key, te, va, note in sources:
                assert set(te["record_id"]) == idset
                for fpr in FPRS:
                    calls, y, thr = leak_free_calls(te, va, ids, fpr)
                    s = summarise(calls, y)
                    thr_all.append(thr.assign(variant=variant, checkpoint_minutes=t, model=MODEL_NAME[key], target_fpr=fpr))
                    rows.append({"variant": variant, "checkpoint_minutes": t, "model": MODEL_NAME[key],
                                 "target_fpr": fpr, "method": "PRIMARY leak-free: validation-control threshold per fold",
                                 "training_note": note, "min_val_controls_per_fold": int(thr["n_val_controls"].min()), **s})
                    # optimistic secondary
                    avg = te.groupby("record_id").agg(y=("y_true", "first"), p=("y_proba", "mean")).loc[ids]
                    yy = avg["y"].to_numpy()
                    thr_o = threshold(avg["p"].to_numpy()[yy == 0], fpr)
                    s2 = summarise((avg["p"].to_numpy() > thr_o).astype(int)[None, :], yy)
                    rows.append({"variant": variant, "checkpoint_minutes": t, "model": MODEL_NAME[key],
                                 "target_fpr": fpr, "method": "SECONDARY OPTIMISTIC: threshold chosen on pooled OOF of the same records",
                                 "training_note": note, "min_val_controls_per_fold": np.nan, **s2})
    df = pd.DataFrame(rows)
    df.round(4).to_csv(OUT / "C_operating_points.csv", index=False)
    pd.concat(thr_all).round(6).to_csv(OUT / "C_fold_thresholds.csv", index=False)
    figure(df)
    print("Task C complete.")


def figure(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    d = df[(df.variant == "ckfilter") & df.method.str.startswith("PRIMARY")]
    models = list(MODEL_NAME.values())
    cols = F.C + ["#eda100", "#e87ba4", "#008300"]
    marks = ["o", "s", "^", "D", "v", "P"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
    for ax, fpr in zip(axes, FPRS):
        for i, m in enumerate(models):
            g = d[(d.model == m) & (d.target_fpr == fpr)].sort_values("checkpoint_minutes")
            x = g.checkpoint_minutes + (i - 2.5) * 0.9
            ax.errorbar(x, g.sensitivity, yerr=[g.sensitivity - g.sensitivity_ci_low, g.sensitivity_ci_high - g.sensitivity],
                        color=cols[i], marker=marks[i], ls="-", lw=1.2, ms=4.5, elinewidth=0.9, label=m)
        ax.axhline(fpr, color=F.MUTED, lw=0.8, ls="--")
        ax.set_title(f"Target false-positive rate {int(fpr * 100)} %", fontsize=9, color=F.INK)
        ax.set_xticks(CHECKPOINTS)
        ax.set_xlabel("t (min)")
    axes[0].set_ylabel("Sensitivity (95 % CI)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.16))
    fig.suptitle("Leak-free operating points, primary cohort (dashed line = chance)", x=0.02, y=1.03, ha="left", fontsize=10)
    F.save(fig, "C1_sensitivity_at_fixed_fpr")


if __name__ == "__main__":
    main()
