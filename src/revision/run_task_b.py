"""Task B: time from checkpoint to delivery. EXPLANATORY / CONFOUNDING CHECK ONLY.
Time to delivery is unknown at prediction time; it is never a deployable predictor.

ttd(t) = (recording length - t) + Sig2Birth; Sig2Birth = 0 in all 552 headers (DEVIATIONS F3),
so ttd(t) = length - t. Within a fixed cohort, ttd(t) is length shifted by a constant, so its
AUC equals the AUC of length at every t, and adjusting for ttd(t) equals adjusting for length.

Pre-declared addition A3a: % of recordings already in the second stage at t
(Pos. II.st. <= t*240 samples; -1 = unknown). Descriptive only.

    python -m src.revision.run_task_b
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
from scipy import stats  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402

from src.analysis.significance import delong_test  # noqa: E402
from src.revision import figures as F  # noqa: E402
from src.revision.causal_features import causal_feature_matrix  # noqa: E402
from src.revision.cohorts import CHECKPOINTS, VARIANTS, labelled_meta, load_splits, maternal_matrix, members, quality_table  # noqa: E402
from src.revision.predictors import MATERNAL_8  # noqa: E402
from src.revision.raw_cache import RawData  # noqa: E402
from src.revision.run_task_a import ctg_block_whole_cohort, firth_block_test  # noqa: E402
from src.revision.stats_utils import auc_pr_ci, holm, paired_delta_ci  # noqa: E402
from src.training.train_cv import aggregate_repeated_oof_anytime, train_sklearn_quality_fold  # noqa: E402

OUT = Path("results/revision")
LABEL = "CONFOUNDING CHECK - time to delivery is unknown at prediction time; not a deployable predictor"
SEED = 42


def iqr_str(x):
    q1, m, q3 = np.percentile(x, [25, 50, 75])
    return round(float(m), 2), round(float(q1), 2), round(float(q3), 2)


def main() -> None:
    raw = RawData()
    meta = labelled_meta(raw)
    q = quality_table(raw)
    assert (meta["sig2birth"] == 0).all()

    # ---------------- B1 sanity + outliers (per record, all 552)
    L = meta["length_min"]
    q1, q3 = L.quantile([0.25, 0.75])
    out = meta[(L < q1 - 3 * (q3 - q1)) | (L > q3 + 3 * (q3 - q1))]
    sanity = {"records": len(meta), "min_length_min": L.min(), "max_length_min": L.max(),
              "min_ttd_at_60": (L - 60).min(), "negative_ttd_any_checkpoint": int(((L - 60) < 0).sum()),
              "sig2birth_nonzero": int((meta["sig2birth"] != 0).sum()),
              "outliers_3IQR_length": ";".join(out["record_id"]) or "(none)"}
    pd.DataFrame([sanity]).round(3).to_csv(OUT / "B_sanity.csv", index=False)

    desc, aucs, stage_rows, firth_rows, cv_rows, len_rows = [], [], [], [], [], []
    oof_rows = []
    scratch = Path(tempfile.mkdtemp(prefix="revB_"))
    try:
        for variant in VARIANTS:
            # manuscript-number re-estimates (fixed per cohort; ckfilter uses its 60-min cohort)
            coh60 = members(meta, q, variant, 60.0)
            y60 = coh60["y"].to_numpy()
            ii = coh60["ii_stage"].where(coh60["ii_stage"] >= 0)
            ok = ii.notna()
            rho, rp = stats.spearmanr(coh60.loc[ok, "length_min"], ii[ok])
            a = auc_pr_ci(y60, coh60["length_min"].to_numpy())
            len_rows.append({"variant": variant, "cohort_at": "t=60" if variant == "ckfilter" else "fixed",
                             "n": len(y60), "cases": int(y60.sum()),
                             "median_length_min": round(float(coh60["length_min"].median()), 2),
                             "length_auc": a["roc_auc"], "length_auc_ci_low": a["roc_auc_ci_low"],
                             "length_auc_ci_high": a["roc_auc_ci_high"],
                             "spearman_length_vs_ii_stage": rho, "spearman_p": rp, "n_ii_stage_known": int(ok.sum()),
                             "label": LABEL})
            for t in CHECKPOINTS:
                coh = members(meta, q, variant, t)
                y = coh["y"].to_numpy()
                ttd = (coh["length_min"] - t + coh["sig2birth"]).to_numpy()
                assert (ttd >= 0).all()
                # ---- B3 descriptive
                mw = stats.mannwhitneyu(ttd[y == 1], ttd[y == 0], alternative="two-sided")
                c_m, c_q1, c_q3 = iqr_str(ttd[y == 1])
                n_m, n_q1, n_q3 = iqr_str(ttd[y == 0])
                desc.append({"variant": variant, "checkpoint_minutes": t, "n": len(y), "cases": int(y.sum()),
                             "ttd_median_cases": c_m, "ttd_q1_cases": c_q1, "ttd_q3_cases": c_q3,
                             "ttd_median_controls": n_m, "ttd_q1_controls": n_q1, "ttd_q3_controls": n_q3,
                             "mann_whitney_p": mw.pvalue, "label": LABEL})
                # ---- B4 AUC of ttd alone (score = +ttd, same orientation as the published length AUC)
                a = auc_pr_ci(y, ttd)
                aucs.append({"variant": variant, "checkpoint_minutes": t, "n": len(y), "cases": int(y.sum()),
                             "score": "+ttd (longer time to delivery ranked higher; AUC < 0.5 means cases are closer to delivery)",
                             "roc_auc": a["roc_auc"], "roc_auc_ci_low": a["roc_auc_ci_low"],
                             "roc_auc_ci_high": a["roc_auc_ci_high"], "label": LABEL})
                # ---- A3a second stage at t
                pos = coh["pos_ii_st"]
                known = pos >= 0
                in2 = known & (pos <= t * 240)
                for grp, m in (("case", y == 1), ("control", y == 0)):
                    k = int((known & m).sum())
                    stage_rows.append({"variant": variant, "checkpoint_minutes": t, "group": grp,
                                       "n": int(m.sum()), "stage_known": k, "unknown_pos_ii_st_minus1": int((~known & m).sum()),
                                       "in_second_stage_at_t": int((in2 & m).sum()),
                                       "pct_in_second_stage_of_known": round(100 * (in2 & m).sum() / k, 1) if k else np.nan,
                                       "label": "descriptive only (A3a)"})
                # ---- B5 Firth with ttd in BOTH nested models
                ctg_raw = causal_feature_matrix(raw, coh["record_id"].to_numpy(), t)
                ctg, _, _ = ctg_block_whole_cohort(ctg_raw)
                ctg_adj = np.column_stack([ctg, ttd])
                mat = SimpleImputer(strategy="median").fit_transform(maternal_matrix(coh))
                for test, B, names in (("maternal_8", mat, MATERNAL_8),
                                       ("parity", coh[["parity"]].to_numpy(float), ["parity"])):
                    r = firth_block_test(ctg_adj, B, names, y)
                    firth_rows.append({"variant": variant, "checkpoint_minutes": t, "test": test,
                                       "adjusted_for": "ttd(t) in both nested models", "n": len(y),
                                       "cases": int(y.sum()), **r,
                                       "label": LABEL + ("; parity POST HOC" if test == "parity" else "")})
                # ---- B5 CV AUCs with and without ttd
                splits = load_splits(coh, variant, t)
                matraw = maternal_matrix(coh)
                fsets = {"ctg_only": ctg_raw, "combined": np.column_stack([ctg_raw, matraw]),
                         "ctg_only+ttd": np.column_stack([ctg_raw, ttd]),
                         "combined+ttd": np.column_stack([ctg_raw, matraw, ttd])}
                for fs, X in fsets.items():
                    for sp in splits:
                        _, _, tp, _ = train_sklearn_quality_fold("logistic_regression", X, y, sp, SEED, scratch)
                        oof_rows += [{"cohort_name": f"{variant}|{fs}", "checkpoint_minutes": t,
                                      "model": "logistic_regression", "record_id": coh["record_id"].iloc[i],
                                      "y_true": int(y[i]), "y_proba": float(v)}
                                     for i, v in zip(np.asarray(sp["outer_test_idx"]), tp)]
                print(f"  B done {variant} t={t:g}", flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    avg = aggregate_repeated_oof_anytime(pd.DataFrame(oof_rows))
    avg[["variant", "feature_set"]] = avg["cohort_name"].str.split("|", expand=True)
    for (variant, t), g in avg.groupby(["variant", "checkpoint_minutes"]):
        by = {fs: gg.sort_values("record_id") for fs, gg in g.groupby("feature_set")}
        y = by["ctg_only"]["y_true"].to_numpy()
        row = {"variant": variant, "checkpoint_minutes": t, "model": "logistic_regression", "label": LABEL}
        for fs, gg in by.items():
            a = auc_pr_ci(y, gg["y_proba"].to_numpy())
            key = fs.replace("+", "_plus_")
            row.update({f"auc_{key}": a["roc_auc"], f"auc_{key}_ci_low": a["roc_auc_ci_low"],
                        f"auc_{key}_ci_high": a["roc_auc_ci_high"]})
        for base in ("ctg_only", "combined"):
            d = paired_delta_ci(y, by[base + "+ttd"]["y_proba"].to_numpy(), by[base]["y_proba"].to_numpy())
            row.update({f"delta_{base}_ttd_minus_without": d["delta_auc"],
                        f"delta_{base}_ci_low": d["delta_ci_low"], f"delta_{base}_ci_high": d["delta_ci_high"]})
        d = paired_delta_ci(y, by["combined+ttd"]["y_proba"].to_numpy(), by["ctg_only+ttd"]["y_proba"].to_numpy())
        dl = delong_test(y, by["combined+ttd"]["y_proba"].to_numpy(), by["ctg_only+ttd"]["y_proba"].to_numpy())
        row.update({"delta_combined_vs_ctg_given_ttd": d["delta_auc"], "delta_given_ttd_ci_low": d["delta_ci_low"],
                    "delta_given_ttd_ci_high": d["delta_ci_high"], "delong_p_combined_vs_ctg_given_ttd": dl["p_value"]})
        cv_rows.append(row)

    firth = pd.DataFrame(firth_rows)
    firth["holm_p"] = np.nan
    for _, g in firth.groupby(["variant", "test"]):
        firth.loc[g.index, "holm_p"] = holm(g["raw_p"].tolist())
    cv = pd.DataFrame(cv_rows)
    cv["delong_holm_p_given_ttd"] = np.nan
    for _, g in cv.groupby("variant"):
        cv.loc[g.index, "delong_holm_p_given_ttd"] = holm(g["delong_p_combined_vs_ctg_given_ttd"].tolist())

    pd.DataFrame(desc).to_csv(OUT / "B_ttd_descriptive.csv", index=False)
    pd.DataFrame(aucs).round(4).to_csv(OUT / "B_ttd_auc.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(OUT / "B_second_stage_at_t.csv", index=False)
    firth.round(4).to_csv(OUT / "B_firth_adjusted_ttd.csv", index=False)
    cv.round(4).to_csv(OUT / "B_lr_auc_with_ttd.csv", index=False)
    pd.DataFrame(len_rows).round(4).to_csv(OUT / "B_recording_length_manuscript_numbers.csv", index=False)
    figure(meta, q)
    print("Task B complete.")


def figure(meta, q) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for ax, v in zip(axes, ["ckfilter", "unfiltered"]):
        data, pos, cols = [], [], []
        for k, t in enumerate(CHECKPOINTS):
            coh = members(meta, q, v, t)
            ttd = coh["length_min"] - t
            for j, (grp, m) in enumerate((("Control", coh["y"] == 0), ("Case", coh["y"] == 1))):
                data.append(ttd[m].to_numpy())
                pos.append(k * 3 + j)
                cols.append(F.C[j])
        bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True, showfliers=True,
                        medianprops={"color": F.INK, "lw": 1.2}, flierprops={"ms": 2, "mec": F.MUTED})
        for patch, c, j in zip(bp["boxes"], cols, [0, 1] * len(CHECKPOINTS)):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
            patch.set_hatch("" if j == 0 else "///")
        ax.set_xticks([k * 3 + 0.5 for k in range(len(CHECKPOINTS))])
        ax.set_xticklabels([f"{int(t)}" for t in CHECKPOINTS])
        ax.set_xlabel("Checkpoint t (min)")
        ax.set_title(F.VARIANT_LABEL[v], fontsize=9, color=F.INK)
    axes[0].set_ylabel("Time from t to delivery (min)")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=F.C[0], alpha=0.55, label="Control (pH ≥ 7.15)"),
                        Patch(facecolor=F.C[1], alpha=0.55, hatch="///", label="Case (pH < 7.05)")],
               loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Time to delivery at each checkpoint (explanatory only)", x=0.02, y=1.05, ha="left", fontsize=10)
    F.save(fig, "B1_time_to_delivery_by_checkpoint")


if __name__ == "__main__":
    main()
