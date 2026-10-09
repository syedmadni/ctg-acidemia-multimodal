"""7e: time-varying modality contribution (STEP7E_SPEC.md, frozen 2026-09-22).

For each fusion type (LR concatenation, RF concatenation, deep `fusion`, deep
`gated_fusion`) reports CTG-only, maternal-only and combined ROC-AUC / PR-AUC with 95% CI
at checkpoints 10-60, gapped label primary and consecutive label as sensitivity, plus:
  M2  combined minus CTG-only: paired stratified bootstrap CI + DeLong, Holm across 5
  M3  LR combined minus deep `fusion` combined: DeLong, Holm across 5, per encoder size
and the main figure (AUC vs checkpoint, one panel per fusion type).

Deep models (spec R1-R3): `ctg_only`/`fusion` use seed-averaged OOF (reduced seeds
42/43/44, full-size 42/43); `maternal_only`/`gated_fusion` are single-seed (42); the
consecutive deep arm is reduced-only and single-seed throughout.

SELF-VERIFICATION (runs first, aborts on failure):
  V1 classical ROC/PR point estimates and CIs vs consec_cv_results.csv (both labels, 4 dp)
  V2 single-seed deep ROC-AUC vs each cohort's cv_summary.json (4 dp)
  V3 seed-averaged OOF files vs an independent recomputation from the per-seed cohorts
  V4 LR gapped M2 bootstrap CI vs step3_bootstrap_auc_diff.csv (4 dp)

Usage: python -m src.analysis.run_7e_modality_contribution
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

from src.analysis.run_consecutive_label_sensitivity import (  # noqa: E402
    bootstrap_ci,
    durations_from_record_ids,
    run_cv_grid,
)
from src.analysis.significance import delong_test  # noqa: E402
from src.training.train_cv import (  # noqa: E402
    aggregate_repeated_oof_anytime,
    load_cohort_arrays,
    precompute_quality_splits,
)

CHECKPOINTS = [10.0, 20.0, 30.0, 45.0, 60.0]
SEED = 42
DIFF_BOOT_ITERS = 2000
DEC = 4
BASE = Path("outputs/quality_cohorts")
OUT_DIR = BASE / "primary_705_anytime_vr060" / "maternal_value"
FIG_DIR = Path("outputs/report_figures")
CONSEC_RESULTS = BASE / "primary_705_consec_vr060" / "consec_cv_results.csv"
STEP3 = OUT_DIR / "step3_bootstrap_auc_diff.csv"
SUFFIX = {"gapped": "_705_anytime_vr060", "consecutive": "_705_consec_vr060"}
CLASSICAL_CACHE = {  # record-level, git-ignored
    "gapped": OUT_DIR / "step7e_classical_gapped_oof.csv",
    "consecutive": OUT_DIR / "step7e_classical_consec_oof.csv",
}

SEED_COHORTS = {
    "full": {42: "step7d_full_gapped", 43: "step7d_full_gapped_seed43"},
    "cap64": {42: "step7d_cap64_gapped", 43: "step7d_cap64_gapped_seed43", 44: "step7d_cap64_gapped_seed44"},
}
SIZE_LABEL = {"full": "full-size", "cap64": "reduced"}
FS_CLASSICAL = {"ctg": "ctg_only_features", "mat": "maternal_only_features", "comb": "both"}


def read_csv(path):
    """Lossless float parsing. pandas' default C parser can be 1 ulp off, which splits
    RF vote-fraction ties and moves RF AUC by up to 0.0003 (STEP7E_SPEC Amendment 1)."""
    return pd.read_csv(path, float_precision="round_trip")


# ---------------------------------------------------------------------------
# Loading: every prediction source becomes {checkpoint: DataFrame(record_id, y_true, y_proba)}
# ---------------------------------------------------------------------------

def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["record_id"] = df["record_id"].astype(int)
    return df[df["checkpoint_minutes"].isin(CHECKPOINTS)]


def repeat_averaged(cohort: str) -> pd.DataFrame:
    return _norm(aggregate_repeated_oof_anytime(read_csv(BASE / cohort / "oof_predictions.csv")))


def seed_averaged_independent(size: str) -> pd.DataFrame:
    """V3: recompute the seed average without the shared aggregation helper."""
    frames = []
    for cohort in SEED_COHORTS[size].values():
        o = read_csv(BASE / cohort / "oof_predictions.csv")
        frames.append(o[o.model.isin(["ctg_only", "fusion"])])
    o = _norm(pd.concat(frames, ignore_index=True))
    return (o.groupby(["model", "checkpoint_minutes", "record_id"], as_index=False)
             .agg(y_true=("y_true", "first"), y_proba=("y_proba", "mean"), n=("y_proba", "size")))


def classical_oof(label: str) -> pd.DataFrame:
    """Both labels come from the 7b code path (run_cv_grid), the one behind the published
    consec_cv_results.csv (STEP7E_SPEC Amendment 1). Cached to CSV and read back
    losslessly via read_csv()."""
    cache = CLASSICAL_CACHE[label]
    if cache.exists():
        return read_csv(cache)
    suffix = SUFFIX[label]
    X_ctg, X_maternal, y, record_ids, _ = load_cohort_arrays(Path("data/processed"), suffix)
    splits = precompute_quality_splits(y, 5, 3, SEED, 0.2)
    print(f"Regenerating {label} classical OOF via the 7b code path (run_cv_grid) ...")
    _, avg = run_cv_grid(label, X_ctg, X_maternal, y, record_ids, splits,
                         durations_from_record_ids(suffix))
    avg = avg.drop(columns=["duration"])
    avg.to_csv(cache, index=False)
    return avg


def build_sources():
    """Returns {source_key: {'df': long df, 'label', 'desc'}}."""
    src = {}

    def add(key, df, label, desc):
        src[key] = {"df": _norm(df), "label": label, "desc": desc}

    for label in ("gapped", "consecutive"):
        df = classical_oof(label)
        for model, tag in (("logistic_regression", "LR"), ("random_forest", "RF")):
            for arm, fs in FS_CLASSICAL.items():
                add(f"{label}|{tag}|{arm}", df[(df.model == model) & (df.cohort_name == fs)], label,
                    f"{tag} {fs}")

    for size in ("full", "cap64"):
        sa = read_csv(OUT_DIR / f"step7d_seed_averaged_oof_{size}.csv")
        seeds = "/".join(str(s) for s in SEED_COHORTS[size])
        add(f"gapped|deep_{size}|ctg", sa[sa.model == "ctg_only"], "gapped", f"ctg_only seed-avg {seeds}")
        add(f"gapped|deep_{size}|fusion", sa[sa.model == "fusion"], "gapped", f"fusion seed-avg {seeds}")
        s42 = repeat_averaged(SEED_COHORTS[size][42])
        add(f"gapped|deep_{size}|mat", s42[s42.model == "maternal_only"], "gapped", "maternal_only single-seed 42")
        add(f"gapped|deep_{size}|gated", s42[s42.model == "gated_fusion"], "gapped", "gated_fusion single-seed 42")

    c = repeat_averaged("step7d_cap64_consec")
    for arm, model in (("ctg", "ctg_only"), ("mat", "maternal_only"), ("fusion", "fusion"), ("gated", "gated_fusion")):
        add(f"consecutive|deep_cap64|{arm}", c[c.model == model], "consecutive", f"{model} single-seed 42")
    return src


def panels():
    """(label, fusion_type, size, ctg_key, mat_key, comb_key, seed notes)."""
    out = []
    for label in ("gapped", "consecutive"):
        for tag, name in (("LR", "LR concatenation"), ("RF", "RF concatenation")):
            out.append(dict(label=label, fusion_type=name, size="n/a",
                            ctg=f"{label}|{tag}|ctg", mat=f"{label}|{tag}|mat", comb=f"{label}|{tag}|comb",
                            seeds="deterministic"))
        sizes = ("full", "cap64") if label == "gapped" else ("cap64",)
        for size in sizes:
            d = f"{label}|deep_{size}"
            multi = label == "gapped"
            out.append(dict(label=label, fusion_type="deep fusion", size=SIZE_LABEL[size],
                            ctg=f"{d}|ctg", mat=f"{d}|mat", comb=f"{d}|fusion",
                            seeds=("ctg/comb seed-avg; mat single-seed" if multi else "all single-seed")))
            out.append(dict(label=label, fusion_type="deep gated fusion", size=SIZE_LABEL[size],
                            ctg=f"{d}|ctg", mat=f"{d}|mat", comb=f"{d}|gated",
                            seeds=("ctg seed-avg; mat and comb single-seed" if multi else "all single-seed")))
    return out


def arrays(src, key, ck, ref_ids=None):
    g = src[key]["df"]
    g = g[g.checkpoint_minutes == ck].sort_values("record_id").reset_index(drop=True)
    assert g["record_id"].is_unique, key
    if ref_ids is not None:
        assert (g["record_id"].to_numpy() == ref_ids).all(), f"record mismatch {key} @ {ck}"
    return g["record_id"].to_numpy(), g["y_true"].to_numpy().astype(int), g["y_proba"].to_numpy().astype(float)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def holm(p):
    return multipletests(p, method="holm")[1]


def paired_diff_bootstrap(rng, yt, pa, pb):
    """Step 3 scheme: stratified resampling of positives and negatives separately."""
    pos, neg = np.where(yt == 1)[0], np.where(yt == 0)[0]
    d = np.empty(DIFF_BOOT_ITERS)
    for i in range(DIFF_BOOT_ITERS):
        s = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        d[i] = roc_auc_score(yt[s], pa[s]) - roc_auc_score(yt[s], pb[s])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi), float(np.mean(d > 0))


def compare(src, key_a, key_b, family: dict) -> pd.DataFrame:
    """a minus b at each checkpoint; one rng per family, consumed 10 -> 60 (spec M2)."""
    rng = np.random.default_rng(SEED)
    rows = []
    for ck in CHECKPOINTS:
        ids, yt, pa = arrays(src, key_a, ck)
        _, yt_b, pb = arrays(src, key_b, ck, ref_ids=ids)
        assert (yt == yt_b).all()
        lo, hi, pgt0 = paired_diff_bootstrap(rng, yt, pa, pb)
        dl = delong_test(yt, pa, pb)
        rows.append({**family, "checkpoint_minutes": ck, "n": len(yt), "positives": int(yt.sum()),
                     "auc_a": dl["auc_a"], "auc_b": dl["auc_b"], "auc_diff": dl["auc_diff"],
                     "boot_ci_low": lo, "boot_ci_high": hi, "prop_boot_diff_gt_0": pgt0,
                     "delong_z": dl["z"], "delong_p": dl["p_value"],
                     "pr_auc_diff": average_precision_score(yt, pa) - average_precision_score(yt, pb)})
    df = pd.DataFrame(rows)
    df["delong_holm_p"] = holm(df["delong_p"].to_numpy())
    df["holm_significant"] = df["delong_holm_p"] < 0.05
    return df


# ---------------------------------------------------------------------------
# Self-verification
# ---------------------------------------------------------------------------

def verify(src, m1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pub = read_csv(CONSEC_RESULTS)
    for _, r in m1[m1.fusion_type.isin(["LR concatenation", "RF concatenation"])].iterrows():
        model = "logistic_regression" if r.fusion_type.startswith("LR") else "random_forest"
        p = pub[(pub.label_design == r.label) & (pub.checkpoint_minutes == r.checkpoint_minutes)
                & (pub.feature_set == FS_CLASSICAL[r.arm]) & (pub.model == model)]
        for col in ("roc_auc", "roc_ci_low", "roc_ci_high", "pr_auc", "pr_ci_low", "pr_ci_high"):
            rows.append({"check": "V1 classical vs consec_cv_results", "item":
                         f"{r.label} {model} {r.arm} {r.checkpoint_minutes:.0f} {col}",
                         "mine": round(r[col], DEC), "reference": float(p.iloc[0][col]),
                         "match": round(r[col], DEC) == round(float(p.iloc[0][col]), DEC)})

    for cohort, label in (("step7d_full_gapped", "gapped"), ("step7d_cap64_gapped", "gapped"),
                          ("step7d_cap64_consec", "consecutive")):
        met = json.load((BASE / cohort / "cv_summary.json").open())["model_metrics_from_averaged_oof"]
        size = "cap64" if "cap64" in cohort else "full"
        models = ["maternal_only", "gated_fusion"] if label == "gapped" else \
            ["ctg_only", "maternal_only", "fusion", "gated_fusion"]
        arm_of = {"ctg_only": "ctg", "maternal_only": "mat", "fusion": "fusion", "gated_fusion": "gated"}
        for model in models:
            for ck in CHECKPOINTS:
                _, yt, yp = arrays(src, f"{label}|deep_{size}|{arm_of[model]}", ck)
                ref = next(float(met[k]["roc_auc"]) for k in (f"{model}@{ck}min", f"{model}@{int(ck)}min") if k in met)
                mine = roc_auc_score(yt, yp)
                rows.append({"check": "V2 single-seed deep vs cv_summary", "item": f"{cohort} {model} {ck:.0f}",
                             "mine": round(mine, DEC), "reference": round(ref, DEC),
                             "match": round(mine, DEC) == round(ref, DEC)})

    for size in ("full", "cap64"):
        ind = seed_averaged_independent(size)
        n_seeds = len(SEED_COHORTS[size])
        ok_n = bool((ind["n"] == 3 * n_seeds).all())
        for model, arm in (("ctg_only", "ctg"), ("fusion", "fusion")):
            a = src[f"gapped|deep_{size}|{arm}"]["df"].sort_values(["checkpoint_minutes", "record_id"])
            b = ind[ind.model == model].sort_values(["checkpoint_minutes", "record_id"])
            maxdiff = float(np.max(np.abs(a["y_proba"].to_numpy() - b["y_proba"].to_numpy())))
            rows.append({"check": "V3 seed-avg file vs independent recompute", "item": f"{size} {model}",
                         "mine": maxdiff, "reference": 0.0,
                         "match": ok_n and len(a) == len(b) == 5 * 345 and maxdiff < 1e-12})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

STYLE = {  # colour + line style + marker, so each arm reads in greyscale
    "ctg": dict(color="#2a78d6", ls="--", marker="s", name="CTG-only"),
    "mat": dict(color="#eb6834", ls=":", marker="^", name="Maternal-only"),
    "comb": dict(color="#1baf7a", ls="-", marker="o", name="Combined"),
}


def figure(m1: pd.DataFrame, label: str, size: str, path: Path, title: str):
    sel = [("LR concatenation", "n/a"), ("RF concatenation", "n/a"),
           ("deep fusion", size), ("deep gated fusion", size)]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2), sharex=True, sharey=True)
    for ax, (ft, sz) in zip(axes.ravel(), sel):
        p = m1[(m1.label == label) & (m1.fusion_type == ft) & (m1["size"] == sz)]
        for arm in ("ctg", "mat", "comb"):
            s = p[p.arm == arm].sort_values("checkpoint_minutes")
            st = STYLE[arm]
            ax.fill_between(s.checkpoint_minutes, s.roc_ci_low, s.roc_ci_high, color=st["color"],
                            alpha=0.12, linewidth=0)
            single = bool(s["single_seed"].iloc[0]) and ft.startswith("deep")
            ax.plot(s.checkpoint_minutes, s.roc_auc, color=st["color"], ls=st["ls"], lw=2,
                    marker=st["marker"], ms=7, markeredgecolor="white", markeredgewidth=1,
                    label=st["name"] + (" (single-seed)" if single else ""))
        ax.axhline(0.5, color="#9a9a94", lw=1, ls=(0, (1, 3)), zorder=0)
        panel = ft[0].upper() + ft[1:] + (f" ({sz} encoder)" if sz != "n/a" else "")
        ax.set_title(panel, fontsize=11, loc="left")
        ax.set_xticks(CHECKPOINTS)
        ax.set_ylim(0.25, 1.0)
        ax.grid(axis="y", color="#e6e6e1", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    for ax in axes[1]:
        ax.set_xlabel("Elapsed CTG recording (min)")
    for ax in axes[:, 0]:
        ax.set_ylabel("ROC-AUC (95% CI)")
    fig.suptitle(title, fontsize=12, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    src = build_sources()

    # ---- M1: per-arm ROC/PR with CI ----
    rows = []
    for pn in panels():
        for arm in ("ctg", "mat", "comb"):
            key = pn[arm]
            single = "single-seed" in src[key]["desc"]
            for ck in CHECKPOINTS:
                _, yt, yp = arrays(src, key, ck)
                rl, rh, pl, ph = bootstrap_ci(yt, yp)
                rows.append({"label": pn["label"], "fusion_type": pn["fusion_type"], "size": pn["size"],
                             "arm": arm, "source": src[key]["desc"], "single_seed": single,
                             "checkpoint_minutes": ck, "n": len(yt), "positives": int(yt.sum()),
                             "roc_auc": roc_auc_score(yt, yp), "roc_ci_low": rl, "roc_ci_high": rh,
                             "pr_auc": average_precision_score(yt, yp), "pr_ci_low": pl, "pr_ci_high": ph})
    m1 = pd.DataFrame(rows)

    ver = verify(src, m1)

    # ---- M2: combined minus CTG-only ----
    m2 = pd.concat([compare(src, pn["comb"], pn["ctg"],
                            {"label": pn["label"], "fusion_type": pn["fusion_type"], "size": pn["size"],
                             "seeds": pn["seeds"], "comparison": "combined minus CTG-only"})
                    for pn in panels()], ignore_index=True)

    # V4: LR gapped must reproduce Step 3's bootstrap
    s3 = read_csv(STEP3)
    lr = m2[(m2.label == "gapped") & (m2.fusion_type == "LR concatenation")]
    for _, r in lr.iterrows():
        ref = s3[s3.checkpoint_minutes == r.checkpoint_minutes].iloc[0]
        for mine, col in ((r.auc_diff, "observed_diff"), (r.boot_ci_low, "percentile_ci_low"),
                          (r.boot_ci_high, "percentile_ci_high"), (r.prop_boot_diff_gt_0, "proportion_boot_diff_gt_0")):
            ver.loc[len(ver)] = {"check": "V4 LR gapped M2 vs step3", "item": f"{r.checkpoint_minutes:.0f} {col}",
                                 "mine": round(mine, DEC), "reference": round(float(ref[col]), DEC),
                                 "match": round(mine, DEC) == round(float(ref[col]), DEC)}

    ver.to_csv(OUT_DIR / "step7e_self_verification.csv", index=False)
    print("Self-verification:")
    print(ver.groupby("check")["match"].agg(["sum", "count"]).to_string())
    if not ver["match"].all():
        print(ver[~ver.match].to_string(index=False))
        raise SystemExit("ABORT: self-verification failed; 7e results not written.")

    # ---- M3: LR combined minus deep fusion combined ----
    m3_parts = []
    for label, sizes in (("gapped", ("full", "cap64")), ("consecutive", ("cap64",))):
        for size in sizes:
            m3_parts.append(compare(src, f"{label}|LR|comb", f"{label}|deep_{size}|fusion",
                                    {"label": label, "fusion_type": "LR combined vs deep fusion combined",
                                     "size": SIZE_LABEL[size],
                                     "seeds": "deep fusion seed-avg" if label == "gapped" else "deep fusion single-seed",
                                     "comparison": "LR combined minus deep fusion combined"}))
    m3 = pd.concat(m3_parts, ignore_index=True)

    num = lambda d: d.round({c: DEC for c in d.select_dtypes("float").columns})  # noqa: E731
    num(m1).to_csv(OUT_DIR / "step7e_modality_auc.csv", index=False)
    num(m2).to_csv(OUT_DIR / "step7e_combined_minus_ctg.csv", index=False)
    num(m3).to_csv(OUT_DIR / "step7e_lr_vs_deep_fusion.csv", index=False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure(m1, "gapped", "full-size", FIG_DIR / "fig_7e_modality_contribution.png",
           "Modality contribution by elapsed time (gapped label; deep panels: full-size encoder)")
    figure(m1, "gapped", "reduced", FIG_DIR / "fig_7e_modality_contribution_reduced_supp.png",
           "Supplementary: gapped label; deep panels: reduced encoder")
    figure(m1, "consecutive", "reduced", FIG_DIR / "fig_7e_modality_contribution_consecutive_supp.png",
           "Supplementary: consecutive label; deep panels: reduced encoder, single-seed")

    pd.set_option("display.width", 220)
    print("\nM2 combined minus CTG-only:")
    print(num(m2)[["label", "fusion_type", "size", "checkpoint_minutes", "auc_a", "auc_b", "auc_diff",
                   "boot_ci_low", "boot_ci_high", "delong_p", "delong_holm_p"]].to_string(index=False))
    print("\nM3 LR combined minus deep fusion combined:")
    print(num(m3)[["label", "size", "checkpoint_minutes", "auc_a", "auc_b", "auc_diff", "boot_ci_low",
                   "boot_ci_high", "delong_p", "delong_holm_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
