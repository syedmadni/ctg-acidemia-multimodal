"""7f (deep part): deep-model tables and the final paper figures.

Adds the retrained (7d, duration-aware) deep models to the fixed tables and rebuilds every
results figure the manuscript uses from fixed/retrained results. Classical tables from
build_fixed_tables.py are NOT modified; this script writes new TF8-TF12 files into the same
directory with its own README_DEEP.md, and figures into outputs/report_figures/final/.
Nothing under paper/ is touched (CLAUDE.md standing rule 1): FIGURES.md maps each final
figure to the paper figure it supersedes.

Deep-model conventions (STEP7E_SPEC R1-R3): ctg_only and fusion = seed-averaged OOF
(full-size seeds 42/43, reduced 42/43/44); maternal_only and gated_fusion = single-seed 42;
consecutive-label deep = reduced encoder, single-seed. Every row carries a `seeds` column.

New tests (TF9 only): classical combined (LR, RF) vs retrained deep ctg_only, paired DeLong
+ stratified bootstrap CI, Holm across 5 checkpoints per (comparison, size, label). This is
the supersession TF6_classical_vs_deep_PROVISIONAL announced (its deep side was the pre-7d
single-seed model). The LR-vs-deep-fusion rows are 7e's M3, recomputed and self-checked.

Self-checks (abort on failure):
  S1 TF8 values equal step7e_modality_auc.csv (4 dp)
  S2 recomputed LR-vs-fusion rows equal step7e_lr_vs_deep_fusion.csv (4 dp)

Usage: python -m src.reporting.build_deep_tables_and_figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from src.analysis.run_7e_modality_contribution import (  # noqa: E402
    CHECKPOINTS,
    STYLE as ARM_STYLE,
    arrays,
    build_sources,
    compare,
    figure as modality_figure,
)
from src.reporting.build_fixed_tables import md  # noqa: E402

QC = Path("outputs/quality_cohorts")
MV = QC / "primary_705_anytime_vr060" / "maternal_value"
TAB = Path("outputs/report_figures/tables_fixed")
FIG = Path("outputs/report_figures/final")
DEC = 4


def rc(path):
    return pd.read_csv(path, float_precision="round_trip")


def ci(v, lo, hi, d=3):
    return f"{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


# Model catalogue: display name -> how to find it in step7e_modality_auc.csv
MODELS = [
    # name, fusion_type, arm, deep?
    ("LR, CTG + maternal", "LR concatenation", "comb", False),
    ("LR, CTG only", "LR concatenation", "ctg", False),
    ("LR, maternal only", "LR concatenation", "mat", False),
    ("RF, CTG + maternal", "RF concatenation", "comb", False),
    ("RF, CTG only", "RF concatenation", "ctg", False),
    ("RF, maternal only", "RF concatenation", "mat", False),
    ("deep ctg_only", "deep fusion", "ctg", True),
    ("deep fusion", "deep fusion", "comb", True),
    ("deep gated_fusion", "deep gated fusion", "comb", True),
    ("deep maternal_only", "deep fusion", "mat", True),
]

# figure series: fixed categorical order (dataviz reference slots 1-6) + line style + marker
SERIES = [
    ("LR, CTG + maternal", "#2a78d6", "-", "o"),
    ("RF, CTG + maternal", "#eb6834", "--", "s"),
    ("deep ctg_only", "#1baf7a", "-.", "^"),
    ("deep fusion", "#eda100", "-", "D"),
    ("deep gated_fusion", "#e87ba4", ":", "v"),
    ("deep maternal_only", "#008300", (0, (8, 2, 2, 2, 2, 2)), "X"),
]


def seeds_of(label, name, size):
    if not name.startswith("deep"):
        return "deterministic"
    if label == "consecutive":
        return "single-seed 42"
    if name in ("deep ctg_only", "deep fusion"):
        return "seed-avg 42/43" if size == "full-size" else "seed-avg 42/43/44"
    return "single-seed 42"


# ---------------------------------------------------------------- TF8
def tf8(m1):
    rows = []
    for label in ("gapped", "consecutive"):
        for name, ft, arm, deep in MODELS:
            for size in (["full-size", "reduced"] if deep and label == "gapped" else ["reduced"] if deep else ["n/a"]):
                sel = m1[(m1.label == label) & (m1.fusion_type == ft) & (m1.arm == arm) & (m1["size"] == size)]
                assert len(sel) == 5, (label, name, size, len(sel))
                for _, r in sel.iterrows():
                    rows.append({"label": label, "model": name, "encoder": size, "seeds": seeds_of(label, name, size),
                                 "checkpoint_minutes": r.checkpoint_minutes, "n": r.n, "positives": r.positives,
                                 "roc_auc": r.roc_auc, "roc_ci_low": r.roc_ci_low, "roc_ci_high": r.roc_ci_high,
                                 "pr_auc": r.pr_auc, "pr_ci_low": r.pr_ci_low, "pr_ci_high": r.pr_ci_high})
    df = pd.DataFrame(rows)
    return df


def pivot_display(df, label, metric):
    d = df[df.label == label].copy()
    d["cell"] = [ci(r[metric], r[f"{metric.split('_')[0]}_ci_low"], r[f"{metric.split('_')[0]}_ci_high"])
                 for _, r in d.iterrows()]
    d["row"] = d.model + np.where(d.encoder == "n/a", "", " (" + d.encoder + ")") + " — " + d.seeds
    p = d.pivot_table(index="row", columns="checkpoint_minutes", values="cell", aggfunc="first", sort=False)
    p.columns = [f"{int(c)} min" for c in p.columns]
    return p.reset_index().rename(columns={"row": "model (encoder) — seeds"})


# ---------------------------------------------------------------- TF9
def tf9(src):
    fams = []
    spec = [("gapped", "full", "full-size"), ("gapped", "cap64", "reduced"), ("consecutive", "cap64", "reduced")]
    for label, sz, szl in spec:
        seed_note = "deep seed-avg" if label == "gapped" else "deep single-seed"
        for tag, cname in (("LR", "LR combined"), ("RF", "RF combined")):
            fams.append(compare(src, f"{label}|{tag}|comb", f"{label}|deep_{sz}|ctg",
                                {"label": label, "size": szl, "seeds": seed_note,
                                 "comparison": f"{cname} minus deep ctg_only"}))
        fams.append(compare(src, f"{label}|LR|comb", f"{label}|deep_{sz}|fusion",
                            {"label": label, "size": szl, "seeds": seed_note,
                             "comparison": "LR combined minus deep fusion combined"}))
    return pd.concat(fams, ignore_index=True)


# ---------------------------------------------------------------- figures
def _style(ax):
    ax.grid(axis="y", color="#e6e6e1", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xticks(CHECKPOINTS)
    ax.set_xlim(5, 65)


def fig_by_checkpoint(tf8df, metric, path, ylabel, floor=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, size in zip(axes, ("full-size", "reduced")):
        g = tf8df[tf8df.label == "gapped"]
        for name, col, ls, mk in SERIES:
            s = g[(g.model == name) & (g.encoder.isin(["n/a", size]))].sort_values("checkpoint_minutes")
            lab = name + (" (single-seed)" if "single" in s.seeds.iloc[0] else "")
            ax.plot(s.checkpoint_minutes, s[metric], color=col, ls=ls, lw=2, marker=mk, ms=7,
                    markeredgecolor="white", markeredgewidth=1, label=lab)
        if floor is not None:
            ax.axhline(floor, color="#9a9a94", lw=1, ls=(0, (1, 3)), zorder=0)
            ax.text(CHECKPOINTS[-1], floor - 0.002, "no-skill (prevalence)", va="top", ha="right", fontsize=8, color="#6b6b66")
        ax.set_title(f"Deep models: {size} encoder", loc="left", fontsize=11)
        ax.set_xlabel("Elapsed CTG recording (min)")
        _style(ax)
    axes[0].set_ylabel(ylabel)
    h, lbl = axes[0].get_legend_handles_labels()
    fig.legend(h, lbl, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def fig_forest_60(tf8df, path):
    g = tf8df[(tf8df.label == "gapped") & (tf8df.checkpoint_minutes == 60.0)].copy()
    order = [("LR, CTG + maternal", "n/a"), ("LR, CTG only", "n/a"), ("LR, maternal only", "n/a"),
             ("RF, CTG + maternal", "n/a"), ("RF, CTG only", "n/a"), ("RF, maternal only", "n/a"),
             ("deep ctg_only", "full-size"), ("deep ctg_only", "reduced"),
             ("deep fusion", "full-size"), ("deep fusion", "reduced"),
             ("deep gated_fusion", "full-size"), ("deep gated_fusion", "reduced"),
             ("deep maternal_only", "full-size"), ("deep maternal_only", "reduced")]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for i, (name, size) in enumerate(order):
        r = g[(g.model == name) & (g.encoder == size)].iloc[0]
        y = len(order) - 1 - i
        deep = name.startswith("deep")
        ax.plot([r.roc_ci_low, r.roc_ci_high], [y, y], color="#6b6b66", lw=2, solid_capstyle="round")
        ax.plot(r.roc_auc, y, marker="s" if deep else "o", ms=8, color="#2a78d6" if not deep else "#1baf7a",
                markeredgecolor="white", markeredgewidth=1)
        ax.text(1.005, y, f"{r.roc_auc:.3f} [{r.roc_ci_low:.3f}, {r.roc_ci_high:.3f}]", va="center", fontsize=8,
                transform=ax.get_yaxis_transform())
    labels = [f"{n}" + ("" if s == "n/a" else f" ({s}{', single-seed' if 'single' in seeds_of('gapped', n, s) else ''})")
              for n, s in order]
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.axvline(0.5, color="#9a9a94", lw=1, ls=(0, (1, 3)))
    ax.set_xlim(0.35, 1.0)
    ax.set_xlabel("ROC-AUC at 60 min (95% CI), gapped label, n=345 / 23 positives")
    ax.grid(axis="x", color="#e6e6e1", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_roc_60(src, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    keymap = {"LR, CTG + maternal": "gapped|LR|comb", "RF, CTG + maternal": "gapped|RF|comb",
              "deep ctg_only": "gapped|deep_{sz}|ctg", "deep fusion": "gapped|deep_{sz}|fusion",
              "deep gated_fusion": "gapped|deep_{sz}|gated", "deep maternal_only": "gapped|deep_{sz}|mat"}
    for ax, (sz, szl) in zip(axes, (("full", "full-size"), ("cap64", "reduced"))):
        for name, col, ls, mk in SERIES:
            _, yt, yp = arrays(src, keymap[name].format(sz=sz), 60.0)
            fpr, tpr, _ = roc_curve(yt, yp)
            lab = name + (" (single-seed)" if "single" in seeds_of("gapped", name, szl) else "")
            ax.plot(fpr, tpr, color=col, ls=ls, lw=2, label=lab)
        ax.plot([0, 1], [0, 1], color="#9a9a94", lw=1, ls=(0, (1, 3)))
        ax.set_title(f"60 min, deep models: {szl} encoder", loc="left", fontsize=11)
        ax.set_xlabel("1 − specificity")
        ax.set_aspect("equal")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Sensitivity")
    h, lbl = axes[0].get_legend_handles_labels()
    fig.legend(h, lbl, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def fig_gate(unit, gsum, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, size in zip(axes, ("full-size", "reduced")):
        s = gsum[gsum["size"] == size].sort_values("checkpoint_minutes")
        u = unit[unit["size"] == size].sort_values("checkpoint_minutes")
        ax.plot(s.checkpoint_minutes, s.gate_mean_when_ctg_right, color=ARM_STYLE["ctg"]["color"], ls="--",
                marker="s", lw=2, ms=7, markeredgecolor="white", label="CTG was right")
        ax.plot(s.checkpoint_minutes, s.gate_mean_when_maternal_right, color=ARM_STYLE["mat"]["color"], ls=":",
                marker="^", lw=2, ms=7, markeredgecolor="white", label="Maternal was right")
        for _, r in u.iterrows():
            star = "*" if r.A_patient_mwu_holm_p < 0.05 else ""
            ax.annotate(f"+{r.gap_patient_level:.3f}{star}", (r.checkpoint_minutes, 0.515), ha="center", fontsize=8,
                        color="#4a4a46")
        ax.set_title(f"Gated fusion ({size} encoder, single-seed)", loc="left", fontsize=11)
        ax.set_xlabel("Elapsed CTG recording (min)")
        ax.set_ylim(0.50, 0.62)
        _style(ax)
    axes[0].set_ylabel("Mean gate (CTG weight), disagreement cases")
    h, lbl = axes[0].get_legend_handles_labels()
    fig.legend(h, lbl, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.text(0.5, 0.005, "Numbers: gap (CTG-right minus maternal-right), patient as unit; * = Holm-corrected "
             "Mann-Whitney p < 0.05", ha="center", fontsize=8, color="#4a4a46")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    m1 = rc(MV / "step7e_modality_auc.csv")
    m1["size"] = m1["size"].fillna("n/a")
    src = build_sources()
    manifest = []

    def write(df, name, note, sources, displays=None):
        TAB.mkdir(parents=True, exist_ok=True)
        df.to_csv(TAB / f"{name}.csv", index=False)
        body = f"# {name}\n\n{note}\n\nSources: " + ", ".join(f"`{s}`" for s in sources) + "\n\n"
        for title, d in (displays or [("", df)]):
            body += (f"## {title}\n\n" if title else "") + md(d) + "\n"
        (TAB / f"{name}.md").write_text(body, encoding="utf-8")
        manifest.append({"table": name, "rows": len(df), "sources": "; ".join(sources), "note": note})
        print(f"  wrote {name} ({len(df)} rows)")

    # TF8
    t8 = tf8(m1)
    # S1
    s1 = t8.merge(m1, left_on=["label", "checkpoint_minutes", "roc_auc", "pr_auc"],
                  right_on=["label", "checkpoint_minutes", "roc_auc", "pr_auc"], how="left", indicator=True)
    assert (s1["_merge"] == "both").all(), "S1 failed: TF8 rows not found in step7e_modality_auc.csv"
    print(f"S1 passed: {len(t8)} TF8 rows found verbatim in step7e_modality_auc.csv")
    write(t8.round(DEC), "TF8_all_models_discrimination",
          "All models, fixed CTG features (classical) and 7d retrained deep models (duration-aware cutoffs), "
          "risk-set scoring, checkpoints 10-60. ROC/PR-AUC with 1000-iteration bootstrap 95% CI on repeat-averaged "
          "(and, where stated, seed-averaged) OOF. `seeds` column: deep ctg_only/fusion seed-averaged (full-size "
          "42/43, reduced 42/43/44); gated_fusion and maternal_only single-seed; consecutive-label deep = reduced "
          "encoder, single-seed. Gapped label PRIMARY, consecutive = sensitivity. Supersedes T1 (old tables/) for "
          "every model. Display: 3 dp.",
          [str(MV / "step7e_modality_auc.csv")],
          [("Gapped label, ROC-AUC [95% CI]", pivot_display(t8, "gapped", "roc_auc")),
           ("Gapped label, PR-AUC [95% CI] (no-skill = prevalence 0.067)", pivot_display(t8, "gapped", "pr_auc")),
           ("Consecutive label, ROC-AUC [95% CI]", pivot_display(t8, "consecutive", "roc_auc")),
           ("Consecutive label, PR-AUC [95% CI] (no-skill = prevalence 0.058)", pivot_display(t8, "consecutive", "pr_auc"))])

    # TF9
    t9 = tf9(src)
    m3 = rc(MV / "step7e_lr_vs_deep_fusion.csv")
    mine = t9[t9.comparison == "LR combined minus deep fusion combined"].reset_index(drop=True)
    for col in ("auc_diff", "boot_ci_low", "boot_ci_high", "delong_p", "delong_holm_p"):
        assert (mine[col].round(DEC).to_numpy() == m3[col].round(DEC).to_numpy()).all(), f"S2 failed on {col}"
    print("S2 passed: LR-vs-fusion rows reproduce step7e_lr_vs_deep_fusion.csv")
    t9 = t9[["label", "size", "seeds", "comparison", "checkpoint_minutes", "n", "positives", "auc_a", "auc_b",
             "auc_diff", "boot_ci_low", "boot_ci_high", "prop_boot_diff_gt_0", "delong_p", "delong_holm_p",
             "holm_significant", "pr_auc_diff"]]
    d9 = t9.copy()
    d9["cell"] = [f"{r.auc_diff:+.3f} [{r.boot_ci_low:+.3f}, {r.boot_ci_high:+.3f}] (Holm {r.delong_holm_p:.2f})"
                  for _, r in d9.iterrows()]
    d9["row"] = d9.comparison + ", " + d9.label + ", " + d9["size"] + " (" + d9.seeds + ")"
    d9p = d9.pivot_table(index="row", columns="checkpoint_minutes", values="cell", aggfunc="first", sort=False)
    d9p.columns = [f"{int(c)} min" for c in d9p.columns]
    write(t9.round(DEC), "TF9_classical_vs_deep_FINAL",
          "Classical combined models vs retrained deep models, paired on identical records and folds. ROC-AUC "
          "difference (a minus b), 2000-iteration paired stratified bootstrap percentile CI (treats OOF as fixed, so "
          "too narrow), paired DeLong p, Holm across the 5 checkpoints within each (comparison, label, size) family. "
          "SUPERSEDES TF6_classical_vs_deep_PROVISIONAL (whose deep side was the pre-7d single-seed ctg_only). "
          "'LR combined minus deep fusion combined' = 7e M3 (STEP7E_SPEC, pre-specified). The classical-vs-deep "
          "ctg_only rows replace TF6 as TF6 itself announced; they are not in STEP7E_SPEC.",
          [str(MV / "step7e_lr_vs_deep_fusion.csv"), "7e OOF sources (run_7e_modality_contribution.build_sources)"],
          [("ROC-AUC difference [95% bootstrap CI] (Holm-adjusted DeLong p)", d9p.reset_index().rename(columns={"row": "comparison"}))])

    # TF10: modality contribution (7e M2)
    m2 = rc(MV / "step7e_combined_minus_ctg.csv")
    m2["size"] = m2["size"].fillna("n/a")
    d10 = m2.copy()
    d10["cell"] = [f"{r.auc_diff:+.3f} [{r.boot_ci_low:+.3f}, {r.boot_ci_high:+.3f}] (Holm {r.delong_holm_p:.2f})"
                   for _, r in d10.iterrows()]
    d10["row"] = d10.fusion_type + ", " + d10.label + np.where(d10["size"] == "n/a", "", ", " + d10["size"]) + \
        " (" + d10.seeds + ")"
    d10p = d10.pivot_table(index="row", columns="checkpoint_minutes", values="cell", aggfunc="first", sort=False)
    d10p.columns = [f"{int(c)} min" for c in d10p.columns]
    write(m2.round(DEC), "TF10_modality_contribution_all_fusion_types",
          "Combined minus CTG-only ROC-AUC per fusion type (7e M2, STEP7E_SPEC): paired stratified bootstrap CI "
          "(2000, seed 42), paired DeLong, Holm across 5 checkpoints within each family. Extends TF2 (classical "
          "only) to deep fusion and deep gated fusion. No family has a Holm-significant checkpoint. gated_fusion "
          "rows compare a single-seed combined model with a seed-averaged CTG-only reference (asymmetry disfavours "
          "gated_fusion).",
          [str(MV / "step7e_combined_minus_ctg.csv")],
          [("Combined minus CTG-only [95% bootstrap CI] (Holm-adjusted DeLong p)", d10p.reset_index().rename(columns={"row": "family"}))])

    # TF11: gate
    unit = rc(MV / "step7e_gate_unit_of_analysis.csv")
    gsum = rc(MV / "step7e_gate_summary.csv")
    t11 = gsum.merge(unit, on=["size", "checkpoint_minutes"])
    d11 = pd.DataFrame({
        "encoder": t11["size"], "min": t11.checkpoint_minutes.astype(int),
        "n disagree (CTG right / maternal right)": [f"{a} ({b} / {c})" for a, b, c in
                                                    zip(t11.n_disagreement, t11.n_ctg_right, t11.n_maternal_right)],
        "gate, CTG right": t11.gate_mean_when_ctg_right.round(3),
        "gate, maternal right": t11.gate_mean_when_maternal_right.round(3),
        "gap [patient bootstrap 95% CI]": [ci(g, lo, hi) for g, lo, hi in
                                           zip(t11.gap_patient_level, t11.B_boot_ci_low, t11.B_boot_ci_high)],
        "Holm p, patient MWU (PRIMARY)": t11.A_patient_mwu_holm_p.map(lambda v: f"{v:.2g}"),
        "Holm p, patient bootstrap": t11.B_boot_holm_p.map(lambda v: f"{v:.2g}"),
    })
    write(t11.round(6), "TF11_gate_interpretability",
          "Pooled gate analysis on the retrained gated_fusion (single-seed, gapped), all 15 folds, n=345. Patient is "
          "the unit: each patient's gate is averaged over its 3 CV repeats before testing (RESULTS_FACTS 19.14a). "
          "Disagreement = seed-42 ctg_only and maternal_only OOF on opposite sides of 0.5. Holm across 5 checkpoints "
          "within encoder size. Retrains reproduced the 7d gated_fusion OOF bit-exactly (both sizes). Supersedes "
          "the pre-7d gate figure/numbers.",
          [str(MV / "step7e_gate_summary.csv"), str(MV / "step7e_gate_unit_of_analysis.csv")],
          [("", d11)])

    # TF12: seed variance
    sv = rc(MV / "step7d_seed_variance_summary.csv")
    write(sv, "TF12_seed_variance",
          "ROC-AUC per training seed for the retrained deep ctg_only and fusion (gapped, identical folds, only "
          "training_seed varies), with mean and min-max. Basis for using seed-averaged OOF (STEP7E_SPEC R1). "
          "fusion-minus-ctg_only sign flips across seeds at 8 of 10 (size, checkpoint) cells (RESULTS_FACTS 18.4).",
          [str(MV / "step7d_seed_variance_summary.csv")])

    readme = ("# Deep-model tables (7f deep part)\n\n"
              "Built by `python -m src.reporting.build_deep_tables_and_figures`. Complements the classical tables in "
              "`README.md` (TF1-TF7, built by `build_fixed_tables`). TF9 supersedes TF6_classical_vs_deep_PROVISIONAL; "
              "TF8 supersedes T1 for every model.\n\n" + md(pd.DataFrame(manifest)))
    (TAB / "README_DEEP.md").write_text(readme, encoding="utf-8")

    # ---- figures
    FIG.mkdir(parents=True, exist_ok=True)
    fig_by_checkpoint(t8, "roc_auc", FIG / "F1_roc_auc_by_checkpoint.png", "ROC-AUC")
    fig_by_checkpoint(t8, "pr_auc", FIG / "F2_pr_auc_by_checkpoint.png", "PR-AUC", floor=23 / 345)
    modality_figure(m1, "gapped", "full-size", FIG / "F3_modality_contribution.png",
                    "Modality contribution by elapsed time (gapped label; deep panels: full-size encoder)")
    modality_figure(m1, "gapped", "reduced", FIG / "S3a_modality_contribution_reduced.png",
                    "Supplementary: gapped label; deep panels: reduced encoder")
    modality_figure(m1, "consecutive", "reduced", FIG / "S3b_modality_contribution_consecutive.png",
                    "Supplementary: consecutive label; deep panels: reduced encoder, single-seed")
    fig_gate(unit, gsum, FIG / "F4_gate_interpretability.png")
    fig_forest_60(t8, FIG / "F5_model_comparison_60min.png")
    fig_roc_60(src, FIG / "F6_roc_curves_60min.png")
    print("Figures written to", FIG)


if __name__ == "__main__":
    main()
