"""Revision figures: 300 dpi, each saved as a separate PNG and PDF in results/revision/figures/.

Palette: categorical slots 1-3 of the dataviz reference palette (blue, orange, aqua), fixed
order, never cycled. Every series also has its own marker and line style, so identity never
relies on colour alone (print / colour-vision deficiency).

    python -m src.revision.figures A      # Task A figures
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("results/revision")
FIG = OUT / "figures"
C = ["#2a78d6", "#eb6834", "#1baf7a"]
MARK = ["o", "s", "^"]
LS = ["-", "--", ":"]
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e4e4e4"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
})

VARIANT_LABEL = {"ckfilter": "Checkpoint quality filter (primary)", "unfiltered": "Unfiltered",
                 "primary345_causal": "Published 345 records"}
FS_LABEL = {"ctg_only": "CTG only", "combined": "CTG + maternal", "maternal_only": "Maternal only"}
MODEL_LABEL = {"logistic_regression": "Logistic regression", "random_forest": "Random forest",
               "deep_ctg_only": "Deep CTG only", "deep_fusion": "Deep fusion"}


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_a_cohorts() -> None:
    s = pd.read_csv(OUT / "A_cohort_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for i, v in enumerate(["ckfilter", "unfiltered", "primary345_causal"]):
        g = s[s.variant == v]
        for ax, col in zip(axes, ["n", "cases"]):
            ax.plot(g.checkpoint_minutes, g[col], color=C[i], marker=MARK[i], ls=LS[i], lw=2, ms=6,
                    label=VARIANT_LABEL[v])
    axes[0].set_ylabel("Records included")
    axes[1].set_ylabel("Cases (pH < 7.05)")
    for ax in axes:
        ax.set_xlabel("Elapsed recording time, t (min)")
        ax.set_xticks([10, 20, 30, 45, 60])
    fig.subplots_adjust(wspace=0.35)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.16))
    fig.suptitle("Cohort size by checkpoint", x=0.02, ha="left", fontsize=10, color=INK)
    save(fig, "A1_cohort_size_by_checkpoint")


def fig_a_auc() -> None:
    d = pd.read_csv(OUT / "A_discrimination.csv")
    variants = ["ckfilter", "unfiltered", "primary345_causal"]
    models = ["logistic_regression", "random_forest"]
    fig, axes = plt.subplots(len(models), len(variants), figsize=(8.6, 5.4), sharey=True, sharex=True)
    for r, m in enumerate(models):
        for c, v in enumerate(variants):
            ax = axes[r, c]
            for i, fs in enumerate(["ctg_only", "combined", "maternal_only"]):
                g = d[(d.variant == v) & (d.model == m) & (d.feature_set == fs)].sort_values("checkpoint_minutes")
                x = g.checkpoint_minutes + (i - 1) * 1.2
                ax.errorbar(x, g.roc_auc, yerr=[g.roc_auc - g.roc_auc_ci_low, g.roc_auc_ci_high - g.roc_auc],
                            color=C[i], marker=MARK[i], ls=LS[i], lw=1.5, ms=5, capsize=0, elinewidth=1,
                            label=FS_LABEL[fs])
            ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
            ax.set_xticks([10, 20, 30, 45, 60])
            if r == 0:
                ax.set_title(VARIANT_LABEL[v], fontsize=9, color=INK)
            if c == 0:
                ax.set_ylabel(f"{MODEL_LABEL[m]}\nROC-AUC (95% CI)")
            if r == 1:
                ax.set_xlabel("t (min)")
    axes[0, 0].legend(loc="lower right", fontsize=7)
    save(fig, "A2_auc_by_checkpoint_classical")


def fig_a_deep() -> None:
    p = OUT / "A_deep_discrimination.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    variants = ["ckfilter", "unfiltered", "primary345_causal"]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.8), sharey=True)
    for c, v in enumerate(variants):
        ax = axes[c]
        for i, m in enumerate(["deep_ctg_only", "deep_fusion"]):
            g = d[(d.variant == v) & (d.model == m)].sort_values("checkpoint_minutes")
            x = g.checkpoint_minutes + (i - 0.5) * 1.2
            ax.errorbar(x, g.roc_auc, yerr=[g.roc_auc - g.roc_auc_ci_low, g.roc_auc_ci_high - g.roc_auc],
                        color=C[i], marker=MARK[i], ls=LS[i], lw=1.5, ms=5, elinewidth=1, label=MODEL_LABEL[m])
        ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
        ax.set_xticks([10, 20, 30, 45, 60])
        ax.set_title(VARIANT_LABEL[v], fontsize=9, color=INK)
        ax.set_xlabel("t (min)")
    axes[0].set_ylabel("ROC-AUC (95% CI)")
    axes[0].legend(loc="lower right", fontsize=7)
    save(fig, "A3_auc_by_checkpoint_deep")




def fig_d_power() -> None:
    p = pd.read_csv(OUT / "D_power_simulation.csv")
    p = p[(p.alpha_level == 0.05) & (p.target_auc_gain > 0)]
    cols = C + ["#eda100", "#e87ba4"]
    marks = ["o", "s", "^", "D", "v"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for i, t in enumerate(sorted(p.checkpoint_minutes.unique())):
        g = p[p.checkpoint_minutes == t].sort_values("target_auc_gain")
        ax.errorbar(g.target_auc_gain, g.power, yerr=1.96 * g.mc_se, color=cols[i], marker=marks[i],
                    lw=1.5, ms=5, elinewidth=0.9, label=f"t = {int(t)} min")
    ax.axhline(0.8, color=MUTED, lw=0.8, ls="--")
    ax.set_xticks([0.03, 0.05, 0.08, 0.10])
    ax.set_xlabel("True AUC gain from maternal block")
    ax.set_ylabel("Power, Firth LRT 8 df, alpha 0.05")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Simulated power, primary cohort (dashed = 80 %)", fontsize=9, color=INK, loc="left")
    save(fig, "D1_power_maternal_block")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "A"
    if which == "A":
        fig_a_cohorts()
        fig_a_auc()
        fig_a_deep()
    if which == "D":
        fig_d_power()
