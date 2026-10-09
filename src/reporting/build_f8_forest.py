"""F8: forest figure of TF8 (gapped label, ROC-AUC [95% CI]) across checkpoints 10-60.

F8a: small multiples, one panel per checkpoint, shared model rows.
F8b: single panel, 5 checkpoint bars dodged within each model row (ordinal blue ramp).

Values come from step7e_modality_auc.csv (round-trip float parsing, RESULTS_FACTS 19.0a).
Self-check (abort on failure): every (point, lower, upper) read back from the plotted
artists equals TF8 gapped ROC-AUC to 3 dp, 65 cells per figure. Results are written to
F8_self_verification.csv. Nothing under paper/ is touched.

Usage: python -m src.reporting.build_f8_forest
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

MV = Path("outputs/quality_cohorts/primary_705_anytime_vr060/maternal_value")
TAB = Path("outputs/report_figures/tables_fixed")
FIG = Path("outputs/report_figures/final")
CHECKPOINTS = [10.0, 20.0, 30.0, 45.0, 60.0]
BLUE, GREEN = "#2A74D0", "#1AAE7A"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # ordinal, light (10) -> dark (60)
FS = 7.5  # base font size (pt); nothing below 7
SIZE = (7.25, 4.5)

# (label, 7e selector, TF8 (model, encoder), kind, single_seed, group)
ROWS = [
    ("LR, CTG only", dict(fusion_type="LR concatenation", arm="ctg"), ("LR, CTG only", "n/a"), "classical", False, 0),
    ("LR, maternal only", dict(fusion_type="LR concatenation", arm="mat"), ("LR, maternal only", "n/a"), "classical", False, 0),
    ("LR, CTG + maternal", dict(fusion_type="LR concatenation", arm="comb"), ("LR, CTG + maternal", "n/a"), "classical", False, 0),
    ("RF, CTG only", dict(fusion_type="RF concatenation", arm="ctg"), ("RF, CTG only", "n/a"), "classical", False, 1),
    ("RF, maternal only", dict(fusion_type="RF concatenation", arm="mat"), ("RF, maternal only", "n/a"), "classical", False, 1),
    ("RF, CTG + maternal", dict(fusion_type="RF concatenation", arm="comb"), ("RF, CTG + maternal", "n/a"), "classical", False, 1),
    ("Deep CTG only, full-size", dict(fusion_type="deep fusion", size="full-size", arm="ctg"), ("deep ctg_only", "full-size"), "deep", False, 2),
    ("Deep fusion, full-size", dict(fusion_type="deep fusion", size="full-size", arm="comb"), ("deep fusion", "full-size"), "deep", False, 2),
    ("Deep gated fusion, full-size", dict(fusion_type="deep gated fusion", size="full-size", arm="comb"), ("deep gated_fusion", "full-size"), "deep", True, 2),
    ("Deep CTG only, reduced", dict(fusion_type="deep fusion", size="reduced", arm="ctg"), ("deep ctg_only", "reduced"), "deep", False, 3),
    ("Deep fusion, reduced", dict(fusion_type="deep fusion", size="reduced", arm="comb"), ("deep fusion", "reduced"), "deep", False, 3),
    ("Deep gated fusion, reduced", dict(fusion_type="deep gated fusion", size="reduced", arm="comb"), ("deep gated_fusion", "reduced"), "deep", True, 3),
    ("Deep maternal-only network", dict(fusion_type="deep fusion", size="full-size", arm="mat"), ("deep maternal_only", "full-size"), "deep", True, 4),
]


def rc(path):
    return pd.read_csv(path, float_precision="round_trip", keep_default_na=False)


def label(r):
    return r[0] + (" (single seed)" if r[4] else "")


def load_values():
    s = rc(MV / "step7e_modality_auc.csv")
    s = s[s.label == "gapped"]
    # Deep maternal-only: one row only if every copy (both sizes, both deep groups) is identical.
    mat = s[s.arm.eq("mat") & s.fusion_type.str.startswith("deep")]
    for ck in CHECKPOINTS:
        m = mat[mat.checkpoint_minutes == ck][["roc_auc", "roc_ci_low", "roc_ci_high"]]
        if len(m) != 4 or len(m.drop_duplicates()) != 1:
            raise SystemExit(f"deep maternal_only differs between full-size and reduced at {ck} min:\n{m}")
    m60 = tuple(round(v, 3) for v in mat.iloc[0][["roc_auc", "roc_ci_low", "roc_ci_high"]])
    if m60 != (0.543, 0.438, 0.642):
        raise SystemExit(f"deep maternal_only is {m60}, expected (0.543, 0.438, 0.642)")
    vals = {}
    for i, r in enumerate(ROWS):
        sel = s
        for k, v in r[1].items():
            sel = sel[sel[k] == v]
        for ck in CHECKPOINTS:
            x = sel[sel.checkpoint_minutes == ck]
            if len(x) != 1:
                raise SystemExit(f"row {r[0]} @ {ck}: {len(x)} source rows")
            x = x.iloc[0]
            vals[i, ck] = (x.roc_auc, x.roc_ci_low, x.roc_ci_high)
    return vals


def style_axis(ax, n):
    ax.axvline(0.5, color="#9a9a94", lw=0.8, ls=(0, (1, 2)), zorder=0)
    ax.set_xlim(0.30, 1.00)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(n - 0.5, -0.5)
    ax.tick_params(labelsize=FS, length=2.5, width=0.6)
    ax.grid(axis="x", color="#ececE7", lw=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)
    # thin separators between groups
    for i in range(1, n):
        if ROWS[i][5] != ROWS[i - 1][5]:
            ax.axhline(i - 0.5, color="#b8b8b2", lw=0.5, zorder=0)


def marker_kw(r, color, ms):
    return dict(marker="o" if r[3] == "classical" else "s", ms=ms, color=color, markeredgecolor=color,
                markeredgewidth=0.9, markerfacecolor="white" if r[4] else color, ls="none", zorder=3)


def draw_row(ax, i, ck, r, y, v, color, ms, lw):
    auc, lo, hi = v
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=2, gid=f"ci|{i}|{ck}")
    ax.plot([auc], [y], gid=f"pt|{i}|{ck}", **marker_kw(r, color, ms))


def read_back(fig):
    """(row, ck) -> (auc, lo, hi) recovered from the drawn artists, not from the inputs."""
    ci, pt = {}, {}
    for ax in fig.axes:
        for ln in ax.lines:
            gid = ln.get_gid()
            if not gid:
                continue
            kind, i, ck = gid.split("|")
            key = (int(i), float(ck))
            xs = [float(x) for x in ln.get_xdata()]
            (ci if kind == "ci" else pt)[key] = xs
    return {k: (pt[k][0], min(ci[k]), max(ci[k])) for k in ci}


def fig_a(vals):
    n = len(ROWS)
    fig, axes = plt.subplots(1, 5, figsize=SIZE, sharey=True)
    for j, (ax, ck) in enumerate(zip(axes, CHECKPOINTS)):
        style_axis(ax, n)
        for i, r in enumerate(ROWS):
            color = BLUE if r[3] == "classical" else GREEN
            draw_row(ax, i, ck, r, i, vals[i, ck], color, ms=4.2, lw=1.1)
        ax.set_title(f"{int(ck)} min", fontsize=FS + 0.5, pad=4)
        if j:
            ax.tick_params(axis="y", left=False)
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels([label(r) for r in ROWS], fontsize=FS)
    fig.tight_layout(rect=(0, 0.03, 1, 1), w_pad=0.6)
    fig.supxlabel("ROC-AUC (95% CI)", fontsize=FS + 0.5, y=0.015)
    return fig


def fig_b(vals):
    n = len(ROWS)
    fig, ax = plt.subplots(figsize=SIZE)
    style_axis(ax, n)
    offs = [-0.32, -0.16, 0.0, 0.16, 0.32]
    for i, r in enumerate(ROWS):
        for ck, off, color in zip(CHECKPOINTS, offs, RAMP):
            draw_row(ax, i, ck, r, i + off, vals[i, ck], color, ms=3.4, lw=1.0)
    ax.set_yticks(range(n))
    ax.set_yticklabels([label(r) for r in ROWS], fontsize=FS)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("ROC-AUC (95% CI)", fontsize=FS + 0.5)
    ck_h = [Line2D([0], [0], color=c, lw=1.6, label=f"{int(ck)} min") for ck, c in zip(CHECKPOINTS, RAMP)]
    mk_h = [Line2D([0], [0], marker="o", ls="none", color="#4a4a46", ms=4, label="LR / RF"),
            Line2D([0], [0], marker="s", ls="none", color="#4a4a46", ms=4, label="deep, seed-averaged"),
            Line2D([0], [0], marker="s", ls="none", color="#4a4a46", markerfacecolor="white", ms=4,
                   label="deep, single seed")]
    fig.legend(handles=ck_h + mk_h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=8, fontsize=FS,
               frameon=False, handlelength=1.4, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig


def smallest_font(fig):
    fig.canvas.draw()
    sizes = [t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip()]
    return min(sizes)


def verify(name, plotted, tf8):
    rows, bad = [], 0
    for i, r in enumerate(ROWS):
        model, enc = r[2]
        for ck in CHECKPOINTS:
            t = tf8[(tf8.model == model) & (tf8.encoder == enc) & (tf8.checkpoint_minutes == ck)]
            assert len(t) == 1, (model, enc, ck)
            t = t.iloc[0]
            p = plotted[i, ck]
            tv = (float(t.roc_auc), float(t.roc_ci_low), float(t.roc_ci_high))
            # Python-float formatting on both sides: np.round and round(float) disagree on
            # half-way cases such as 0.6485, which would flag identical values as mismatches.
            ok = all(f"{float(a):.3f}" == f"{b:.3f}" for a, b in zip(p, tv))
            bad += not ok
            rows.append(dict(figure=name, row=label(r), checkpoint=int(ck), plotted_auc=p[0], plotted_lo=p[1],
                             plotted_hi=p[2], tf8_auc=tv[0], tf8_lo=tv[1], tf8_hi=tv[2], match=ok))
    return rows, len(rows) - bad, len(rows)


def main():
    vals = load_values()
    tf8 = rc(TAB / "TF8_all_models_discrimination.csv")
    tf8 = tf8[tf8.label == "gapped"]
    out, report = [], []
    for name, builder in (("F8a", fig_a), ("F8b", fig_b)):
        fig = builder(vals)
        rows, ok, total = verify(name, read_back(fig), tf8)
        out += rows
        if ok != total:
            pd.DataFrame(out).to_csv(FIG / "F8_self_verification.csv", index=False)
            raise SystemExit(f"{name}: {total - ok} cells do not match TF8; figure NOT saved")
        stem = FIG / f"{name}_forest_auc_all_checkpoints"
        fig.savefig(stem.with_suffix(".png"), dpi=600)
        fig.savefig(stem.with_suffix(".pdf"))
        w, h = fig.get_size_inches()
        report.append(f"{name}: {ok}/{total} cells match TF8; size {w:.2f} x {h:.2f} in; "
                      f"smallest font {smallest_font(fig):.1f} pt")
        plt.close(fig)
    pd.DataFrame(out).to_csv(FIG / "F8_self_verification.csv", index=False)
    print("\n".join(report))


if __name__ == "__main__":
    main()
