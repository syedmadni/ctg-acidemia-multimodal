"""Task A.6 evaluation: seed-averaged deep OOF (reduced encoder, causal inputs) on the
ckfilter / unfiltered / primary345_causal subsets.

Seed averaging: per record and checkpoint, mean over 3 repeats x all finished seeds
(one common grouping key, so rows from different seeds are pooled; CLAUDE.md rule 2).

    python -m src.revision.run_task_a_deep
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.significance import delong_test
from src.revision.cohorts import CHECKPOINTS, VARIANTS, labelled_meta, members, quality_table
from src.revision.raw_cache import RawData
from src.revision.stats_utils import auc_pr_ci, holm, paired_delta_ci

OUT = Path("results/revision")
OOF = OUT / "oof" / "deep_oof.csv"


def seed_averaged(oof: pd.DataFrame) -> pd.DataFrame:
    return (oof.groupby(["model", "checkpoint_minutes", "record_id"], as_index=False)
               .agg(y_true=("y_true", "first"), y_proba=("y_proba", "mean"), n_preds=("y_proba", "size")))


def main() -> None:
    oof = pd.read_csv(OOF, dtype={"record_id": str}, float_precision="round_trip")
    seeds = sorted(oof["seed"].unique())
    counts = oof.groupby(["seed", "model"]).size()
    avg = seed_averaged(oof)
    expected = 3 * len(seeds)
    if not (avg["n_preds"] == expected).all():
        raise SystemExit(f"ABORT: incomplete seeds; expected {expected} predictions per record.\n{counts}")
    raw = RawData()
    meta = labelled_meta(raw)
    q = quality_table(raw)
    disc, comp = [], []
    for variant in VARIANTS:
        for t in CHECKPOINTS:
            ids = set(members(meta, q, variant, t)["record_id"])
            g = avg[(avg.checkpoint_minutes == t) & avg.record_id.isin(ids)]
            by = {m: gg.sort_values("record_id") for m, gg in g.groupby("model")}
            for m, gg in by.items():
                y = gg["y_true"].to_numpy()
                disc.append({"variant": variant, "checkpoint_minutes": t, "model": f"deep_{m}",
                             "encoder": "reduced (d64, 1 layer)", "seeds": ";".join(map(str, seeds)),
                             "trained_on": "unfiltered", "n": len(y), "cases": int(y.sum()),
                             **auc_pr_ci(y, gg["y_proba"].to_numpy())})
            a, b = by["fusion"], by["ctg_only"]
            assert (a.record_id.to_numpy() == b.record_id.to_numpy()).all()
            y = a["y_true"].to_numpy()
            d = delong_test(y, a["y_proba"].to_numpy(), b["y_proba"].to_numpy())
            comp.append({"variant": variant, "checkpoint_minutes": t, "model": "deep fusion vs deep ctg_only",
                         "n": len(y), "cases": int(y.sum()), "auc_combined": d["auc_a"], "auc_ctg_only": d["auc_b"],
                         "delong_p": d["p_value"], **paired_delta_ci(y, a["y_proba"].to_numpy(), b["y_proba"].to_numpy())})
    comp = pd.DataFrame(comp)
    for col_in, col_out in (("delong_p", "delong_holm_p"), ("boot_p_two_sided", "boot_holm_p")):
        comp[col_out] = np.nan
        for _, g in comp.groupby("variant"):
            comp.loc[g.index, col_out] = holm(g[col_in].tolist())
    pd.DataFrame(disc).round(4).to_csv(OUT / "A_deep_discrimination.csv", index=False)
    comp.round(4).to_csv(OUT / "A_deep_fusion_vs_ctg.csv", index=False)
    # per-seed AUC spread (unfiltered, all records) as a stability check
    rows = []
    for (s, m, t), g in oof.groupby(["seed", "model", "checkpoint_minutes"]):
        ga = g.groupby("record_id").agg(y=("y_true", "first"), p=("y_proba", "mean"))
        from sklearn.metrics import roc_auc_score
        rows.append({"seed": s, "model": m, "checkpoint_minutes": t, "auc_unfiltered": roc_auc_score(ga.y, ga.p)})
    pd.DataFrame(rows).round(4).to_csv(OUT / "A_deep_per_seed_auc.csv", index=False)
    print("deep evaluation written")


if __name__ == "__main__":
    main()
