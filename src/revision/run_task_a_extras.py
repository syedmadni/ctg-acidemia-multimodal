"""Task A extras (author request 2026-09-29): parity 2x2 + fragility per checkpoint, and the
"leak effect" table (same 345 records and folds: v2 interpolated vs causal features).

    python -m src.revision.run_task_a_extras

Parity: POST HOC, descriptive. OR convention matches robust4 (Woolf CI; Haldane-Anscombe
+0.5 only when a cell is 0; fragility = one parous control recoded as a case).
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.run_maternal_value_steps123 import odds_ratio_2x2
from src.analysis.significance import delong_test
from src.data.fixed_ctg_features import ctg_summary_features_v2
from src.revision.causal_features import causal_feature_matrix
from src.revision.cohorts import CHECKPOINTS, labelled_meta, load_splits, maternal_matrix, members, quality_table
from src.revision.raw_cache import RawData
from src.revision.stats_utils import auc_pr_ci, holm, paired_delta_ci
from src.training.train_cv import aggregate_repeated_oof_anytime, load_cohort_arrays, train_sklearn_quality_fold

OUT = Path("results/revision")
SEED = 42


def or_row(a, b, c, d):
    fo, fp = stats.fisher_exact(np.array([[a, b], [c, d]]))
    or_, se, ha = odds_ratio_2x2(a, b, c, d)
    return {"odds_ratio": or_, "ci_low": float(np.exp(np.log(or_) - 1.96 * se)),
            "ci_high": float(np.exp(np.log(or_) + 1.96 * se)), "fisher_p": fp, "haldane_used": ha}


def parity_table(raw, meta, q) -> pd.DataFrame:
    rows = []
    for variant in ["ckfilter", "unfiltered"]:
        for t in CHECKPOINTS:
            coh = members(meta, q, variant, t)
            nulli = (coh["parity"] == 0).to_numpy()
            y = coh["y"].to_numpy()
            a, b = int((nulli & (y == 1)).sum()), int((nulli & (y == 0)).sum())
            c, d = int((~nulli & (y == 1)).sum()), int((~nulli & (y == 0)).sum())
            obs, frag = or_row(a, b, c, d), or_row(a, b, c + 1, d - 1)
            rows.append({
                "label": "POST HOC, descriptive", "variant": variant, "checkpoint_minutes": t,
                "n": len(y), "cases": int(y.sum()), "parous_cases": c,
                "parous_cases_of_total": f"{c}/{int(y.sum())}",
                "nulliparous_cases": a, "nulliparous_controls": b, "parous_controls": d,
                "nulliparity_OR": obs["odds_ratio"], "OR_ci_low": obs["ci_low"], "OR_ci_high": obs["ci_high"],
                "fisher_p": obs["fisher_p"], "haldane_used": obs["haldane_used"],
                "fragility_OR_one_parous_control_to_case": frag["odds_ratio"],
                "fragility_ci_low": frag["ci_low"], "fragility_ci_high": frag["ci_high"],
                "fragility_fisher_p": frag["fisher_p"],
            })
    return pd.DataFrame(rows)


def leak_effect(raw, meta, q) -> pd.DataFrame:
    X_ctg, X_mat, y, rid_df, _ = load_cohort_arrays(Path("data/processed"), "_705_anytime_vr060")
    rows = []
    scratch = Path(tempfile.mkdtemp(prefix="revA_leak_"))
    try:
        for t in CHECKPOINTS:
            coh = members(meta, q, "primary345_causal", t)
            assert coh["record_id"].tolist() == rid_df["record_id"].astype(str).tolist()
            assert (coh["y"].to_numpy() == y).all()
            splits = load_splits(coh, "primary345_causal", t)
            mat = maternal_matrix(coh)
            feats = {
                "v2_interpolated": ctg_summary_features_v2(X_ctg, checkpoint_minutes=t),
                "causal": causal_feature_matrix(raw, coh["record_id"].to_numpy(), t),
            }
            for model in ["logistic_regression", "random_forest"]:
                for fs in ["ctg_only", "combined"]:
                    p = {}
                    for fv, ctg in feats.items():
                        X = ctg if fs == "ctg_only" else np.concatenate([ctg, mat], axis=1)
                        rows_oof = []
                        for sp in splits:
                            _, _, tp, _ = train_sklearn_quality_fold(model, X, y, sp, SEED, scratch)
                            rows_oof += [{"cohort_name": fv, "checkpoint_minutes": t, "model": model,
                                          "record_id": coh["record_id"].iloc[i], "y_true": int(y[i]), "y_proba": float(v)}
                                         for i, v in zip(np.asarray(sp["outer_test_idx"]), tp)]
                        # same repeat-averaging as Task A (summation order matters for RF ties)
                        avg = aggregate_repeated_oof_anytime(pd.DataFrame(rows_oof)).set_index("record_id")
                        p[fv] = avg.loc[coh["record_id"], "y_proba"].to_numpy()
                    a = auc_pr_ci(y, p["v2_interpolated"])
                    b = auc_pr_ci(y, p["causal"])
                    d = paired_delta_ci(y, p["causal"], p["v2_interpolated"])
                    dl = delong_test(y, p["causal"], p["v2_interpolated"])
                    rows.append({
                        "checkpoint_minutes": t, "model": model, "feature_set": fs, "n": len(y), "cases": int(y.sum()),
                        "auc_v2_interpolated": a["roc_auc"], "v2_ci_low": a["roc_auc_ci_low"], "v2_ci_high": a["roc_auc_ci_high"],
                        "auc_causal": b["roc_auc"], "causal_ci_low": b["roc_auc_ci_low"], "causal_ci_high": b["roc_auc_ci_high"],
                        "delta_causal_minus_v2": d["delta_auc"], "delta_ci_low": d["delta_ci_low"],
                        "delta_ci_high": d["delta_ci_high"], "delong_p": dl["p_value"],
                    })
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    df = pd.DataFrame(rows)
    df["delong_holm_p"] = np.nan
    for _, g in df.groupby(["model", "feature_set"]):
        df.loc[g.index, "delong_holm_p"] = holm(g["delong_p"].tolist())
    return df


def main():
    raw = RawData()
    meta = labelled_meta(raw)
    q = quality_table(raw)
    parity_table(raw, meta, q).round(4).to_csv(OUT / "A_parity_2x2_fragility.csv", index=False)
    leak_effect(raw, meta, q).round(4).to_csv(OUT / "A_leak_effect_345.csv", index=False)
    print("written")


if __name__ == "__main__":
    main()
