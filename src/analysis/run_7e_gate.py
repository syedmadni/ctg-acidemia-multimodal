"""7e pooled gate analysis on the retrained (duration-aware) gated_fusion (STEP7E_SPEC.md s5).

Per encoder size:
  1. DETERMINISM GATE: the all-folds retrain (step7e_gate_allfolds_<size>) must reproduce
     step7d_<size>_gapped's gated_fusion OOF bit-exactly. On failure the size is reported
     as FAILED and no gate finding is computed for it.
  2. run_anytime_disagreement_pooled.py, unchanged, with --gate-cohort-dir set to the
     retrain and --oof-cohort-dir step7d_<size>_gapped (seed-42 ctg_only/maternal_only,
     same run and seed as the gate model).
  3. Per checkpoint (10-60): mean gate when CTG was right vs when maternal was right, gap,
     two-sided Mann-Whitney U, Holm across 5 within each size.

Sizes whose retrain has not finished (no cv_summary.json) are skipped and listed.

Usage: python -m src.analysis.run_7e_gate
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

BASE = Path("outputs/quality_cohorts")
OUT_DIR = BASE / "primary_705_anytime_vr060" / "maternal_value"
CHECKPOINTS = [10.0, 20.0, 30.0, 45.0, 60.0]
SIZES = {"full": "full-size", "cap64": "reduced"}
DEC = 4


def determinism(size: str) -> dict:
    keys = ["checkpoint_minutes", "repeat", "fold", "record_id"]
    a = pd.read_csv(BASE / f"step7d_{size}_gapped" / "oof_predictions.csv")
    b = pd.read_csv(BASE / f"step7e_gate_allfolds_{size}" / "oof_predictions.csv")
    a = a[a.model == "gated_fusion"].sort_values(keys).reset_index(drop=True)
    b = b[b.model == "gated_fusion"].sort_values(keys).reset_index(drop=True)
    same_keys = len(a) == len(b) and (a[keys].to_numpy() == b[keys].to_numpy()).all()
    maxdiff = float(np.max(np.abs(a.y_proba.to_numpy() - b.y_proba.to_numpy()))) if same_keys else float("nan")
    return {"size": SIZES[size], "rows_7d": len(a), "rows_retrain": len(b), "keys_identical": bool(same_keys),
            "max_abs_diff": maxdiff, "bit_exact": bool(same_keys and maxdiff == 0.0)}


def main():
    det_rows, summ_rows, skipped = [], [], []
    for size, size_label in SIZES.items():
        gate_dir = BASE / f"step7e_gate_allfolds_{size}"
        if not (gate_dir / "cv_summary.json").exists():
            skipped.append(size_label)
            continue
        det = determinism(size)
        det_rows.append(det)
        print(f"[{size_label}] determinism: {det}")
        if not det["bit_exact"]:
            print(f"[{size_label}] FAILED determinism gate -- no gate finding computed.")
            continue

        subprocess.run([sys.executable, "-m", "src.analysis.run_anytime_disagreement_pooled",
                        "--gate-cohort-dir", str(gate_dir),
                        "--oof-cohort-dir", str(BASE / f"step7d_{size}_gapped")], check=True)
        df = pd.read_csv(gate_dir / "anytime_disagreement_analysis_pooled.csv")
        df = df[df.checkpoint_minutes.isin(CHECKPOINTS)]
        assert df.groupby("checkpoint_minutes").size().eq(345).all()
        assert (df["n_folds_pooled"] == 3).all()

        rows = []
        for ck in CHECKPOINTS:
            g = df[df.checkpoint_minutes == ck]
            dis = g[g.disagreement]
            ctg_r = dis[dis.ctg_reading_closer_to_truth == True]["gate_value"]   # noqa: E712
            mat_r = dis[dis.ctg_reading_closer_to_truth == False]["gate_value"]  # noqa: E712
            p = mannwhitneyu(ctg_r, mat_r, alternative="two-sided").pvalue if len(ctg_r) and len(mat_r) else np.nan
            rows.append({"size": size_label, "checkpoint_minutes": ck, "n": len(g),
                         "n_disagreement": len(dis), "n_ctg_right": len(ctg_r), "n_maternal_right": len(mat_r),
                         "gate_mean_all": g.gate_value.mean(),
                         "gate_mean_when_ctg_right": ctg_r.mean(), "gate_mean_when_maternal_right": mat_r.mean(),
                         "gap": ctg_r.mean() - mat_r.mean(), "mannwhitney_p": p})
        s = pd.DataFrame(rows)
        s["mannwhitney_holm_p"] = multipletests(s["mannwhitney_p"], method="holm")[1]
        summ_rows.append(s)

    if det_rows:
        pd.DataFrame(det_rows).to_csv(OUT_DIR / "step7e_gate_determinism.csv", index=False)
    if summ_rows:
        out = pd.concat(summ_rows, ignore_index=True)
        out = out.round({c: DEC for c in out.select_dtypes("float").columns if c not in ("mannwhitney_p", "mannwhitney_holm_p")})
        out.to_csv(OUT_DIR / "step7e_gate_summary.csv", index=False)
        print(out.to_string(index=False))
    if skipped:
        print(f"NOT YET AVAILABLE (retrain unfinished): {skipped}")


if __name__ == "__main__":
    main()
