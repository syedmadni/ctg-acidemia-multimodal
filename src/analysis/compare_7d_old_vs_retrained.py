"""7d: old vs retrained deep-model comparison.

Compares deep-model ROC-AUC before and after the duration-aware training-cutoff fix
(cutoffs drawn from [8, min(90, duration)] instead of [8, 90]).

Spec: outputs/quality_cohorts/primary_705_anytime_vr060/maternal_value/STEP7D_SPEC.md

Pairings (like with like -- same encoder size, same label, same folds/seed):
  full size,  gapped:  primary_705_anytime_vr060  ->  step7d_full_gapped
  reduced,    gapped:  anytime_cap64              ->  step7d_cap64_gapped
  reduced,    consec:  (no prior run)             ->  step7d_cap64_consec   [new, no old side]

Old values are READ from the existing cv_summary.json files, never retyped. Checkpoints
10-60 only; 90 min is excluded from all inference per Amendment 1.

Usage: python -m src.analysis.compare_7d_old_vs_retrained
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BASE = Path("outputs/quality_cohorts")
OUT_DIR = Path("outputs/quality_cohorts/primary_705_anytime_vr060/maternal_value")
CHECKPOINTS = [10.0, 20.0, 30.0, 45.0, 60.0]
DEC = 4

PAIRS = [
    {"arm": "PRIMARY (full-size encoder, gapped)", "old": "primary_705_anytime_vr060", "new": "step7d_full_gapped"},
    {"arm": "SENSITIVITY 1 (reduced encoder, gapped)", "old": "anytime_cap64", "new": "step7d_cap64_gapped"},
    {"arm": "SENSITIVITY 2 (reduced encoder, consecutive)", "old": None, "new": "step7d_cap64_consec"},
]
DEEP_MODELS = ["ctg_only", "maternal_only", "fusion", "gated_fusion"]


def load_metrics(cohort: str) -> dict | None:
    p = BASE / cohort / "cv_summary.json"
    if not p.exists():
        return None
    return json.load(p.open())["model_metrics_from_averaged_oof"]


def get(metrics: dict | None, model: str, ck: float, field: str = "roc_auc"):
    if metrics is None:
        return np.nan
    for key in (f"{model}@{ck}min", f"{model}@{ck:.1f}min", f"{model}@{int(ck)}min"):
        if key in metrics:
            return float(metrics[key][field])
    return np.nan


def main():
    rows = []
    missing = []
    for pair in PAIRS:
        new_m = load_metrics(pair["new"])
        if new_m is None:
            missing.append(pair["new"])
            continue
        old_m = load_metrics(pair["old"]) if pair["old"] else None
        if pair["old"] and old_m is None:
            missing.append(pair["old"])
        for model in DEEP_MODELS:
            for ck in CHECKPOINTS:
                old_v = get(old_m, model, ck)
                new_v = get(new_m, model, ck)
                if np.isnan(old_v) and np.isnan(new_v):
                    continue
                rows.append({
                    "arm": pair["arm"], "old_cohort": pair["old"] or "(none)", "new_cohort": pair["new"],
                    "model": model, "checkpoint_minutes": ck,
                    "old_roc_auc": round(old_v, DEC) if not np.isnan(old_v) else np.nan,
                    "retrained_roc_auc": round(new_v, DEC) if not np.isnan(new_v) else np.nan,
                    "delta": round(new_v - old_v, DEC) if not (np.isnan(old_v) or np.isnan(new_v)) else np.nan,
                    "old_pr_auc": round(get(old_m, model, ck, "pr_auc"), DEC) if old_m else np.nan,
                    "retrained_pr_auc": round(get(new_m, model, ck, "pr_auc"), DEC),
                })

    if missing:
        print(f"NOTE: cohorts not yet available, skipped: {sorted(set(missing))}")
    if not rows:
        raise SystemExit("No 7d results found yet. Run the retrain(s) first.")

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "step7d_old_vs_retrained.csv", index=False)

    for arm, g in df.groupby("arm", sort=False):
        print(f"\n{'=' * 80}\n{arm}\n{'=' * 80}")
        if g["old_roc_auc"].isna().all():
            piv = g.pivot(index="checkpoint_minutes", columns="model", values="retrained_roc_auc")
            print("(no prior run; retrained ROC-AUC only)")
            print(piv.to_string())
            continue
        for field, lbl in [("old_roc_auc", "OLD"), ("retrained_roc_auc", "RETRAINED"), ("delta", "DELTA")]:
            piv = g.pivot(index="checkpoint_minutes", columns="model", values=field)
            cols = [c for c in DEEP_MODELS if c in piv.columns]
            print(f"\n-- {lbl} ROC-AUC --")
            print(piv[cols].to_string())
        d = g["delta"].dropna()
        if len(d):
            print(f"\nsummary: mean delta {d.mean():+.4f} | worse in {int((d < 0).sum())}/{len(d)} cells "
                  f"| range [{d.min():+.4f}, {d.max():+.4f}]")

    print(f"\nWrote {OUT_DIR / 'step7d_old_vs_retrained.csv'}")


if __name__ == "__main__":
    main()
