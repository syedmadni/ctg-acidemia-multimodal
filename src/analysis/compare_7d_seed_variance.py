"""7d seed variance: how much does ROC-AUC move when only the training seed changes?

Motivation: in 7d, `fusion` rose at all five checkpoints with the full-size encoder and
fell at all five with the reduced encoder, while `ctg_only` did the opposite. Two runs
differing only in encoder size gave opposite answers, which is the signature of
run-to-run noise. This script quantifies that noise directly by re-running the same
models with different training seeds while holding the folds fixed (fold seed stays 42;
only `training_seed` changes, so weight init and crop sampling differ).

Reports per seed, plus mean, min-max range, and `fusion` minus `ctg_only` per seed.

PRE-SPECIFIED (STEP7D_SPEC.md section 8): 7e uses SEED-AVERAGED OOF predictions for
every deep model that has more than one seed. This script writes those averaged OOF
files so 7e consumes them directly rather than re-deriving them.

Usage: python -m src.analysis.compare_7d_seed_variance
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.training.train_cv import aggregate_repeated_oof_anytime  # noqa: E402

BASE = Path("outputs/quality_cohorts")
OUT_DIR = Path("outputs/quality_cohorts/primary_705_anytime_vr060/maternal_value")
CHECKPOINTS = [10.0, 20.0, 30.0, 45.0, 60.0]
MODELS = ["ctg_only", "fusion"]
DEC = 4

# arm -> {seed: cohort_dir_name}. Seed 42 is the original 7d run.
ARMS = {
    "reduced encoder (PRIMARY seed check)": {
        42: "step7d_cap64_gapped",
        43: "step7d_cap64_gapped_seed43",
        44: "step7d_cap64_gapped_seed44",
    },
    "full-size encoder (OPTIONAL)": {
        42: "step7d_full_gapped",
        43: "step7d_full_gapped_seed43",
    },
}


def load_metrics(cohort: str):
    p = BASE / cohort / "cv_summary.json"
    if not p.exists():
        return None
    return json.load(p.open())["model_metrics_from_averaged_oof"]


def get(metrics, model: str, ck: float):
    if metrics is None:
        return np.nan
    for key in (f"{model}@{ck}min", f"{model}@{ck:.1f}min", f"{model}@{int(ck)}min"):
        if key in metrics:
            return float(metrics[key]["roc_auc"])
    return np.nan


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []

    for arm, seed_map in ARMS.items():
        for seed, cohort in seed_map.items():
            m = load_metrics(cohort)
            if m is None:
                missing.append(f"{cohort} (seed {seed})")
                continue
            for model in MODELS:
                for ck in CHECKPOINTS:
                    v = get(m, model, ck)
                    if not np.isnan(v):
                        rows.append({"arm": arm, "seed": seed, "cohort": cohort,
                                     "model": model, "checkpoint_minutes": ck,
                                     "roc_auc": round(v, DEC)})
    if missing:
        print(f"NOT YET AVAILABLE (skipped): {missing}\n")
    if not rows:
        raise SystemExit("No seed-variance results found. Run scripts/run_seed_variance.ps1 first.")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "step7d_seed_variance_raw.csv", index=False)

    summary_rows = []
    for arm in df["arm"].unique():
        a = df[df.arm == arm]
        seeds = sorted(a["seed"].unique())
        print("=" * 78)
        print(f"{arm}  |  seeds present: {seeds}")
        print("=" * 78)

        for model in MODELS:
            sub = a[a.model == model]
            if sub.empty:
                continue
            piv = sub.pivot(index="checkpoint_minutes", columns="seed", values="roc_auc")
            piv["mean"] = piv[seeds].mean(axis=1).round(DEC)
            piv["min"] = piv[seeds].min(axis=1)
            piv["max"] = piv[seeds].max(axis=1)
            piv["range"] = (piv["max"] - piv["min"]).round(DEC)
            print(f"\n-- {model} ROC-AUC by seed --")
            print(piv.to_string())
            for ck in CHECKPOINTS:
                if ck in piv.index:
                    summary_rows.append({
                        "arm": arm, "model": model, "checkpoint_minutes": ck,
                        **{f"seed_{s}": piv.loc[ck, s] for s in seeds},
                        "mean": piv.loc[ck, "mean"], "min": piv.loc[ck, "min"],
                        "max": piv.loc[ck, "max"], "seed_range": piv.loc[ck, "range"],
                    })

        # fusion minus ctg_only, per seed
        print("\n-- fusion MINUS ctg_only, per seed --")
        diffs = []
        for s in seeds:
            f = a[(a.model == "fusion") & (a.seed == s)].set_index("checkpoint_minutes")["roc_auc"]
            c = a[(a.model == "ctg_only") & (a.seed == s)].set_index("checkpoint_minutes")["roc_auc"]
            if f.empty or c.empty:
                continue
            diffs.append((f - c).rename(f"seed_{s}"))
        if diffs:
            dd = pd.concat(diffs, axis=1).round(DEC)
            dd["mean"] = dd.mean(axis=1).round(DEC)
            dd["sign_agreement"] = dd[[c for c in dd.columns if c.startswith("seed_")]].apply(
                lambda r: "all +" if (r > 0).all() else ("all -" if (r < 0).all() else "MIXED"), axis=1)
            print(dd.to_string())
            dd.to_csv(OUT_DIR / f"step7d_seed_variance_fusion_minus_ctgonly_{arm.split()[0]}.csv")
            n_mixed = int((dd["sign_agreement"] == "MIXED").sum())
            print(f"\n  checkpoints where the fusion-vs-ctg_only SIGN flips across seeds: "
                  f"{n_mixed} of {len(dd)}")
        print()

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "step7d_seed_variance_summary.csv", index=False)

    # ---- seed-averaged OOF for 7e (pre-specified) ----
    written = []
    for arm, seed_map in ARMS.items():
        frames = []
        for seed, cohort in seed_map.items():
            p = BASE / cohort / "oof_predictions.csv"
            if p.exists():
                o = pd.read_csv(p)
                o["seed"] = seed
                frames.append(o)
        if len(frames) < 2:
            continue
        pooled = pd.concat(frames, ignore_index=True)
        pooled = pooled[pooled.model.isin(MODELS)]
        tag = "cap64" if "reduced" in arm else "full"
        # aggregate_repeated_oof_anytime groups by cohort_name, which differs per seed
        # (step7d_x_gapped vs step7d_x_gapped_seed43). Without a common name nothing is
        # averaged ACROSS seeds (bug found 2026-09-22 before 7e consumed these files).
        pooled["cohort_name"] = f"seed_averaged_{tag}"
        # average across seeds AND repeats per (model, checkpoint, record)
        avg = aggregate_repeated_oof_anytime(pooled)
        n_per_key = pooled.groupby(["model", "checkpoint_minutes", "record_id"]).size()
        n_seeds = len(frames)
        assert (n_per_key == 3 * n_seeds).all(), "expected 3 repeats x n_seeds rows per record"
        assert len(avg) == pooled.groupby(["model", "checkpoint_minutes", "record_id"]).ngroups
        out = OUT_DIR / f"step7d_seed_averaged_oof_{tag}.csv"
        avg.to_csv(out, index=False)
        written.append(f"{out.name} (seeds {sorted(s for s in seed_map if (BASE / seed_map[s] / 'oof_predictions.csv').exists())})")
    if written:
        print("Seed-averaged OOF written for 7e:")
        for w in written:
            print("  " + w)

    print(f"\nWrote summary tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
