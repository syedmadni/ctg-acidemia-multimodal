"""Phase 6c: post-hoc ensemble of a deep model's OOF predictions with
logistic_regression's, per checkpoint. NO retraining -- pure combination of
predictions already sitting in oof_predictions.csv.

Motivation: a CNN and a logistic regression make different kinds of mistakes.
Averaging their predicted probabilities is one of the most reliable, lowest-risk
ways to gain accuracy on small data, and costs nothing since both models already
exist. This is deliberately the cheapest experiment in the improvement plan.

Usage:
  python -m src.analysis.run_ensemble_analysis --cohort-dir outputs/quality_cohorts/primary_705_anytime_vr060
  python -m src.analysis.run_ensemble_analysis --cohort-dir outputs/quality_cohorts/anytime_cap64 --lr-cohort-dir outputs/quality_cohorts/primary_705_anytime_vr060

--lr-cohort-dir lets the deep model come from one cohort (e.g. the shrunk
anytime_cap64, which only trained ctg_only/fusion/gated_fusion) while
logistic_regression's OOF is pulled from a different cohort that actually has it
(e.g. the original primary_705_anytime_vr060) -- record_id + checkpoint join, not
row-position, so this is safe as long as both share the same held-out record ids
per checkpoint (they do: same underlying arrays, same seed, same split recipe).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.training.train_cv import aggregate_repeated_oof_anytime, bootstrap_metric_ci_anytime

CHECKPOINTS = [10.0, 20.0, 30.0, 45.0, 60.0, 90.0]
DEEP_MODELS = ["ctg_only", "fusion", "gated_fusion"]
THRESHOLD = 0.5
BOOTSTRAP_ITERATIONS = 1000
SEED = 42


def load_averaged(cohort_dir: Path) -> pd.DataFrame:
    oof = pd.read_csv(cohort_dir / "oof_predictions.csv")
    return aggregate_repeated_oof_anytime(oof)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc ensemble of a deep model with logistic_regression.")
    parser.add_argument("--cohort-dir", required=True, help="Cohort providing the deep model(s) to ensemble.")
    parser.add_argument("--lr-cohort-dir", default=None, help="Cohort providing logistic_regression OOF (default: same as --cohort-dir).")
    args = parser.parse_args()

    deep_dir = Path(args.cohort_dir)
    lr_dir = Path(args.lr_cohort_dir) if args.lr_cohort_dir else deep_dir

    deep_avg = load_averaged(deep_dir)
    lr_avg = load_averaged(lr_dir)
    lr_only = lr_avg[lr_avg["model"] == "logistic_regression"]
    if lr_only.empty:
        raise ValueError(f"No logistic_regression OOF found in {lr_dir}")

    available_deep = [m for m in DEEP_MODELS if m in set(deep_avg["model"].unique())]
    if not available_deep:
        raise ValueError(f"No deep models ({DEEP_MODELS}) found in {deep_dir}")

    ensemble_rows = []
    for deep_model in available_deep:
        deep_only = deep_avg[deep_avg["model"] == deep_model]
        for ck in CHECKPOINTS:
            d = deep_only[deep_only["checkpoint_minutes"] == ck][["record_id", "y_true", "y_proba"]].rename(
                columns={"y_proba": "proba_deep"}
            )
            lr = lr_only[lr_only["checkpoint_minutes"] == ck][["record_id", "y_true", "y_proba"]].rename(
                columns={"y_proba": "proba_lr"}
            )
            merged = d.merge(lr, on="record_id", suffixes=("", "_lr"))
            mismatched = merged[merged["y_true"] != merged["y_true_lr"]]
            if len(mismatched) > 0:
                raise ValueError(f"y_true mismatch on {len(mismatched)} record(s) at ckpt {ck} -- cohorts do not share record ids/labels.")
            merged["proba_ensemble"] = 0.5 * merged["proba_deep"] + 0.5 * merged["proba_lr"]
            for _, row in merged.iterrows():
                ensemble_rows.append({
                    "cohort_name": f"{deep_model}_x_lr",
                    "checkpoint_minutes": ck,
                    "model": f"ensemble_{deep_model}_lr",
                    "record_id": row["record_id"],
                    "y_true": int(row["y_true"]),
                    "y_proba": float(row["proba_ensemble"]),
                })

    ensemble_avg = pd.DataFrame(ensemble_rows)
    ci = bootstrap_metric_ci_anytime(ensemble_avg, THRESHOLD, BOOTSTRAP_ITERATIONS, SEED)

    out_dir = deep_dir
    ensemble_avg.to_csv(out_dir / "ensemble_oof_predictions.csv", index=False)
    ci.to_csv(out_dir / "ensemble_bootstrap_confidence_intervals.csv", index=False)

    print(f"\n=== Ensemble (0.5*deep + 0.5*LR) vs individual models, {deep_dir.name} (LR from {lr_dir.name}) ===")
    for deep_model in available_deep:
        deep_only = deep_avg[deep_avg["model"] == deep_model]
        print(f"\n{deep_model}")
        for ck in CHECKPOINTS:
            d_roc = deep_only[deep_only["checkpoint_minutes"] == ck]
            lr_roc = lr_only[lr_only["checkpoint_minutes"] == ck]
            ens_roc = ensemble_avg[
                (ensemble_avg["model"] == f"ensemble_{deep_model}_lr") & (ensemble_avg["checkpoint_minutes"] == ck)
            ]
            from sklearn.metrics import roc_auc_score
            r_deep = roc_auc_score(d_roc["y_true"], d_roc["y_proba"])
            r_lr = roc_auc_score(lr_roc["y_true"], lr_roc["y_proba"])
            r_ens = roc_auc_score(ens_roc["y_true"], ens_roc["y_proba"])
            best_single = max(r_deep, r_lr)
            beats = "BEATS both" if r_ens > best_single else ("ties best" if abs(r_ens - best_single) < 1e-9 else "does NOT beat best single")
            print(f"  {ck:5.0f}min  {deep_model}={r_deep:.3f}  LR={r_lr:.3f}  ensemble={r_ens:.3f}  -> {beats}")

    print(f"\nWrote {out_dir / 'ensemble_oof_predictions.csv'}")
    print(f"Wrote {out_dir / 'ensemble_bootstrap_confidence_intervals.csv'}")


if __name__ == "__main__":
    main()
