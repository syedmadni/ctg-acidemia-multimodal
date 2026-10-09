"""Scan outputs/quality_cohorts/*/ and build a single cross-cohort results summary.

Reads every cv_summary.json under the cohorts directory (plus, where present,
bootstrap_confidence_intervals.csv in the same cohort directory) and writes:

  outputs/results_summary.json - one entry per cohort, models nested underneath.
  outputs/results_summary.csv  - flattened, one row per (cohort, model).

Also prints a console summary flagging cohorts missing bootstrap CIs, and
cohorts that share the same underlying data suffix (same processed-array
build) but were produced by different runs and disagree on results - e.g. the
documented primary_705_vr060 vs primary_705_vr060_gated_ab mismatch in
CLAUDE.md, which predates the GPU-determinism fix and must not be treated as
directly comparable.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

CI_METRIC = "roc_auc"
FLAT_MODEL_METRICS = ["roc_auc", "pr_auc", "recall_sensitivity", "specificity", "precision", "f1"]
EARLY_WINDOW_NAME_RE = re.compile(r"early(\d+)")
ROC_AUC_MATCH_TOLERANCE = 1e-9


def find_cohort_dirs(cohorts_dir: Path) -> List[Path]:
    return sorted({p.parent for p in cohorts_dir.glob("*/cv_summary.json")})


def load_ci_lookup(cohort_dir: Path) -> Dict[str, Dict[str, float]]:
    ci_path = cohort_dir / "bootstrap_confidence_intervals.csv"
    if not ci_path.exists():
        return {}
    ci = pd.read_csv(ci_path)
    ci = ci[ci["metric"] == CI_METRIC]
    return {
        row["model"]: {"ci_low": row["ci_low"], "ci_high": row["ci_high"]}
        for _, row in ci.iterrows()
    }


def infer_window_info(cohort_name: str, config_snapshot_path: Path) -> Tuple[Optional[int], Optional[str]]:
    """Prefer the exact values recorded at build time; fall back to name parsing."""
    if config_snapshot_path.exists():
        try:
            snapshot = json.loads(config_snapshot_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            snapshot = {}
        cohort_cfg = snapshot.get("cohort") or {}
        if "window_minutes" in cohort_cfg:
            return cohort_cfg["window_minutes"], cohort_cfg.get("window_position")
        signals = (snapshot.get("base_config") or {}).get("signals") or {}
        if "window_minutes" in signals:
            return signals["window_minutes"], signals.get("window_position")

    match = EARLY_WINDOW_NAME_RE.search(cohort_name)
    if match:
        return int(match.group(1)), "first"
    return 30, "last"


def build_cohort_entry(cohort_dir: Path) -> Dict:
    summary = json.loads((cohort_dir / "cv_summary.json").read_text(encoding="utf-8"))
    ci_lookup = load_ci_lookup(cohort_dir)
    window_minutes, window_position = infer_window_info(
        summary["cohort_name"], cohort_dir / "config_snapshot.json"
    )
    last_modified = datetime.fromtimestamp(cohort_dir.stat().st_mtime).isoformat()

    models: Dict[str, Dict] = {}
    for model, metrics in summary["model_metrics_from_averaged_oof"].items():
        model_entry = dict(metrics)
        ci = ci_lookup.get(model, {})
        model_entry["roc_auc_ci_low"] = ci.get("ci_low")
        model_entry["roc_auc_ci_high"] = ci.get("ci_high")
        models[model] = model_entry

    return {
        "cohort_name": summary["cohort_name"],
        "cohort_dir": str(cohort_dir),
        "suffix": summary.get("suffix"),
        "valid_ratio_threshold": summary.get("valid_ratio_threshold"),
        "label": summary.get("label"),
        "cohort_size": summary.get("cohort_size"),
        "positive_count": summary.get("positive_count"),
        "window_minutes": window_minutes,
        "window_position": window_position,
        "last_modified": last_modified,
        "has_bootstrap_ci": bool(ci_lookup),
        "models": models,
    }


def flatten_rows(entries: List[Dict]) -> pd.DataFrame:
    rows = []
    for entry in entries:
        for model, metrics in entry["models"].items():
            row = {
                "cohort_name": entry["cohort_name"],
                "window_position": entry["window_position"],
                "window_minutes": entry["window_minutes"],
                "valid_ratio_threshold": entry["valid_ratio_threshold"],
                "n": entry["cohort_size"],
                "n_positive": entry["positive_count"],
                "model": model,
                "roc_auc": metrics.get("roc_auc"),
                "roc_auc_ci_low": metrics.get("roc_auc_ci_low"),
                "roc_auc_ci_high": metrics.get("roc_auc_ci_high"),
                "pr_auc": metrics.get("pr_auc"),
                "recall_sensitivity": metrics.get("recall_sensitivity"),
                "specificity": metrics.get("specificity"),
                "precision": metrics.get("precision"),
                "f1": metrics.get("f1"),
                "last_modified": entry["last_modified"],
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(["window_minutes", "model"]).reset_index(drop=True)


def report_suffix_mismatches(entries: List[Dict]) -> List[str]:
    """Group cohorts sharing a data suffix (same underlying processed-array build)
    and flag cases where shared models disagree on roc_auc despite the shared
    lineage - a sign the two cohorts came from different runs and should not be
    silently treated as comparable (see CLAUDE.md's documented vr060/gated_ab case).
    """
    warnings: List[str] = []
    by_suffix: Dict[str, List[Dict]] = {}
    for entry in entries:
        if entry["suffix"] is None:
            continue
        by_suffix.setdefault(entry["suffix"], []).append(entry)

    for suffix, group in sorted(by_suffix.items()):
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                shared_models = set(a["models"]) & set(b["models"])
                mismatched_models = []
                for model in sorted(shared_models):
                    roc_a = a["models"][model].get("roc_auc")
                    roc_b = b["models"][model].get("roc_auc")
                    if roc_a is None or roc_b is None:
                        continue
                    if abs(roc_a - roc_b) > ROC_AUC_MATCH_TOLERANCE:
                        mismatched_models.append((model, roc_a, roc_b))
                if mismatched_models:
                    warnings.append(
                        f"  STALE/OUT-OF-SYNC: '{a['cohort_name']}' (mtime {a['last_modified']}) vs "
                        f"'{b['cohort_name']}' (mtime {b['last_modified']}) share suffix '{suffix}' "
                        f"but disagree: "
                        + "; ".join(f"{m} roc_auc {ra:.6f} vs {rb:.6f}" for m, ra, rb in mismatched_models)
                        + " -- do not merge these as comparable results."
                    )
                else:
                    warnings.append(
                        f"  OK (reproducible): '{a['cohort_name']}' (mtime {a['last_modified']}) vs "
                        f"'{b['cohort_name']}' (mtime {b['last_modified']}) share suffix '{suffix}' "
                        f"and shared-model roc_auc values match exactly."
                    )
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a cross-cohort results summary from outputs/quality_cohorts/*/cv_summary.json."
    )
    parser.add_argument("--cohorts-dir", default="outputs/quality_cohorts")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    cohorts_dir = Path(args.cohorts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cohort_dirs = find_cohort_dirs(cohorts_dir)
    entries = [build_cohort_entry(d) for d in cohort_dirs]

    json_path = output_dir / "results_summary.json"
    json_payload = {
        "generated_at": datetime.now().isoformat(),
        "cohorts_scanned": str(cohorts_dir),
        "cohorts": entries,
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    df = flatten_rows(entries)
    csv_path = output_dir / "results_summary.csv"
    df.to_csv(csv_path, index=False)

    missing_ci = [e["cohort_name"] for e in entries if not e["has_bootstrap_ci"]]
    mismatch_warnings = report_suffix_mismatches(entries)

    print(f"Scanned {cohorts_dir} -> {len(entries)} cohort(s) with cv_summary.json")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print()
    if missing_ci:
        print(f"Missing bootstrap_confidence_intervals.csv ({len(missing_ci)}):")
        for name in missing_ci:
            print(f"  - {name}")
    else:
        print("All cohorts have bootstrap_confidence_intervals.csv.")
    print()
    if mismatch_warnings:
        print("Cross-cohort lineage check (same data suffix, compared across cohorts):")
        for line in mismatch_warnings:
            print(line)
    else:
        print("No cohorts share a data suffix with another cohort - nothing to cross-check.")


if __name__ == "__main__":
    main()
