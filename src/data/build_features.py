from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from src.data.read_wfdb import read_record_signals


def safe_stats(x: np.ndarray) -> Dict[str, float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "median": np.nan, "range": np.nan}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "range": float(np.max(x) - np.min(x)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build simple engineered CTG features for classical baselines.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    label_col = args.label or cfg["labels"]["primary_name"]
    meta = pd.read_csv(processed_dir / "record_meta_labels_acidemia.csv").dropna(subset=[label_col]).copy()
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []

    window_samples = int(cfg["signals"]["window_minutes"]) * 60 * int(cfg["signals"]["source_sample_rate_hz"])
    min_valid_ratio = float(cfg["signals"]["min_fhr_valid_ratio"])

    for _, row in meta.iterrows():
        rid = str(row["record_id"])
        try:
            fhr, uc, _ = read_record_signals(raw_dir, rid, int(cfg["signals"].get("fhr_channel", 0)), int(cfg["signals"].get("uc_channel", 1)))
            fhr_win = fhr[-window_samples:] if len(fhr) >= window_samples else np.pad(fhr, (window_samples - len(fhr), 0), constant_values=0)
            uc_win = uc[-window_samples:] if len(uc) >= window_samples else np.pad(uc, (window_samples - len(uc), 0), constant_values=0)
            valid = np.isfinite(fhr_win) & (fhr_win > 0)
            valid_ratio = float(valid.mean())
            if valid_ratio < min_valid_ratio:
                continue
            fhr_stats = safe_stats(fhr_win[valid])
            uc_stats = safe_stats(uc_win[np.isfinite(uc_win)])
            out = {"record_id": rid, "label": int(row[label_col]), "valid_ratio": valid_ratio}
            out.update({f"fhr_{k}": v for k, v in fhr_stats.items()})
            out.update({f"uc_{k}": v for k, v in uc_stats.items()})
            for c in cfg["maternal_features"]["columns"]:
                out[c] = row.get(c, np.nan)
            rows.append(out)
        except Exception as e:
            failures.append({"record_id": rid, "reason": repr(e)})

    df = pd.DataFrame(rows)
    out_path = processed_dir / "engineered_features_acidemia.csv"
    df.to_csv(out_path, index=False)
    summary = {"rows": int(len(df)), "positives": int(df["label"].sum()) if len(df) else 0, "failures": failures[:20], "output": str(out_path)}
    (processed_dir / "engineered_features_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
