from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from src.data.read_wfdb import read_record_signals


def linear_interpolate_signal(x: np.ndarray, invalid_mask: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Replace invalid values by linear interpolation; fill all-invalid signals with fill_value."""
    x = x.astype(np.float32, copy=True)
    x[invalid_mask] = np.nan
    s = pd.Series(x)
    s = s.interpolate(method="linear", limit_direction="both")
    if s.isna().all():
        return np.full_like(x, fill_value, dtype=np.float32)
    return s.fillna(float(np.nanmedian(s.values))).to_numpy(dtype=np.float32)


def downsample_mean(x: np.ndarray, source_hz: int, target_hz: int) -> np.ndarray:
    if source_hz == target_hz:
        return x.astype(np.float32, copy=False)
    if source_hz % target_hz != 0:
        raise ValueError(f"source_hz must be divisible by target_hz. Got {source_hz}, {target_hz}")
    factor = source_hz // target_hz
    usable = (len(x) // factor) * factor
    x = x[:usable]
    return x.reshape(-1, factor).mean(axis=1).astype(np.float32)


def extract_last_window(x: np.ndarray, window_samples: int) -> Tuple[np.ndarray, int, int]:
    if len(x) >= window_samples:
        start = len(x) - window_samples
        end = len(x)
        return x[start:end], start, end
    pad_len = window_samples - len(x)
    padded = np.pad(x, (pad_len, 0), mode="constant", constant_values=0)
    return padded, -pad_len, len(x)


def extract_first_window(x: np.ndarray, window_samples: int) -> Tuple[np.ndarray, int, int]:
    """Extract the first window_samples of the recording (early-labor proxy).

    Note: CTU-UHB recordings carry no labor-onset timestamp; this is the
    first N minutes of the *recorded trace*, not first N minutes of labor.
    """
    if len(x) >= window_samples:
        return x[:window_samples], 0, window_samples
    pad_len = window_samples - len(x)
    padded = np.pad(x, (0, pad_len), mode="constant", constant_values=0)
    return padded, 0, len(x)


def extract_window(x: np.ndarray, window_samples: int, window_position: str) -> Tuple[np.ndarray, int, int]:
    if window_position == "last":
        return extract_last_window(x, window_samples)
    if window_position == "first":
        return extract_first_window(x, window_samples)
    raise ValueError(f"Unknown window_position '{window_position}'. Expected 'first' or 'last'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build raw CTG windows for deep learning.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--label", default=None, help="Label column, e.g. y_acidemia_705 or y_acidemia_710")
    parser.add_argument("--min-valid-ratio", type=float, default=None, help="Override minimum valid FHR ratio.")
    parser.add_argument("--window-minutes", type=int, default=None, help="Override window length in minutes, e.g. 10/15/20 for early windows.")
    parser.add_argument("--window-position", choices=["first", "last"], default=None, help="Override window position: 'first' N minutes of the trace (early proxy) or 'last' N minutes before delivery.")
    parser.add_argument("--suffix", default="", help="Optional suffix for output files, e.g. _705_vr050.")
    parser.add_argument("--cohort-run", action="store_true", help="Enable safety checks and cohort manifest output.")
    parser.add_argument("--cohort-name", default=None, help="Required with --cohort-run, e.g. primary_705_vr050.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing cohort output files.")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    label_col = args.label or cfg["labels"]["primary_name"]
    source_hz = int(cfg["signals"]["source_sample_rate_hz"])
    target_hz = int(cfg["signals"]["target_sample_rate_hz"])
    window_minutes = int(args.window_minutes) if args.window_minutes is not None else int(cfg["signals"]["window_minutes"])
    window_position = str(args.window_position) if args.window_position is not None else str(cfg["signals"]["window_position"])
    window_samples_source = window_minutes * 60 * source_hz
    expected_steps = window_minutes * 60 * target_hz
    min_valid_ratio = (
        float(args.min_valid_ratio)
        if args.min_valid_ratio is not None
        else float(cfg["signals"]["min_fhr_valid_ratio"])
    )

    fhr_ch = int(cfg["signals"].get("fhr_channel", 0))
    uc_ch = int(cfg["signals"].get("uc_channel", 1))
    maternal_cols = list(cfg["maternal_features"]["columns"])
    suffix = str(args.suffix)

    if args.cohort_run:
        if not suffix:
            raise ValueError("--cohort-run requires a non-empty --suffix to avoid unsuffixed processed arrays.")
        if not args.cohort_name:
            raise ValueError("--cohort-run requires --cohort-name.")
        planned_outputs = [
            processed_dir / f"X_ctg{suffix}.npy",
            processed_dir / f"X_maternal{suffix}.npy",
            processed_dir / f"y{suffix}.npy",
            processed_dir / f"record_ids{suffix}.csv",
            processed_dir / f"maternal_feature_names{suffix}.json",
            processed_dir / f"window_build_summary{suffix}.json",
            processed_dir / f"cohort_manifest{suffix}.csv",
        ]
        existing = [str(path) for path in planned_outputs if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(
                "Cohort output files already exist. Re-run with --overwrite only if intentional: "
                + "; ".join(existing)
            )

    labels_path = processed_dir / "record_meta_labels_acidemia.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}. Run build_labels.py first.")

    meta = pd.read_csv(labels_path)
    meta["record_id"] = meta["record_id"].astype(int).astype(str)
    meta = meta.dropna(subset=[label_col]).copy()
    meta[label_col] = meta[label_col].astype(int)

    X_ctg: List[np.ndarray] = []
    X_maternal: List[np.ndarray] = []
    y: List[int] = []
    record_rows: List[Dict[str, object]] = []
    cohort_manifest_rows: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []

    for _, row in meta.iterrows():
        rid = str(row["record_id"])
        try:

            fhr, uc, fs = read_record_signals(raw_dir, rid, fhr_ch, uc_ch)
            if not np.isfinite(fs) or int(round(fs)) != source_hz:
                # CTU-UHB is expected to be 4 Hz. If different, use configured source rate but log it.
                pass

            fhr_win_raw, start, end = extract_window(fhr, window_samples_source, window_position)
            uc_win_raw, _, _ = extract_window(uc, window_samples_source, window_position)

            fhr_valid_mask = np.isfinite(fhr_win_raw) & (fhr_win_raw > 0)
            valid_ratio = float(fhr_valid_mask.mean())
            if valid_ratio < min_valid_ratio:
                failures.append({"record_id": rid, "reason": f"low_fhr_valid_ratio={valid_ratio:.3f}"})
                if args.cohort_run:
                    cohort_manifest_rows.append({
                        "record_id": rid,
                        "label": int(row[label_col]),
                        "ph": float(row["ph"]) if pd.notna(row.get("ph")) else np.nan,
                        "valid_ratio": valid_ratio,
                        "inclusion_status": "excluded",
                        "exclusion_reason": f"low_fhr_valid_ratio={valid_ratio:.3f}",
                        "cohort_name": args.cohort_name,
                        "threshold": min_valid_ratio,
                    })
                continue

            # FHR invalid: <=0 or non-finite. UC invalid: non-finite only.
            fhr_clean = linear_interpolate_signal(fhr_win_raw, ~fhr_valid_mask, fill_value=0.0)
            uc_clean = linear_interpolate_signal(uc_win_raw, ~np.isfinite(uc_win_raw), fill_value=0.0)

            fhr_ds = downsample_mean(fhr_clean, source_hz, target_hz)
            uc_ds = downsample_mean(uc_clean, source_hz, target_hz)

            if len(fhr_ds) != expected_steps or len(uc_ds) != expected_steps:
                failures.append({"record_id": rid, "reason": "unexpected_downsampled_length"})
                continue

            ctg = np.stack([fhr_ds, uc_ds], axis=-1).astype(np.float32)
            maternal = pd.to_numeric(row.reindex(maternal_cols), errors="coerce").to_numpy(dtype=np.float32)

            X_ctg.append(ctg)
            X_maternal.append(maternal)
            y.append(int(row[label_col]))
            record_rows.append({
                "row_index": len(y) - 1,
                "record_id": rid,
                "label": int(row[label_col]),
                "ph": float(row["ph"]) if pd.notna(row.get("ph")) else np.nan,
                "valid_ratio": valid_ratio,
                "window_start_source_index": int(start),
                "window_end_source_index": int(end),
                "recording_length_source_samples": int(len(fhr)),
                "window_minutes": window_minutes,
                "window_position": window_position,
                "target_sample_rate_hz": target_hz,
            })
            if args.cohort_run:
                cohort_manifest_rows.append({
                    "record_id": rid,
                    "label": int(row[label_col]),
                    "ph": float(row["ph"]) if pd.notna(row.get("ph")) else np.nan,
                    "valid_ratio": valid_ratio,
                    "inclusion_status": "included",
                    "exclusion_reason": "",
                    "cohort_name": args.cohort_name,
                    "threshold": min_valid_ratio,
                })
        except Exception as e:
            failures.append({"record_id": rid, "reason": repr(e)})
            if args.cohort_run:
                cohort_manifest_rows.append({
                    "record_id": rid,
                    "label": int(row[label_col]),
                    "ph": float(row["ph"]) if pd.notna(row.get("ph")) else np.nan,
                    "valid_ratio": np.nan,
                    "inclusion_status": "excluded",
                    "exclusion_reason": repr(e),
                    "cohort_name": args.cohort_name,
                    "threshold": min_valid_ratio,
                })

    if not X_ctg:
        raise RuntimeError("No CTG windows were built. Check labels, raw path, and min_valid_ratio.")

    X_ctg_arr = np.stack(X_ctg).astype(np.float32)
    X_mat_arr = np.stack(X_maternal).astype(np.float32)
    y_arr = np.asarray(y, dtype=np.int64)

    np.save(processed_dir / f"X_ctg{suffix}.npy", X_ctg_arr)
    np.save(processed_dir / f"X_maternal{suffix}.npy", X_mat_arr)
    np.save(processed_dir / f"y{suffix}.npy", y_arr)
    pd.DataFrame(record_rows).to_csv(processed_dir / f"record_ids{suffix}.csv", index=False)
    if args.cohort_run:
        pd.DataFrame(cohort_manifest_rows).to_csv(processed_dir / f"cohort_manifest{suffix}.csv", index=False)
    (processed_dir / f"maternal_feature_names{suffix}.json").write_text(json.dumps(maternal_cols, indent=2), encoding="utf-8")

    summary = {
        "label_col": label_col,
        "X_ctg_shape": list(X_ctg_arr.shape),
        "X_maternal_shape": list(X_mat_arr.shape),
        "y_shape": list(y_arr.shape),
        "positives": int(y_arr.sum()),
        "negatives": int((y_arr == 0).sum()),
        "window_minutes": window_minutes,
        "window_position": window_position,
        "source_sample_rate_hz": source_hz,
        "target_sample_rate_hz": target_hz,
        "min_fhr_valid_ratio": min_valid_ratio,
        "suffix": suffix,
        "failures_count": len(failures),
        "failures_first_20": failures[:20],
    }
    (processed_dir / f"window_build_summary{suffix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
