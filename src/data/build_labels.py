from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from src.data.read_wfdb import load_record_ids


def normalize_key(raw_key: str) -> str:
    k = raw_key.strip().lower().strip(" :\t")
    k = k.replace("/", "_").replace("-", "_").replace(".", "")
    k = k.replace("(", "_").replace(")", "")
    k = re.sub(r"\s+", "_", k)
    k = re.sub(r"_+", "_", k).strip("_")
    return k


def parse_value(raw_val: str) -> Any:
    v = raw_val.strip()
    if v == "":
        return None
    if re.fullmatch(r"[-+]?\d+", v):
        return int(v)
    if re.fullmatch(r"[-+]?\d*\.\d+|[-+]?\d+\.\d*", v):
        return float(v)
    return v


def parse_hea_to_kv(header_text: str) -> Dict[str, Any]:
    kv: Dict[str, Any] = {}
    for line in header_text.splitlines():
        if not line.startswith("#"):
            continue
        content = line.lstrip("#").strip()
        if not content:
            continue
        parts = re.split(r"\s{2,}", content, maxsplit=1)
        if len(parts) != 2:
            parts = re.split(r"\s+", content, maxsplit=1)
        if len(parts) != 2:
            continue
        raw_key, raw_val = parts
        kv[normalize_key(raw_key)] = parse_value(raw_val)
    return kv


def build_acidemia_label(df: pd.DataFrame, positive_ph_lt: float = 7.05, normal_ph_gte: float = 7.15) -> pd.DataFrame:
    """Primary thesis label: pH < 7.05 positive, pH >= 7.15 negative, middle excluded."""
    df = df.copy()
    df["label"] = np.nan
    df.loc[df["ph"] < positive_ph_lt, "label"] = 1
    df.loc[df["ph"] >= normal_ph_gte, "label"] = 0
    return df.dropna(subset=["label"]).copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pH-based acidemia labels and safe maternal table.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    primary_name = cfg["labels"]["primary_name"]
    primary_thr = float(cfg["labels"]["primary_positive_ph_lt"])
    normal_thr = float(cfg["labels"]["normal_ph_gte"])
    secondary_name = cfg["labels"]["secondary_name"]
    secondary_thr = float(cfg["labels"]["secondary_positive_ph_lt"])

    record_ids = load_record_ids(raw_dir)
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    wanted_keys = [
        "ph", "apgar1", "apgar5", "be", "gest_weeks", "weight_g", "weightg", "sex",
        "age", "gravidity", "parity", "diabetes", "hypertension", "preeclampsia",
        "liq_praecox", "pyrexia", "meconium", "presentation", "induced",
        "istage", "i_stage", "noprogress", "no_progress", "ck_kp", "iistage", "ii_stage", "deliv_type",
    ]

    for rid in record_ids:
        hea_path = raw_dir / f"{rid}.hea"
        if not hea_path.exists():
            failures.append({"record_id": rid, "reason": "missing_hea"})
            continue
        kv = parse_hea_to_kv(hea_path.read_text(encoding="utf-8", errors="ignore"))
        row = {"record_id": rid}
        for key in wanted_keys:
            row[key] = kv.get(key)
        row["weight_g"] = row.get("weight_g") if row.get("weight_g") is not None else row.get("weightg")
        row["i_stage"] = row.get("i_stage") if row.get("i_stage") is not None else row.get("istage")
        row["ii_stage"] = row.get("ii_stage") if row.get("ii_stage") is not None else row.get("iistage")
        row["no_progress"] = row.get("no_progress") if row.get("no_progress") is not None else row.get("noprogress")
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in ["ph", "age", "gravidity", "parity", "gest_weeks", "diabetes", "hypertension", "preeclampsia", "induced", "apgar1", "apgar5", "be", "weight_g"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    #The primary label is built here:
 
    df[primary_name] = np.nan
    df.loc[df["ph"] < primary_thr, primary_name] = 1
    df.loc[df["ph"] >= normal_thr, primary_name] = 0

    df[secondary_name] = np.nan
    df.loc[df["ph"] < secondary_thr, secondary_name] = 1
    df.loc[df["ph"] >= normal_thr, secondary_name] = 0

    out_csv = processed_dir / "record_meta_labels_acidemia.csv"
    df.to_csv(out_csv, index=False)

    primary_df = df.dropna(subset=[primary_name])
    secondary_df = df.dropna(subset=[secondary_name])
    summary = {
        "records_total_in_RECORDS": len(record_ids),
        "rows_written": int(len(df)),
        "primary_label": primary_name,
        "primary_positive_ph_lt": primary_thr,
        "normal_ph_gte": normal_thr,
        "primary_usable_records": int(len(primary_df)),
        "primary_positive_records": int(primary_df[primary_name].sum()) if len(primary_df) else 0,
        "secondary_label": secondary_name,
        "secondary_positive_ph_lt": secondary_thr,
        "secondary_usable_records": int(len(secondary_df)),
        "secondary_positive_records": int(secondary_df[secondary_name].sum()) if len(secondary_df) else 0,
        "failures": failures,
        "output": str(out_csv),
    }
    (processed_dir / "labels_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
