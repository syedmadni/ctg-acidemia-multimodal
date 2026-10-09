"""Task A.6 (approved, DEVIATIONS A2): reduced-encoder ctg_only + concat fusion, 5 seeds,
causal inputs, trained on the `unfiltered` cohort (487/40), evaluated at every checkpoint.

Reuses train_cv.train_deep_fold_anytime unchanged (duration-aware cutoffs, per-fold
preprocessing, val-PR-AUC checkpoint selection). New here: causal inputs, splits read from
disk, and VALIDATION-split predictions at each checkpoint are saved for Task C thresholds.
Resumable: one progress file per (seed, model, fold).

    python -m src.revision.run_deep_revision            # all seeds
    python -m src.revision.run_deep_revision --seeds 42 # a subset
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.revision.causal_features import causal_deep_input
from src.revision.cohorts import labelled_meta, load_splits, make_and_save_splits, members, quality_table, split_path
from src.revision.raw_cache import RawData
from src.training.train_cv import (
    CTGMaternalDataset, choose_device, crop_ctg_array, predict_model,
    train_deep_fold_anytime, transform_arrays,
)

SEEDS = [42, 43, 44, 45, 46]
MODELS = ["ctg_only", "fusion"]
CHECKPOINTS = [10.0, 20.0, 30.0, 45.0, 60.0]
ENCODER = {"d_model": 64, "transformer_layers": 1, "n_heads": 4, "dropout": 0.3}  # = step7d_cap64
OUT = Path("results/revision/deep")
PROGRESS = OUT / "progress"
OOF_OUT = Path("results/revision/oof/deep_oof.csv")
VAL_OUT = Path("results/revision/valpred/deep_valpred.csv")


def build_inputs():
    raw = RawData()
    meta = labelled_meta(raw)
    q = quality_table(raw)
    coh = members(meta, q, "unfiltered", 60.0)  # min length is 60, so identical at every t
    if not split_path("unfiltered", 60.0).exists():
        make_and_save_splits(coh, "unfiltered", 60.0)
    splits = load_splits(coh, "unfiltered", 60.0)
    X_ctg = np.stack([causal_deep_input(raw.fhr(r), raw.uc(r)) for r in coh["record_id"]])
    X_mat = coh[["age", "gravidity", "parity", "gest_weeks", "diabetes", "hypertension",
                 "preeclampsia", "induced"]].to_numpy(dtype=np.float32)
    return coh, splits, X_ctg.astype(np.float32), X_mat, coh["y"].to_numpy(int), coh["length_min"].to_numpy(float)


def predict_at(model_name, model, Xc_norm, Xm_norm, y, idx, t, device, bs):
    loader = DataLoader(CTGMaternalDataset(crop_ctg_array(Xc_norm[idx], t, 1), Xm_norm[idx], y[idx]),
                        batch_size=bs, shuffle=False)
    return predict_model(model_name, model, loader, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    base_cfg = yaml.safe_load(Path("configs/baseline.yaml").read_text(encoding="utf-8"))
    coh, splits, X_ctg, X_mat, y, dur = build_inputs()
    rid = coh["record_id"].to_numpy()
    device = choose_device(str(base_cfg["training"]["device"]))
    bs = int(base_cfg["training"]["batch_size"])
    PROGRESS.mkdir(parents=True, exist_ok=True)
    print(f"unfiltered cohort n={len(y)} cases={int(y.sum())}; X_ctg {X_ctg.shape}; device={device}", flush=True)

    for seed in seeds:
        for sp in splits:
            for model_name in MODELS:
                tag = f"seed{seed}_{model_name}_{sp['fold_name']}"
                pf = PROGRESS / f"{tag}.json"
                if pf.exists():
                    continue
                t0 = time.time()
                art = OUT / "_tmp" / tag
                model, info, imp, sc, cm, cs = train_deep_fold_anytime(
                    model_name, X_ctg, X_mat, y, sp, base_cfg, seed, art,
                    target_sample_rate_hz=1, min_crop_minutes=8.0, max_crop_minutes=90.0,
                    deep_overrides={"encoder": ENCODER}, duration_minutes=dur,
                )
                Xc, Xm = transform_arrays(X_ctg, X_mat, imp, sc, cm, cs)
                rows = []
                for t in CHECKPOINTS:
                    for role, key in (("outer_test", "outer_test_idx"), ("val", "val_idx")):
                        idx = np.asarray(sp[key], dtype=int)
                        p = predict_at(model_name, model, Xc, Xm, y, idx, t, device, bs)
                        rows += [{"seed": seed, "model": model_name, "repeat": sp["repeat"], "fold": sp["fold"],
                                  "role": role, "checkpoint_minutes": t, "record_id": rid[i],
                                  "y_true": int(y[i]), "y_proba": float(pv)} for i, pv in zip(idx, p)]
                pf.write_text(json.dumps({"rows": rows, "best_epoch": info["best_epoch"],
                                          "best_val_pr_auc": info["best_val_pr_auc"],
                                          "seconds": round(time.time() - t0, 1)}), encoding="utf-8")
                shutil.rmtree(art, ignore_errors=True)
                del model
                torch.cuda.empty_cache()
                print(f"[done] {tag} {time.time() - t0:.0f}s best_epoch={info['best_epoch']}", flush=True)

    # combine everything on disk (any seeds finished so far)
    rows = []
    for pf in sorted(PROGRESS.glob("*.json")):
        rows += json.loads(pf.read_text(encoding="utf-8"))["rows"]
    df = pd.DataFrame(rows)
    OOF_OUT.parent.mkdir(parents=True, exist_ok=True)
    VAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    df[df.role == "outer_test"].drop(columns="role").to_csv(OOF_OUT, index=False)
    df[df.role == "val"].drop(columns="role").to_csv(VAL_OUT, index=False)
    print(f"wrote {OOF_OUT} and {VAL_OUT}: {len(df)} rows", flush=True)


if __name__ == "__main__":
    main()
