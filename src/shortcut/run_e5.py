"""Pre-registered experiment E5 (results/shortcut_e5/PREREGISTRATION_E5.md): training and prediction.
Statistics are computed afterwards by src/shortcut/analyze_e5.py from the files written here.

Train on F5 (plain end marker), evaluate the same networks on F5 and F5x. Everything is as in E2
(src/shortcut/run_shortcut.py, imported, not copied) except two things (plan section 4): the input
format, and early stopping on the lowest validation loss instead of the highest validation PR-AUC.
Cohort: all 487 labelled recordings; folds read from results/revision/splits/splits_unfiltered_t60.csv
(never regenerated); reduced encoder, CTG-only head; seeds 42-46; no tuning; no occlusion.

    python -m src.shortcut.run_e5            # full run (resumable)
    python -m src.shortcut.run_e5 --smoke    # 1 seed, 2 folds, 2 epochs, small subsets (pipeline check only)

Outputs (results/shortcut_e5/):
  cohort.csv          record_id, y, length_min, in_primary60 (asserted equal to results/shortcut/cohort.csv)
  progress/*.json     one file per (seed, fold): predictions and training info
  predictions.csv     experiment, train_format, eval_format, seed, repeat, fold, role, record_id, y_true, y_proba
  training_log.csv    best_epoch, best_val_loss, val_pr_auc_at_best_epoch, pos_weight per (seed, fold)
  environment.json    versions, device, git commit
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.ctg_cnn import CTGOnlyModel
from src.shortcut import inputs_e5 as F5
from src.shortcut.run_shortcut import ENCODER, SEEDS, build_cohort, predict
from src.utils.seed import set_global_seed

OUT = Path("results/shortcut_e5")
E2_COHORT = Path("results/shortcut/cohort.csv")
EXPERIMENT = "E5"
TRAIN_FORMAT = "F5"
EVAL_FORMATS = ["F5", "F5x"]


def build_format(raw, rids, name):
    if name == "F5":
        return np.stack([F5.f5_plain_end(raw.fhr(r), raw.uc(r)) for r in rids])
    if name == "F5x":
        return np.stack([F5.f5_end_marker_removed(F5.f5_plain_end(raw.fhr(r), raw.uc(r)), len(raw.fhr(r)))
                         for r in rids])
    raise ValueError(name)


def check_cohort(coh: pd.DataFrame):
    """The E5 cohort must equal the E1-E4 cohort: same record IDs (same order), outcomes, lengths."""
    ref = pd.read_csv(E2_COHORT, dtype={"record_id": str}, float_precision="round_trip")
    new = coh[["record_id", "y", "length_min", "in_primary60"]].reset_index(drop=True)
    assert len(new) == len(ref), f"cohort size {len(new)} != {len(ref)}"
    assert (new.record_id.to_numpy() == ref.record_id.to_numpy()).all(), "record IDs differ from E1-E4"
    assert (new.y.to_numpy() == ref.y.to_numpy()).all(), "outcomes differ from E1-E4"
    assert (new.length_min.to_numpy(float) == ref.length_min.to_numpy(float)).all(), "lengths differ from E1-E4"
    assert (new.in_primary60.to_numpy() == ref.in_primary60.to_numpy()).all(), "primary60 flags differ from E1-E4"


def train_fold_valloss(Xtr, ytr, Xva, yva, cfg, seed, device):
    """As run_shortcut.train_fold, except early stopping: lowest validation loss (the same class-weighted
    BCEWithLogitsLoss as in training, mean reduction, eval mode); the first epoch wins a tie; the weights
    from the best epoch are restored. Validation PR-AUC at that epoch is logged only."""
    pos, neg = int(ytr.sum()), int((ytr == 0).sum())
    model = CTGOnlyModel(input_channels=Xtr.shape[-1], **ENCODER).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]),
                            weight_decay=float(cfg["training"]["weight_decay"]))
    loader = DataLoader(TensorDataset(torch.as_tensor(Xtr), torch.as_tensor(ytr, dtype=torch.float32)),
                        batch_size=int(cfg["training"]["batch_size"]), shuffle=True)
    yva_t = torch.as_tensor(yva, dtype=torch.float32, device=device)
    best, best_ep, bad, best_state, best_pr = np.inf, 0, 0, None, float("nan")
    for ep in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        logits = predict(model, Xva, device)            # model.eval() and no_grad inside
        with torch.no_grad():
            loss = float(crit(torch.as_tensor(logits, dtype=torch.float32, device=device), yva_t))
        pv = 1 / (1 + np.exp(-logits))
        pr = float(average_precision_score(yva, pv)) if len(np.unique(yva)) == 2 else float("nan")
        if best_ep == 0 or loss < best:                  # strictly lower: the first epoch wins a tie
            best, best_ep, bad, best_state, best_pr = loss, ep, 0, copy.deepcopy(model.state_dict()), pr
        else:
            bad += 1
            if bad >= int(cfg["training"]["early_stopping_patience"]):
                break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_ep, "best_val_loss": best, "val_pr_auc_at_best_epoch": best_pr,
                   "pos_weight": neg / max(pos, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--smoke", action="store_true", help="1 seed, 2 folds, 2 epochs, into results/shortcut_e5_smoke")
    args = ap.parse_args()
    global OUT
    cfg = yaml.safe_load(Path("configs/baseline.yaml").read_text(encoding="utf-8"))
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.smoke:
        OUT = Path("results/shortcut_e5_smoke")
        cfg["training"]["epochs"] = 2
        seeds = seeds[:1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (OUT / "progress").mkdir(parents=True, exist_ok=True)

    raw, coh, splits = build_cohort()
    check_cohort(coh)
    if args.smoke:
        splits = splits[:2]
    rid = coh["record_id"].to_numpy()
    y = coh["y"].to_numpy(int)
    coh[["record_id", "y", "length_min", "in_primary60"]].to_csv(OUT / "cohort.csv", index=False)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "unknown"
    (OUT / "environment.json").write_text(json.dumps({
        "python": platform.python_version(), "torch": torch.__version__, "device": str(device),
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": commit, "n": int(len(y)), "cases": int(y.sum()), "seeds": seeds,
        "smoke": bool(args.smoke)}, indent=1), encoding="utf-8")
    print(f"cohort n={len(y)} cases={int(y.sum())} device={device} commit={commit}", flush=True)

    X = {name: build_format(raw, rid, name) for name in EVAL_FORMATS}
    Xtrain = X[TRAIN_FORMAT]
    for seed in seeds:
        for sp in splits:
            tag = f"{EXPERIMENT}_seed{seed}_{sp['fold_name']}"
            pf = OUT / "progress" / f"{tag}.json"
            if pf.exists():
                continue
            t0 = time.time()
            set_global_seed(seed + int(sp["repeat"]) * 100 + int(sp["fold"]))
            tr, va, te = (np.asarray(sp[k], dtype=int) for k in ("inner_train_idx", "val_idx", "outer_test_idx"))
            if args.smoke:  # pipeline check only: small subsets so it runs on a CPU (as in run_shortcut.py)
                tr, va, te = tr[:64], np.concatenate([va[y[va] == 1][:4], va[y[va] == 0][:28]]), te[:24]
            mu = Xtrain[tr].reshape(-1, Xtrain.shape[-1]).mean(axis=0)
            sd = Xtrain[tr].reshape(-1, Xtrain.shape[-1]).std(axis=0)
            sd = np.where(sd < 1e-6, 1.0, sd)
            norm = lambda A: ((A - mu) / sd).astype(np.float32)
            Xn = norm(Xtrain)
            model, info = train_fold_valloss(Xn[tr], y[tr], Xn[va], y[va], cfg, seed, device)
            rows = []
            for name in EVAL_FORMATS:
                Xe = norm(X[name])
                for role, idx in (("outer_test", te), ("val", va)):
                    p = 1 / (1 + np.exp(-predict(model, Xe[idx], device)))
                    rows += [{"experiment": EXPERIMENT, "train_format": TRAIN_FORMAT, "eval_format": name,
                              "seed": seed, "repeat": int(sp["repeat"]), "fold": int(sp["fold"]), "role": role,
                              "record_id": rid[i], "y_true": int(y[i]), "y_proba": float(v)} for i, v in zip(idx, p)]
            pf.write_text(json.dumps({"rows": rows,
                                      "info": {**info, "experiment": EXPERIMENT, "seed": seed,
                                               "fold_name": sp["fold_name"], "seconds": round(time.time() - t0, 1)}}),
                          encoding="utf-8")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[done] {tag} {time.time() - t0:.0f}s best_epoch={info['best_epoch']} "
                  f"val_loss={info['best_val_loss']:.4f}", flush=True)

    preds, logs = [], []
    for pf in sorted((OUT / "progress").glob("*.json")):
        d = json.loads(pf.read_text(encoding="utf-8"))
        preds += d["rows"]
        logs.append(d["info"])
    pd.DataFrame(preds).to_csv(OUT / "predictions.csv", index=False)
    pd.DataFrame(logs).to_csv(OUT / "training_log.csv", index=False)
    print(f"wrote {OUT}/predictions.csv ({len(preds)} rows)", flush=True)


if __name__ == "__main__":
    main()
