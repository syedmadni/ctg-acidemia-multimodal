"""Pre-registered shortcut experiments E1-E3 (training and prediction). E4 and all statistics
are computed afterwards by src/shortcut/analyze.py from the files written here.

Cohort: all 487 labelled recordings (`unfiltered`); folds read from
results/revision/splits/splits_unfiltered_t60.csv (never regenerated). Reduced encoder,
CTG-only head, revision training settings, seeds 42-46, no tuning, no random cropping.

    python -m src.shortcut.run_shortcut                     # everything (resumable)
    python -m src.shortcut.run_shortcut --experiments E1    # one experiment
    python -m src.shortcut.run_shortcut --smoke             # 1 seed, 2 folds, 2 epochs (pipeline check only)

Outputs (results/shortcut/):
  cohort.csv              record_id, y, length_min, in_primary60
  progress/*.json         one file per (experiment, seed, fold): predictions, occlusion, training info
  predictions.csv         experiment, train_format, eval_format, seed, repeat, fold, role, record_id, y_true, y_proba
  occlusion.csv           E2, seed 42 only: seed, repeat, fold, record_id, block, minute_start, length_min, abs_dlogit
  training_log.csv        best epoch and validation PR-AUC per (experiment, seed, fold)
  reference/              existing leak-free predictions copied from results/revision/oof (if present)
  environment.json        versions, device, git commit
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import shutil
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
from src.revision.cohorts import labelled_meta, load_splits, members, quality_table
from src.revision.raw_cache import RawData
from src.shortcut import inputs as F
from src.utils.seed import set_global_seed

OUT = Path("results/shortcut")
SEEDS = [42, 43, 44, 45, 46]
ENCODER = {"d_model": 64, "transformer_layers": 1, "n_heads": 4, "dropout": 0.3}  # = revision reduced encoder
EXPERIMENTS = {
    "E1": {"train": "F2", "evals": ["F2"]},
    "E2": {"train": "F1", "evals": ["F1", "F1x", "F1c60"], "occlusion": True},
    "E3": {"train": "F3", "evals": ["F3", "F3s"]},
}
BLOCK_SECONDS = 120


def build_cohort():
    raw = RawData()
    meta = labelled_meta(raw)
    q = quality_table(raw)
    coh = members(meta, q, "unfiltered", 60.0)
    prim = set(members(meta, q, "ckfilter", 60.0)["record_id"])
    coh = coh.copy()
    coh["in_primary60"] = coh["record_id"].isin(prim).astype(int)
    splits = load_splits(coh, "unfiltered", 60.0)
    return raw, coh, splits


def build_format(raw, rids, name):
    if name == "F1":
        return np.stack([F.f1_conventional(raw.fhr(r), raw.uc(r)) for r in rids])
    if name == "F1x":
        return np.stack([F.f1_end_marker_removed(F.f1_conventional(raw.fhr(r), raw.uc(r)), len(raw.fhr(r))) for r in rids])
    if name == "F1c60":
        return np.stack([F.f1_cut(raw.fhr(r), raw.uc(r), 60.0) for r in rids])
    if name == "F2":
        return np.stack([F.f2_envelope(len(raw.fhr(r)), r) for r in rids])
    if name == "F3":
        return np.stack([F.f3_delivery_anchored(raw.fhr(r), raw.uc(r)) for r in rids])
    if name == "F3s":
        return np.stack([F.f3_start_anchored(raw.fhr(r), raw.uc(r)) for r in rids])
    raise ValueError(name)


def predict(model, X, device, bs=64):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(torch.as_tensor(X[i:i + bs], device=device)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def occlusion(model, X, lengths_min, device):
    """|logit change| when each 2-min block is replaced by the recording's mean (recorded part)."""
    T = X.shape[1]
    nb = T // BLOCK_SECONDS
    base = predict(model, X, device)
    rows = []
    for i in range(len(X)):
        L = F.recorded_steps_1hz(int(round(lengths_min[i] * 240)), T / 60.0)
        mu = X[i, :max(L, 1)].mean(axis=0)
        batch = np.repeat(X[i:i + 1], nb, axis=0)
        for b in range(nb):
            batch[b, b * BLOCK_SECONDS:(b + 1) * BLOCK_SECONDS] = mu
        d = np.abs(predict(model, batch, device) - base[i])
        rows.append(d)
    return np.vstack(rows)  # [n, nb]


def train_fold(Xtr, ytr, Xva, yva, cfg, seed, device):
    pos, neg = int(ytr.sum()), int((ytr == 0).sum())
    model = CTGOnlyModel(input_channels=Xtr.shape[-1], **ENCODER).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]),
                            weight_decay=float(cfg["training"]["weight_decay"]))
    loader = DataLoader(TensorDataset(torch.as_tensor(Xtr), torch.as_tensor(ytr, dtype=torch.float32)),
                        batch_size=int(cfg["training"]["batch_size"]), shuffle=True)
    best, best_ep, bad, best_state = -np.inf, 0, 0, None
    for ep in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        pv = 1 / (1 + np.exp(-predict(model, Xva, device)))
        s = float(average_precision_score(yva, pv)) if len(np.unique(yva)) == 2 else -np.inf
        if best_ep == 0 or s > best:
            best, best_ep, bad, best_state = s, ep, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= int(cfg["training"]["early_stopping_patience"]):
                break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_ep, "best_val_pr_auc": best, "pos_weight": neg / max(pos, 1)}


def copy_reference(coh):
    """Existing leak-free OOF predictions (git-ignored) needed for E4."""
    ref = OUT / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    ids = set(coh["record_id"])
    deep = Path("results/revision/oof/deep_oof.csv")
    clas = Path("results/revision/oof/classical_oof.csv")
    if deep.exists():
        d = pd.read_csv(deep, dtype={"record_id": str})
        d[(d.checkpoint_minutes == 60.0) & d.record_id.isin(ids)].to_csv(ref / "deep_oof_t60.csv", index=False)
    if clas.exists():
        c = pd.read_csv(clas, dtype={"record_id": str})
        c[(c.checkpoint_minutes == 60.0) & (c.variant == "unfiltered")].to_csv(ref / "classical_oof_unfiltered_t60.csv", index=False)
    return deep.exists(), clas.exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", default="E1,E2,E3")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--smoke", action="store_true", help="1 seed, 2 folds, 2 epochs, into results/shortcut_smoke")
    args = ap.parse_args()
    global OUT
    cfg = yaml.safe_load(Path("configs/baseline.yaml").read_text(encoding="utf-8"))
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.smoke:
        OUT = Path("results/shortcut_smoke")
        cfg["training"]["epochs"] = 2
        seeds = seeds[:1]
    exps = [e.strip() for e in args.experiments.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (OUT / "progress").mkdir(parents=True, exist_ok=True)

    raw, coh, splits = build_cohort()
    if args.smoke:
        splits = splits[:2]
    rid = coh["record_id"].to_numpy()
    y = coh["y"].to_numpy(int)
    length_min = coh["length_min"].to_numpy(float)
    coh[["record_id", "y", "length_min", "in_primary60"]].to_csv(OUT / "cohort.csv", index=False)
    has_deep, has_clas = copy_reference(coh)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "unknown"
    (OUT / "environment.json").write_text(json.dumps({
        "python": platform.python_version(), "torch": torch.__version__, "device": str(device),
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": commit, "n": int(len(y)), "cases": int(y.sum()), "seeds": seeds,
        "reference_deep_oof_found": has_deep, "reference_classical_oof_found": has_clas,
        "smoke": bool(args.smoke)}, indent=1), encoding="utf-8")
    print(f"cohort n={len(y)} cases={int(y.sum())} device={device} commit={commit}", flush=True)

    cache = {}
    def fmt(name):
        if name not in cache:
            cache[name] = build_format(raw, rid, name)
        return cache[name]

    for exp in exps:
        spec = EXPERIMENTS[exp]
        Xtrain = fmt(spec["train"])
        for name in spec["evals"]:
            fmt(name)
        for seed in seeds:
            for sp in splits:
                tag = f"{exp}_seed{seed}_{sp['fold_name']}"
                pf = OUT / "progress" / f"{tag}.json"
                if pf.exists():
                    continue
                t0 = time.time()
                set_global_seed(seed + int(sp["repeat"]) * 100 + int(sp["fold"]))
                tr, va, te = (np.asarray(sp[k], dtype=int) for k in ("inner_train_idx", "val_idx", "outer_test_idx"))
                if args.smoke:  # pipeline check only: small subsets so it runs on a CPU
                    tr, va, te = tr[:64], np.concatenate([va[y[va] == 1][:4], va[y[va] == 0][:28]]), te[:24]
                mu = Xtrain[tr].reshape(-1, Xtrain.shape[-1]).mean(axis=0)
                sd = Xtrain[tr].reshape(-1, Xtrain.shape[-1]).std(axis=0)
                sd = np.where(sd < 1e-6, 1.0, sd)
                norm = lambda X: ((X - mu) / sd).astype(np.float32)
                Xn = norm(Xtrain)
                model, info = train_fold(Xn[tr], y[tr], Xn[va], y[va], cfg, seed, device)
                rows = []
                for name in spec["evals"]:
                    Xe = norm(fmt(name))
                    for role, idx in (("outer_test", te), ("val", va)):
                        p = 1 / (1 + np.exp(-predict(model, Xe[idx], device)))
                        rows += [{"experiment": exp, "train_format": spec["train"], "eval_format": name,
                                  "seed": seed, "repeat": int(sp["repeat"]), "fold": int(sp["fold"]), "role": role,
                                  "record_id": rid[i], "y_true": int(y[i]), "y_proba": float(v)} for i, v in zip(idx, p)]
                occ = []
                if spec.get("occlusion") and seed == 42:
                    D = occlusion(model, Xn[te], length_min[te], device)
                    for k, i in enumerate(te):
                        occ += [{"seed": seed, "repeat": int(sp["repeat"]), "fold": int(sp["fold"]),
                                 "record_id": rid[i], "block": b, "minute_start": b * BLOCK_SECONDS / 60,
                                 "length_min": float(length_min[i]), "abs_dlogit": float(D[k, b])}
                                for b in range(D.shape[1])]
                pf.write_text(json.dumps({"rows": rows, "occlusion": occ,
                                          "info": {**info, "experiment": exp, "seed": seed, "fold_name": sp["fold_name"],
                                                   "seconds": round(time.time() - t0, 1)}}), encoding="utf-8")
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"[done] {tag} {time.time() - t0:.0f}s best_epoch={info['best_epoch']}", flush=True)

    # combine everything on disk
    preds, occs, logs = [], [], []
    for pf in sorted((OUT / "progress").glob("*.json")):
        d = json.loads(pf.read_text(encoding="utf-8"))
        preds += d["rows"]
        occs += d["occlusion"]
        logs.append(d["info"])
    pd.DataFrame(preds).to_csv(OUT / "predictions.csv", index=False)
    if occs:
        pd.DataFrame(occs).to_csv(OUT / "occlusion.csv", index=False)
    pd.DataFrame(logs).to_csv(OUT / "training_log.csv", index=False)
    print(f"wrote {OUT}/predictions.csv ({len(preds)} rows)", flush=True)
    print("Next: zip the folder results/shortcut and upload it.", flush=True)


if __name__ == "__main__":
    main()
