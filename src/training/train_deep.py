from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.ctg_cnn import CTGOnlyModel
from src.models.fusion_model import FusionModel, MaternalOnlyModel
from src.training.evaluate import evaluate_predictions, save_curves
from src.utils.seed import set_global_seed


class CTGMaternalDataset(Dataset):
    def __init__(self, X_ctg: np.ndarray, X_maternal: np.ndarray, y: np.ndarray):
        self.X_ctg = torch.tensor(X_ctg, dtype=torch.float32)
        self.X_maternal = torch.tensor(X_maternal, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X_ctg[idx], self.X_maternal[idx], self.y[idx]


class AnyTimeCTGDataset(Dataset):
    """Wraps a fixed-size [N, T_max, C] CTG array (T_max = the longest stored
    elapsed-time window, e.g. 90 min) and, every time a sample is drawn,
    zeroes out everything past a randomly chosen elapsed-time cutoff.

    This is the mechanism behind the "any elapsed time" model: rather than
    training six separate fixed-window models, one model is trained to give
    a sensible prediction from however much signal happens to be available
    -- sometimes 9 minutes' worth, sometimes 80. Zeroing the tail (instead of
    truncating + padding/masking) keeps every sample the same tensor shape,
    so no changes are needed to CTGEncoder/FusionModel/GatedFusionModel and
    no attention-masking machinery is required -- it also matches the
    zero-padding convention build_windows.py already uses for recordings
    shorter than the stored window.

    Used for the training AND validation loaders only. Outer-test evaluation
    uses fixed, deliberately chosen checkpoints (see crop_ctg_array in
    train_cv.py) so reported numbers are clean and reproducible, not
    randomized.
    """

    def __init__(
        self,
        X_ctg: np.ndarray,
        X_maternal: np.ndarray,
        y: np.ndarray,
        target_sample_rate_hz: int,
        min_crop_minutes: float,
        max_crop_minutes: float,
        seed: int = 0,
        augment: Dict[str, float] | None = None,
        duration_minutes: np.ndarray | None = None,
    ):
        self.X_ctg = torch.tensor(X_ctg, dtype=torch.float32)
        self.X_maternal = torch.tensor(X_maternal, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.min_crop_samples = int(round(min_crop_minutes * 60 * target_sample_rate_hz))
        self.max_crop_samples = min(
            int(round(max_crop_minutes * 60 * target_sample_rate_hz)),
            self.X_ctg.shape[1],
        )
        if self.min_crop_samples > self.max_crop_samples:
            raise ValueError("min_crop_minutes must not exceed max_crop_minutes / stored array length.")

        # Step 7d: duration-aware cutoffs (OPT-IN -- None reproduces the original
        # behaviour bit-for-bit). Recordings are anchored to delivery and interpolated
        # past their true end, so a cutoff beyond a record's real duration feeds the
        # model tail that encodes total recording length, which is only knowable after
        # delivery. With durations supplied, each sample's upper bound becomes
        # min(max_crop_samples, that record's true length). See
        # outputs/quality_cohorts/primary_705_anytime_vr060/maternal_value/STEP7D_SPEC.md
        self.per_sample_max_samples = None
        if duration_minutes is not None:
            dur = np.asarray(duration_minutes, dtype=float)
            if len(dur) != len(self.y):
                raise ValueError("duration_minutes must have one entry per record.")
            per_sample = np.minimum(
                np.round(dur * 60 * target_sample_rate_hz).astype(int), self.max_crop_samples
            )
            too_short = int((per_sample < self.min_crop_samples).sum())
            if too_short:
                raise ValueError(
                    f"{too_short} record(s) have true duration below min_crop_minutes; "
                    "7d spec assumes a 60.0-minute floor, so this needs an explicit decision."
                )
            self.per_sample_max_samples = per_sample
        # Phase 6a augmentation (train loader only). Applied to the active (non-zero)
        # region ONLY, so the "how much signal is available" semantics of the crop
        # survive -- noise is never written into the zeroed tail. None -> no aug, so
        # the un-augmented path is bit-identical to before.
        self._augment = augment
        self._noise_std = float(augment.get("noise_std", 0.0)) if augment else 0.0
        self._scale_range = float(augment.get("scale_range", 0.0)) if augment else 0.0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        upper = (
            int(self.per_sample_max_samples[idx])
            if self.per_sample_max_samples is not None
            else self.max_crop_samples
        )
        crop_len = int(self._rng.integers(self.min_crop_samples, upper + 1))
        x_ctg = self.X_ctg[idx].clone()
        if crop_len < x_ctg.shape[0]:
            x_ctg[crop_len:] = 0.0
        if self._augment is not None and crop_len > 0:
            head = x_ctg[:crop_len]
            if self._scale_range > 0.0:
                scale = 1.0 + (float(self._rng.random()) * 2.0 - 1.0) * self._scale_range
                head = head * scale
            if self._noise_std > 0.0:
                noise = torch.from_numpy(
                    self._rng.normal(0.0, self._noise_std, size=tuple(head.shape)).astype(np.float32)
                )
                head = head + noise
            x_ctg[:crop_len] = head
        return x_ctg, self.X_maternal[idx], self.y[idx]


class FocalLossWithLogits(nn.Module):
    """Binary focal loss (Lin et al. 2017), an alternative to BCE + pos_weight for
    the 6.7%-positive setting. gamma down-weights easy (well-classified) examples
    so gradient focuses on the hard, mostly-positive ones; alpha class-weights the
    positive term. Used INSTEAD of pos_weight (not on top of it) so imbalance is
    handled once."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        focal = (1.0 - p_t) ** self.gamma * bce
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * focal).mean()


def pretrain_ctg_encoder(
    encoder: nn.Module,
    X_ctg_norm_train: np.ndarray,
    targets_train: np.ndarray,
    device: torch.device,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    seed: int = 0,
) -> Dict[str, float]:
    """Phase 6d: auxiliary pretraining, adapted from DeepCTG 2.0 (Boudet et al.,
    Computers in Biology and Medicine 2024).

    The CNN+encoder first learns to REGRESS clinically-known CTG features
    (decelerations, variability, baseline...) from the raw signal. That auxiliary
    task has a target for EVERY record -- the features are computed, not
    label-scarce -- so the encoder learns clinically-grounded filters from all
    345 records before ever touching the 23-positive acidemia label. The
    pretrained encoder is then fine-tuned on acidemia by the normal harness.

    No label leakage: targets are derived from the input signal itself, never
    from y. Still, only inner-train rows are passed in by the caller, and target
    standardization is fit on those rows only, per the Validation Standard.

    Mutates `encoder` in place; returns simple training diagnostics.
    """
    torch.manual_seed(seed)
    n_targets = targets_train.shape[1]
    head = nn.Linear(128, n_targets).to(device)  # CTGEncoder always projects to 128
    encoder = encoder.to(device)

    x = torch.tensor(X_ctg_norm_train, dtype=torch.float32)
    t = torch.tensor(targets_train, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(x, t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=learning_rate)
    criterion = nn.MSELoss()

    first_loss = None
    last_loss = None
    for _ in range(epochs):
        encoder.train()
        head.train()
        losses = []
        for xb, tb in loader:
            xb, tb = xb.to(device), tb.to(device)
            optimizer.zero_grad()
            loss = criterion(head(encoder(xb)), tb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        if first_loss is None:
            first_loss = epoch_loss
        last_loss = epoch_loss

    return {
        "pretrain_epochs": epochs,
        "pretrain_first_epoch_loss": first_loss if first_loss is not None else float("nan"),
        "pretrain_last_epoch_loss": last_loss if last_loss is not None else float("nan"),
        "pretrain_n_targets": n_targets,
    }


def choose_device(config_device: str) -> torch.device:
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)


def fit_preprocessors(
    X_ctg_train: np.ndarray,
    X_mat_train: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler, np.ndarray, np.ndarray]:
    # CTG normalization is fit on train only, per channel.
    ctg_mean = X_ctg_train.reshape(-1, X_ctg_train.shape[-1]).mean(axis=0)
    ctg_std = X_ctg_train.reshape(-1, X_ctg_train.shape[-1]).std(axis=0)
    ctg_std = np.where(ctg_std < 1e-6, 1.0, ctg_std)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_mat_train_imp = imputer.fit_transform(X_mat_train)
    scaler.fit(X_mat_train_imp)
    return X_mat_train_imp, scaler.transform(X_mat_train_imp), imputer, scaler, ctg_mean, ctg_std


def transform_arrays(
    X_ctg: np.ndarray,
    X_mat: np.ndarray, 
    imputer: SimpleImputer,
    scaler: StandardScaler,
    ctg_mean: np.ndarray,
    ctg_std: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    X_ctg_norm = ((X_ctg - ctg_mean.reshape(1, 1, -1)) / ctg_std.reshape(1, 1, -1)).astype(np.float32)
    X_mat_norm = scaler.transform(imputer.transform(X_mat)).astype(np.float32)
    return X_ctg_norm, X_mat_norm


def train_one_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    pos_weight: float,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    models_dir: Path,
) -> nn.Module:
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float("inf")
    bad_epochs = 0
    best_path = models_dir / f"{name}.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x_ctg, x_mat, y in train_loader:
            x_ctg, x_mat, y = x_ctg.to(device), x_mat.to(device), y.to(device)
            optimizer.zero_grad()
            if name == "ctg_only":
                logits = model(x_ctg)
            elif name == "maternal_only":
                logits = model(x_mat)
            else:
                logits = model(x_ctg, x_mat)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_ctg, x_mat, y in val_loader:
                x_ctg, x_mat, y = x_ctg.to(device), x_mat.to(device), y.to(device)
                if name == "ctg_only":
                    logits = model(x_ctg)
                elif name == "maternal_only":
                    logits = model(x_mat)
                else:
                    logits = model(x_ctg, x_mat)
                loss = criterion(logits, y)
                val_losses.append(float(loss.item()))

        val_loss = float(np.mean(val_losses))
        print(f"{name} | epoch={epoch:03d} train_loss={np.mean(train_losses):.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"{name} early stopped at epoch {epoch}")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    return model


def predict_model(name: str, model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for x_ctg, x_mat, _ in loader:
            x_ctg, x_mat = x_ctg.to(device), x_mat.to(device)
            if name == "ctg_only":
                logits = model(x_ctg)
            elif name == "maternal_only":
                logits = model(x_mat)
            else:
                logits = model(x_ctg, x_mat)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CTG-only, maternal-only, and fusion deep models.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    set_global_seed(seed)

    processed_dir = Path(cfg["paths"]["processed_dir"])
    splits_dir = Path(cfg["paths"]["splits_dir"])
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    metrics_dir = outputs_dir / "metrics"
    figures_dir = outputs_dir / "figures"
    models_dir = outputs_dir / "models"
    tables_dir = outputs_dir / "tables"
    for d in [splits_dir, metrics_dir, figures_dir, models_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    X_ctg = np.load(processed_dir / "X_ctg.npy")
    X_maternal = np.load(processed_dir / "X_maternal.npy")
    y = np.load(processed_dir / "y.npy").astype(int)
    record_ids = pd.read_csv(processed_dir / "record_ids.csv")

    idx = np.arange(len(y))
    trainval_idx, test_idx = train_test_split(
        idx,
        test_size=float(cfg["training"]["test_size"]),
        random_state=seed,
        stratify=y,
    )
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=float(cfg["training"]["val_size"]),
        random_state=seed,
        stratify=y[trainval_idx],
    )

    np.savez(splits_dir / "deep_split_indices.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    split_labels = pd.Series("unused", index=np.arange(len(y)))
    split_labels.iloc[train_idx] = "train"
    split_labels.iloc[val_idx] = "val"
    split_labels.iloc[test_idx] = "test"
    record_ids.assign(split=split_labels.values).to_csv(splits_dir / "record_ids_all.csv", index=False)

    _, _, imputer, scaler, ctg_mean, ctg_std = fit_preprocessors(X_ctg[train_idx], X_maternal[train_idx])
    X_ctg_norm, X_mat_norm = transform_arrays(X_ctg, X_maternal, imputer, scaler, ctg_mean, ctg_std)
    joblib.dump({"maternal_imputer": imputer, "maternal_scaler": scaler, "ctg_mean": ctg_mean, "ctg_std": ctg_std}, models_dir / "preprocessors.joblib")

    train_ds = CTGMaternalDataset(X_ctg_norm[train_idx], X_mat_norm[train_idx], y[train_idx])
    val_ds = CTGMaternalDataset(X_ctg_norm[val_idx], X_mat_norm[val_idx], y[val_idx])
    test_ds = CTGMaternalDataset(X_ctg_norm[test_idx], X_mat_norm[test_idx], y[test_idx])

    batch_size = int(cfg["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    pos = int(y[train_idx].sum())
    neg = int((y[train_idx] == 0).sum())
    pos_weight = float(neg / max(pos, 1))
    device = choose_device(str(cfg["training"]["device"]))

    common = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "device": device,
        "pos_weight": pos_weight,
        "epochs": int(cfg["training"]["epochs"]),
        "learning_rate": float(cfg["training"]["learning_rate"]),
        "weight_decay": float(cfg["training"]["weight_decay"]),
        "patience": int(cfg["training"]["early_stopping_patience"]),
        "models_dir": models_dir,
    }

    models: Dict[str, nn.Module] = {
        "ctg_only": CTGOnlyModel(input_channels=X_ctg.shape[-1]),
        "maternal_only": MaternalOnlyModel(num_features=X_maternal.shape[1]),
        "fusion": FusionModel(num_maternal_features=X_maternal.shape[1], input_channels=X_ctg.shape[-1]),
    }

    results = {}
    for name, model in models.items():
        trained = train_one_model(name=name, model=model, **common)
        proba = predict_model(name, trained, test_loader, device)
        metrics = evaluate_predictions(y[test_idx], proba, threshold=float(cfg["training"]["threshold"]))
        results[name] = metrics
        save_curves(y[test_idx], proba, figures_dir, name)
        pd.DataFrame({"record_id": record_ids.iloc[test_idx]["record_id"].values, "y_true": y[test_idx], "y_proba": proba}).to_csv(
            tables_dir / f"{name}_test_predictions.csv", index=False
        )

    experiment_summary = {
        "seed": seed,
        "device": str(device),
        "X_ctg_shape": list(X_ctg.shape),
        "X_maternal_shape": list(X_maternal.shape),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "train_positives": int(y[train_idx].sum()),
        "test_positives": int(y[test_idx].sum()),
        "pos_weight": pos_weight,
        "results": results,
    }

    (metrics_dir / "deep_metrics.json").write_text(json.dumps(experiment_summary, indent=2), encoding="utf-8")
    pd.DataFrame(results).T.to_csv(tables_dir / "deep_metrics.csv")
    print(json.dumps(experiment_summary, indent=2))


if __name__ == "__main__":
    main()
