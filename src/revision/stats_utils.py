"""Stratified patient-level bootstrap and small helpers (2000 resamples, seed 42; DEVIATIONS §3)."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from statsmodels.stats.multitest import multipletests

BOOT_ITERS = 2000
SEED = 42


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="holm")[1]
    return out.tolist()


def strat_boot_index(y: np.ndarray, iters: int = BOOT_ITERS, seed: int = SEED) -> np.ndarray:
    """[iters, n] resample indices; the first n_pos columns are cases, the rest controls."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    return np.hstack([rng.choice(pos, size=(iters, len(pos))), rng.choice(neg, size=(iters, len(neg)))])


def _auc_rows(scores: np.ndarray, n_pos: int) -> np.ndarray:
    """AUC per row of a [B, n] score matrix whose first n_pos columns are cases (ties averaged)."""
    r = rankdata(scores, axis=1)
    n_neg = scores.shape[1] - n_pos
    return (r[:, :n_pos].sum(axis=1) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auc_pr_ci(y: np.ndarray, p: np.ndarray, iters: int = BOOT_ITERS, seed: int = SEED) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    idx = strat_boot_index(y, iters, seed)
    n_pos = int(y.sum())
    aucs = _auc_rows(p[idx], n_pos)
    prs = np.array([average_precision_score(y[i], p[i]) for i in idx])
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "roc_auc_ci_low": float(np.percentile(aucs, 2.5)), "roc_auc_ci_high": float(np.percentile(aucs, 97.5)),
        "pr_auc": float(average_precision_score(y, p)),
        "pr_auc_ci_low": float(np.percentile(prs, 2.5)), "pr_auc_ci_high": float(np.percentile(prs, 97.5)),
        "pr_auc_floor_prevalence": float(y.mean()),
        "bootstrap_iters": iters, "bootstrap": "stratified patient-level, percentile",
    }


def paired_delta_ci(y: np.ndarray, pa: np.ndarray, pb: np.ndarray,
                    iters: int = BOOT_ITERS, seed: int = SEED) -> dict:
    """AUC(pa) - AUC(pb) with a paired stratified bootstrap."""
    y = np.asarray(y, dtype=int)
    idx = strat_boot_index(y, iters, seed)
    n_pos = int(y.sum())
    d = _auc_rows(np.asarray(pa, float)[idx], n_pos) - _auc_rows(np.asarray(pb, float)[idx], n_pos)
    obs = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    return {
        "delta_auc": float(obs),
        "delta_ci_low": float(np.percentile(d, 2.5)), "delta_ci_high": float(np.percentile(d, 97.5)),
        "prop_boot_delta_gt0": float(np.mean(d > 0)),
        # two-sided bootstrap p (percentile-inversion), reported alongside DeLong
        "boot_p_two_sided": float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))),
    }
