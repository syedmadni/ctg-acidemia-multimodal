"""Paired significance tests for comparing two ROC-AUCs on the same records.

Both tests here are paired: they assume proba_a and proba_b are two sets of
predictions for the *same* records in the *same* order, with one shared
y_true. That's the right structure for (a) two models compared within one
cohort (trivially paired, same OOF rows) and (b) two cohorts at different
window lengths compared on the subset of records present in both (paired on
the intersection) -- see run_comparison() in this module, which builds that
alignment via an inner join on record_id before calling either test.

Do not feed these two independent/unpaired samples; that requires a different
test (e.g. an unpaired DeLong variant), which is out of scope here since every
comparison in this project's checklist is naturally paired at the record level.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score


def _psi(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Placement/comparison kernel: 1 if x>y, 0.5 if x==y, 0 if x<y. Vectorized to [len(x), len(y)]."""
    return (x[:, None] > y[None, :]).astype(float) + 0.5 * (x[:, None] == y[None, :]).astype(float)


def _structural_components(pos_scores: np.ndarray, neg_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pos_scores/neg_scores: [K classifiers, m or n cases]. Returns V10 [K,m], V01 [K,n]."""
    K = pos_scores.shape[0]
    m = pos_scores.shape[1]
    n = neg_scores.shape[1]
    V10 = np.empty((K, m))
    V01 = np.empty((K, n))
    for k in range(K):
        psi = _psi(pos_scores[k], neg_scores[k])  # [m, n]
        V10[k] = psi.mean(axis=1)
        V01[k] = psi.mean(axis=0)
    return V10, V01


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def delong_test(y_true: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray) -> Dict[str, float]:
    """Paired DeLong (1988) test for two correlated ROC-AUCs on the same cases.

    Direct O(m*n) structural-component computation (not the sorted O(N log N)
    fast variant) -- fine at this project's sample sizes (n well under 500).
    """
    y_true = np.asarray(y_true, dtype=int)
    proba_a = np.asarray(proba_a, dtype=float)
    proba_b = np.asarray(proba_b, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {"auc_a": float("nan"), "auc_b": float("nan"), "auc_diff": float("nan"), "z": float("nan"), "p_value": float("nan")}

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    pos_scores = np.vstack([proba_a[pos_mask], proba_b[pos_mask]])
    neg_scores = np.vstack([proba_a[neg_mask], proba_b[neg_mask]])
    m = pos_scores.shape[1]
    n = neg_scores.shape[1]

    V10, V01 = _structural_components(pos_scores, neg_scores)
    auc = V10.mean(axis=1)  # [2]: auc_a, auc_b -- equals V01.mean(axis=1) too

    S10 = np.cov(V10, ddof=1)
    S01 = np.cov(V01, ddof=1)
    S10 = np.atleast_2d(S10)
    S01 = np.atleast_2d(S01)
    S = S10 / m + S01 / n

    var_diff = float(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    auc_diff = float(auc[0] - auc[1])

    if var_diff <= 0:
        z = 0.0 if auc_diff == 0 else math.copysign(float("inf"), auc_diff)
        p_value = 1.0 if auc_diff == 0 else 0.0
    else:
        z = auc_diff / math.sqrt(var_diff)
        p_value = 2.0 * (1.0 - _norm_cdf(abs(z)))

    return {
        "auc_a": float(auc[0]),
        "auc_b": float(auc[1]),
        "auc_diff": auc_diff,
        "z": float(z),
        "p_value": float(p_value),
        "n_pos": int(m),
        "n_neg": int(n),
    }


def paired_permutation_test(
    y_true: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    n_permutations: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Nonparametric complement to delong_test: under H0, which prediction is
    'model A' vs 'model B' for a given record is exchangeable. Permute that
    per-record label assignment, rebuild both AUCs, and see how extreme the
    observed |auc_a - auc_b| is against that null distribution.
    """
    y_true = np.asarray(y_true, dtype=int)
    proba_a = np.asarray(proba_a, dtype=float)
    proba_b = np.asarray(proba_b, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {"auc_a": float("nan"), "auc_b": float("nan"), "observed_diff": float("nan"), "p_value": float("nan"), "n_permutations": 0}

    auc_a = float(roc_auc_score(y_true, proba_a))
    auc_b = float(roc_auc_score(y_true, proba_b))
    observed_diff = auc_a - auc_b

    rng = np.random.default_rng(seed)
    stacked = np.vstack([proba_a, proba_b])  # [2, N]
    n = len(y_true)
    count_extreme = 0
    for _ in range(n_permutations):
        swap = rng.integers(0, 2, size=n)
        perm_a = np.where(swap == 0, stacked[0], stacked[1])
        perm_b = np.where(swap == 0, stacked[1], stacked[0])
        diff = roc_auc_score(y_true, perm_a) - roc_auc_score(y_true, perm_b)
        if abs(diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)
    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "observed_diff": observed_diff,
        "p_value": float(p_value),
        "n_permutations": int(n_permutations),
    }
