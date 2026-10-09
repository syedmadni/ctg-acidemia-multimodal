"""From-scratch Firth (1993) penalized-likelihood logistic regression.

Implements the modified-score Newton-Raphson algorithm of Firth (1993) / Heinze &
Schemper (2002), including the RESTRICTED-model convention used by R's `logistf::
logistftest()`: to test H0: beta_j = 0 (for one or more j), the restricted model is
fit by holding those coefficients at 0 and re-optimizing the rest, but the Jeffreys
penalty (0.5*log|I(beta)|) is computed from the FULL design's Fisher information at
every iteration -- never from a smaller model matrix with the tested columns dropped.
This is the exact convention documented for `logistftest`, and is the one required by
the maternal-value spec (Step 1's M0 vs M1 test, and the parity test).

Standard errors use logistf's own convention, confirmed by reading its C source
(logistf.c, `logistffit_revised`): NOT the plain inverse of X'WX at the converged
Firth estimate, but the inverse of an "augmented" information matrix
X' diag(W*(1 + 2*tau*h)) X, where h is the hat diagonal from the plain (non-augmented)
X'WX and tau=0.5 is the standard Jeffreys penalty weight. This is Firth's method
expressed via its equivalent data-augmentation representation (each observation gets
an implicit pseudo-observation of weight h_i, response 0.5). Using the plain X'WX
inverse instead (a common textbook simplification) reproduces logistf's coefficients
exactly but overstates every SE by 5-15% on the sex2 validation set -- confirmed
against logistf 1.26.1 to 6 decimals; see src/analysis/firth_ref/validate_firth.py.

Validated against R's logistf 1.26.1 on the sex2 dataset -- see
src/analysis/firth_ref/sex2_reference.R and validate_firth.py. Do not modify this
docstring's algorithm description without re-running that validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-eta))


def _penalized_loglik(X: np.ndarray, y: np.ndarray, beta: np.ndarray, eps: float = 1e-12):
    eta = X @ beta
    pr = _sigmoid(eta)
    W = pr * (1.0 - pr)
    XtWX = (X * W[:, None]).T @ X
    ll = float(np.sum(y * np.log(np.clip(pr, eps, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - pr, eps, 1.0))))
    sign, logdet = np.linalg.slogdet(XtWX)
    if sign <= 0:
        logdet = -np.inf
    pen_ll = ll + 0.5 * logdet
    return pr, W, XtWX, ll, pen_ll


@dataclass
class FirthFit:
    beta: np.ndarray                 # length p; fixed entries hold their fixed value
    se: np.ndarray                   # length p; NaN at fixed indices
    loglik: float                    # unpenalized log-likelihood at beta
    penalized_loglik: float          # Firth-penalized log-likelihood at beta
    iters: int
    converged: bool
    free_idx: np.ndarray
    fixed_idx: np.ndarray
    info: np.ndarray                 # full p x p Fisher information at beta
    max_score: float                 # max |modified score| over free params at convergence
    warnings: list = field(default_factory=list)


def fit_firth(
    X: np.ndarray,
    y: np.ndarray,
    fixed_idx: np.ndarray | None = None,
    fixed_val: np.ndarray | None = None,
    max_iter: int = 200,
    tol: float = 1e-8,
    max_halving: int = 30,
) -> FirthFit:
    """Fit (possibly restricted) Firth logistic regression by modified-score Newton-Raphson.

    X must include an intercept column if one is wanted; this function adds none itself.
    fixed_idx: 0-based column indices held fixed at fixed_val (default 0) during
    optimization; the Jeffreys penalty always uses the full X.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape
    warnings: list[str] = []

    if fixed_idx is None:
        fixed_idx = np.array([], dtype=int)
        fixed_val = np.array([], dtype=np.float64)
    else:
        fixed_idx = np.asarray(fixed_idx, dtype=int)
        fixed_val = np.zeros(len(fixed_idx)) if fixed_val is None else np.asarray(fixed_val, dtype=np.float64)

    free_idx = np.array([j for j in range(p) if j not in set(fixed_idx.tolist())], dtype=int)

    beta = np.zeros(p)
    beta[fixed_idx] = fixed_val

    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        pr, W, XtWX, ll, pen_ll = _penalized_loglik(X, y, beta)
        try:
            XtWX_inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            warnings.append(f"singular information matrix at iter {it}")
            break

        Xh = X @ XtWX_inv
        h = W * np.einsum("ij,ij->i", Xh, X)
        U_star = X.T @ (y - pr + h * (0.5 - pr))

        if len(free_idx) == 0:
            break

        Ufree = U_star[free_idx]
        Ifree = XtWX[np.ix_(free_idx, free_idx)]
        try:
            delta_free = np.linalg.solve(Ifree, Ufree)
        except np.linalg.LinAlgError:
            warnings.append(f"singular free-block information matrix at iter {it}")
            break

        step = 1.0
        accepted = False
        for _ in range(max_halving):
            beta_trial = beta.copy()
            beta_trial[free_idx] = beta[free_idx] + step * delta_free
            _, _, _, _, pen_ll_trial = _penalized_loglik(X, y, beta_trial)
            if np.isfinite(pen_ll_trial) and pen_ll_trial >= pen_ll - 1e-10:
                accepted = True
                break
            step /= 2.0
        if not accepted:
            warnings.append(f"step-halving exhausted at iter {it}; accepting last trial")
            beta_trial = beta.copy()
            beta_trial[free_idx] = beta[free_idx] + step * delta_free

        diff = float(np.max(np.abs(beta_trial[free_idx] - beta[free_idx]))) if len(free_idx) else 0.0
        beta = beta_trial
        if diff < tol:
            converged = True
            break
    else:
        warnings.append(f"did not converge in {max_iter} iterations")

    pr, W, XtWX, ll, pen_ll = _penalized_loglik(X, y, beta)
    Xh = X @ np.linalg.inv(XtWX)
    h = W * np.einsum("ij,ij->i", Xh, X)
    U_star = X.T @ (y - pr + h * (0.5 - pr))
    max_score = float(np.max(np.abs(U_star[free_idx]))) if len(free_idx) else 0.0
    if converged and max_score > 1e-4:
        warnings.append(f"converged by step-size but max|score|={max_score:.2e} > 1e-4")

    # Standard errors: logistf's convention, NOT plain inv(XtWX) -- see module docstring.
    # tau=0.5 (standard Jeffreys weight) => 2*tau = 1, matching the score/penalty above.
    W_aug = W * (1.0 + h)
    XtWaugX = (X * W_aug[:, None]).T @ X
    se = np.full(p, np.nan)
    if len(free_idx) > 0:
        cov_free = np.linalg.inv(XtWaugX)[np.ix_(free_idx, free_idx)]
        se[free_idx] = np.sqrt(np.diag(cov_free))

    return FirthFit(
        beta=beta, se=se, loglik=ll, penalized_loglik=pen_ll, iters=it, converged=converged,
        free_idx=free_idx, fixed_idx=fixed_idx, info=XtWX, max_score=max_score, warnings=warnings,
    )


@dataclass
class FirthLRT:
    lr_stat: float
    df: int
    p_value: float
    loglik_full: float
    loglik_restricted: float
    full_fit: FirthFit
    restricted_fit: FirthFit


def firth_lrt(X: np.ndarray, y: np.ndarray, test_idx: np.ndarray, test_val: np.ndarray | None = None) -> FirthLRT:
    """Penalized LRT for H0: beta[test_idx] = test_val (default 0), logistftest convention.

    Full model: unrestricted Firth fit.
    Restricted model: Firth fit with test_idx fixed at test_val, penalty still computed
    on the full design (see module docstring).
    LR statistic = 2 * (penalized_loglik_full - penalized_loglik_restricted).
    """
    test_idx = np.asarray(test_idx, dtype=int)
    full = fit_firth(X, y)
    restricted = fit_firth(X, y, fixed_idx=test_idx, fixed_val=test_val)
    lr_stat = 2.0 * (full.penalized_loglik - restricted.penalized_loglik)
    df = len(test_idx)
    p_value = float(stats.chi2.sf(lr_stat, df))
    return FirthLRT(
        lr_stat=lr_stat, df=df, p_value=p_value,
        loglik_full=full.penalized_loglik, loglik_restricted=restricted.penalized_loglik,
        full_fit=full, restricted_fit=restricted,
    )
