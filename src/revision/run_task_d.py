"""Task D: numbers for the underpowered framing.

D1. Upper 95 % bound of the LR combined - CTG-only ROC-AUC difference (the largest gain the
    study cannot exclude), per checkpoint, for ckfilter / unfiltered / primary345_causal
    (from Task A), with the published Step 3 values alongside.

D2. Simulation-based power of the Firth 8-df maternal-block LRT (DGM frozen in DEVIATIONS §3):
    covariate rows fixed at the observed primary (ckfilter) cohort at each checkpoint,
    median-imputed; true logit = alpha + CTG part (Firth M0 coefficients) + gamma * s, where
    s is the maternal linear predictor in the direction of the observed Firth full-model
    maternal coefficients, standardised to SD 1; alpha keeps expected prevalence equal to the
    observed prevalence. gamma is calibrated so that the population AUC of the true logit
    minus the population AUC of its CTG part equals the target gain (population AUC = the
    expected AUC over the fixed covariate rows under the true risks). 500 outcome vectors per
    effect size; power at alpha 0.05 and 0.01 (the strictest Holm step over 5 checkpoints),
    with Monte Carlo SE. A gain of 0 (simulation size check) is included as a validity check.

    python -m src.revision.run_task_d
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.impute import SimpleImputer

from src.analysis.firth import fit_firth, firth_lrt
from src.revision.causal_features import causal_feature_matrix
from src.revision.cohorts import CHECKPOINTS, labelled_meta, maternal_matrix, members, quality_table
from src.revision.raw_cache import RawData
from src.revision.run_task_a import ctg_block_whole_cohort

OUT = Path("results/revision")
PUBLISHED_STEP3 = Path("outputs/quality_cohorts/primary_705_anytime_vr060/maternal_value/step3_bootstrap_auc_diff.csv")
GAINS = [0.0, 0.03, 0.05, 0.08, 0.10]
N_SIM = 500
SEED = 42


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def population_auc(score: np.ndarray, p: np.ndarray) -> float:
    """E[AUC] over fixed rows when row i is a case with prob p_i (pairs i != j, ties = 0.5)."""
    order = np.argsort(score, kind="mergesort")
    s, pp = score[order], p[order]
    q = 1.0 - pp
    # for each i (as case), sum of q_j over j with s_j < s_i, plus half over ties (excluding i)
    num = 0.0
    uniq, start = np.unique(s, return_index=True)
    cum_q = np.concatenate([[0.0], np.cumsum(q)])
    for k, st in enumerate(start):
        en = start[k + 1] if k + 1 < len(start) else len(s)
        below = cum_q[st]
        tie_q = cum_q[en] - cum_q[st]
        blk_p, blk_q = pp[st:en], q[st:en]
        num += (blk_p * (below + 0.5 * (tie_q - blk_q))).sum()
    den = pp.sum() * q.sum() - (pp * q).sum()
    return float(num / den)


def calibrate_alpha(lp: np.ndarray, prev: float) -> float:
    return brentq(lambda a: sigmoid(a + lp).mean() - prev, -30, 30)


def gain_for(gamma, ctg_lp, s, prev):
    eta0 = ctg_lp + gamma * s
    a = calibrate_alpha(eta0, prev)
    p = sigmoid(a + eta0)
    return population_auc(eta0, p) - population_auc(ctg_lp, p), a


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # ---------------- D1
    a = pd.read_csv(OUT / "A_combined_vs_ctg.csv")
    a = a[a.model == "logistic_regression"][["variant", "checkpoint_minutes", "n", "cases", "delta_auc",
                                             "delta_ci_low", "delta_ci_high"]]
    a = a.rename(columns={"delta_ci_high": "largest_gain_not_excluded_upper95"})
    pub = pd.read_csv(PUBLISHED_STEP3)[["checkpoint_minutes", "observed_diff", "percentile_ci_low", "percentile_ci_high"]]
    pub = pub.assign(variant="published_345_v2_features (old, comparison only)", n=345, cases=23).rename(
        columns={"observed_diff": "delta_auc", "percentile_ci_low": "delta_ci_low",
                 "percentile_ci_high": "largest_gain_not_excluded_upper95"})
    d1 = pd.concat([a, pub], ignore_index=True)
    d1["note"] = "LR CTG+maternal minus LR CTG-only ROC-AUC; stratified bootstrap percentile CI (2000)"
    d1.round(4).to_csv(OUT / "D_delta_auc_upper_bounds.csv", index=False)

    # ---------------- D2
    raw = RawData()
    meta = labelled_meta(raw)
    q = quality_table(raw)
    rng = np.random.default_rng(SEED)
    rows, cal_rows = [], []
    for t in CHECKPOINTS:
        coh = members(meta, q, "ckfilter", t)
        y_obs = coh["y"].to_numpy()
        prev = float(y_obs.mean())
        ctg, _, _ = ctg_block_whole_cohort(causal_feature_matrix(raw, coh["record_id"].to_numpy(), t))
        mat = SimpleImputer(strategy="median").fit_transform(maternal_matrix(coh))
        X0 = np.column_stack([np.ones(len(y_obs)), ctg])
        X1 = np.column_stack([X0, mat])
        mat_idx = list(range(X0.shape[1], X1.shape[1]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f0 = fit_firth(X0, y_obs)
            f1 = fit_firth(X1, y_obs)
        ctg_lp = ctg @ f0.beta[1:]
        s = mat @ f1.beta[mat_idx]
        s = (s - s.mean()) / s.std()
        for g in GAINS:
            if g == 0:
                gamma = 0.0
            else:
                gamma = brentq(lambda gm: gain_for(gm, ctg_lp, s, prev)[0] - g, 1e-6, 10.0, xtol=1e-6)
            achieved, alpha = gain_for(gamma, ctg_lp, s, prev)
            p = sigmoid(alpha + ctg_lp + gamma * s)
            cal_rows.append({"checkpoint_minutes": t, "target_gain": g, "gamma": gamma, "alpha": alpha,
                             "achieved_population_gain": achieved,
                             "population_auc_true_logit": population_auc(ctg_lp + gamma * s, p),
                             "population_auc_ctg_part": population_auc(ctg_lp, p),
                             "expected_cases": p.sum(), "n": len(p)})
            pvals, events, nonconv = [], [], 0
            t0 = time.time()
            for _ in range(N_SIM):
                ys = (rng.random(len(p)) < p).astype(float)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = firth_lrt(X1, ys, test_idx=mat_idx)
                nonconv += int(not (r.full_fit.converged and r.restricted_fit.converged))
                pvals.append(r.p_value)
                events.append(ys.sum())
            pv = np.asarray(pvals)
            for alpha_level in (0.05, 0.01):
                pw = float((pv < alpha_level).mean())
                rows.append({"checkpoint_minutes": t, "n": len(p), "observed_cases": int(y_obs.sum()),
                             "target_auc_gain": g, "alpha_level": alpha_level, "power": pw,
                             "mc_se": float(np.sqrt(pw * (1 - pw) / N_SIM)), "n_sim": N_SIM,
                             "mean_simulated_cases": float(np.mean(events)),
                             "nonconverged_fits": nonconv,
                             "label": "size check (gain 0)" if g == 0 else "power",
                             "seconds": round(time.time() - t0, 1)})
            print(f"t={t:g} gain={g}: power@.05={np.mean(pv < 0.05):.3f} ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(cal_rows).round(5).to_csv(OUT / "D_power_calibration.csv", index=False)
    pd.DataFrame(rows).round(4).to_csv(OUT / "D_power_simulation.csv", index=False)
    print("Task D complete.")


if __name__ == "__main__":
    main()
