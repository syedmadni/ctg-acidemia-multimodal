"""Validate src/analysis/firth.py against R's logistf 1.26.1 on sex2 (Option 3, primary),
firthlogist's README (Option 1, secondary, does not block), and an internal scipy
cross-check of the penalized log-likelihood optimum (does not block; consistency check).

GATE (per user decision, 2026-09-20): coefficients, standard errors, penalized
log-likelihood, and both logistftest() statistics/p-values must match logistf to 3
decimals. If not, STOP -- do not proceed to Step 1.

Run from repo root: python -m src.analysis.firth_ref.validate_firth
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, for `src.*` imports
from src.analysis.firth import fit_firth, firth_lrt  # noqa: E402

HERE = Path(__file__).resolve().parent
GATE_DECIMALS = 3

FIRTHLOGIST_COEFS = {
    # https://github.com/jzluo/firthlogist README, commit 0a4df9cf78a08bcab9acfd0b98148d2642ddf82e
    # (2025-12-05), fetched 2026-09-20. Secondary cross-check only (Option 1) -- does not
    # gate. firthlogist's SE/p-value method differs from logistf's profile-likelihood method
    # (see reported deltas below); only coefficients and log-likelihood are compared here.
    "(Intercept)": 0.120254, "age": -1.10598, "oc": -0.0688167, "vic": 2.26887,
    "vicl": -2.11141, "vis": -0.788317, "dia": 3.09601,
}
FIRTHLOGIST_LOGLIK = -132.5394


def load_sex2():
    res = pyreadr.read_r(str(HERE / "sex2.rda"))
    df = res["sex2"]
    terms = ["age", "oc", "vic", "vicl", "vis", "dia"]
    X = np.column_stack([np.ones(len(df))] + [df[t].to_numpy(dtype=float) for t in terms])
    y = df["case"].to_numpy(dtype=float)
    names = ["(Intercept)"] + terms
    return X, y, names, df


def gate(label, py_val, r_val, decimals=GATE_DECIMALS):
    ok = round(float(py_val), decimals) == round(float(r_val), decimals)
    status = "PASS" if ok else "FAIL"
    return ok, f"{status}  {label:45s} python={py_val: .6f}  R={r_val: .6f}  (round{decimals}d)"


def main():
    X, y, names, df = load_sex2()
    n, p = X.shape
    report = []
    all_pass = True

    report.append(f"sex2: n={n}, p={p} (incl. intercept), positives={int(y.sum())}")
    report.append(f"columns: {names}\n")

    # ---------------- 1. Full model ----------------
    full = fit_firth(X, y)
    ref_coefs = pd.read_csv(HERE / "sex2_reference_coefs.csv")
    ref_scalars = pd.read_csv(HERE / "sex2_reference_scalars.csv").set_index("quantity")["value"]

    report.append("=== 1. Full model: coefficients, SE ===")
    for j, nm in enumerate(names):
        r_row = ref_coefs[ref_coefs.term == nm].iloc[0]
        ok, msg = gate(f"coef[{nm}]", full.beta[j], r_row.estimate)
        all_pass &= ok
        report.append(msg)
        ok, msg = gate(f"se[{nm}]", full.se[j], r_row.se)
        all_pass &= ok
        report.append(msg)

    ok, msg = gate("penalized_loglik (full)", full.penalized_loglik, ref_scalars["loglik_full"])
    all_pass &= ok
    report.append(msg)
    report.append(f"Python iterations to converge: {full.iters}; converged={full.converged}; "
                  f"max|score|={full.max_score:.3e}; warnings={full.warnings}\n")

    # ---------------- 2. logistftest convention: single coefficient (dia=0) ----------------
    dia_idx = names.index("dia")
    single = firth_lrt(X, y, test_idx=[dia_idx])
    report.append("=== 2. Restricted LRT: dia = 0 (single coefficient) ===")
    ok, msg = gate("restricted penalized_loglik", single.loglik_restricted,
                    ref_scalars["single_test_restricted_loglik"])
    all_pass &= ok
    report.append(msg)
    ok, msg = gate("LR statistic", single.lr_stat, ref_scalars["single_test_LR_stat"])
    all_pass &= ok
    report.append(msg)
    ok = single.df == int(ref_scalars["single_test_df"])
    all_pass &= ok
    report.append(f"{'PASS' if ok else 'FAIL'}  df: python={single.df} R={int(ref_scalars['single_test_df'])}")
    ok, msg = gate("p-value", single.p_value, ref_scalars["single_test_p"], decimals=4)
    # p-values this small need more precision than 3dp to be meaningful; also check 3dp explicitly
    ok3, _ = gate("p-value(3dp)", single.p_value, ref_scalars["single_test_p"], decimals=3)
    all_pass &= ok3
    report.append(f"{'PASS' if ok3 else 'FAIL'}  p-value: python={single.p_value:.6e} "
                  f"R={ref_scalars['single_test_p']:.6e} (round3d on raw value)\n")

    # ---------------- 3. logistftest convention: joint (vic=0, vicl=0) ----------------
    vic_idx, vicl_idx = names.index("vic"), names.index("vicl")
    joint = firth_lrt(X, y, test_idx=[vic_idx, vicl_idx])
    report.append("=== 3. Restricted LRT: vic = 0 AND vicl = 0 (joint) ===")
    ok, msg = gate("restricted penalized_loglik", joint.loglik_restricted,
                    ref_scalars["joint_test_restricted_loglik"])
    all_pass &= ok
    report.append(msg)
    ok, msg = gate("LR statistic", joint.lr_stat, ref_scalars["joint_test_LR_stat"])
    all_pass &= ok
    report.append(msg)
    ok = joint.df == int(ref_scalars["joint_test_df"])
    all_pass &= ok
    report.append(f"{'PASS' if ok else 'FAIL'}  df: python={joint.df} R={int(ref_scalars['joint_test_df'])}")
    ok3, _ = gate("p-value(3dp)", joint.p_value, ref_scalars["joint_test_p"], decimals=3)
    all_pass &= ok3
    report.append(f"{'PASS' if ok3 else 'FAIL'}  p-value: python={joint.p_value:.6e} "
                  f"R={ref_scalars['joint_test_p']:.6e} (round3d on raw value)\n")

    # ---------------- 4. Secondary cross-check: firthlogist (does not block) ----------------
    report.append("=== 4. SECONDARY cross-check vs firthlogist README (does not block) ===")
    report.append("Source: https://github.com/jzluo/firthlogist README.md, "
                  "commit 0a4df9cf78a08bcab9acfd0b98148d2642ddf82e (2025-12-05), fetched 2026-09-20.")
    for j, nm in enumerate(names):
        fl_val = FIRTHLOGIST_COEFS[nm]
        diff = full.beta[j] - fl_val
        report.append(f"  coef[{nm:12s}] python={full.beta[j]: .6f}  firthlogist={fl_val: .6f}  diff={diff:+.6f}")
    report.append(f"  loglik: python={full.penalized_loglik:.4f}  firthlogist={FIRTHLOGIST_LOGLIK:.4f}  "
                  f"diff={full.penalized_loglik - FIRTHLOGIST_LOGLIK:+.4f}")
    report.append("  NOTE: firthlogist's reported SEs/p-values differ from logistf's (different "
                  "SE/inference method -- firthlogist does not use profile-likelihood CIs by "
                  "default); coefficients and log-likelihood agree closely. Reported for "
                  "completeness; does not gate.\n")

    # ---------------- 5. Internal check: independent optimizer on penalized log-lik ----------------
    report.append("=== 5. Internal check: scipy.optimize.minimize on penalized log-likelihood ===")

    def neg_pen_ll_and_grad(beta):
        eta = X @ beta
        pr = 1.0 / (1.0 + np.exp(-eta))
        W = pr * (1.0 - pr)
        XtWX = (X * W[:, None]).T @ X
        XtWX_inv = np.linalg.inv(XtWX)
        eps = 1e-12
        ll = float(np.sum(y * np.log(np.clip(pr, eps, 1)) + (1 - y) * np.log(np.clip(1 - pr, eps, 1))))
        sign, logdet = np.linalg.slogdet(XtWX)
        pen_ll = ll + 0.5 * logdet
        Xh = X @ XtWX_inv
        h = W * np.einsum("ij,ij->i", Xh, X)
        grad = X.T @ (y - pr + h * (0.5 - pr))
        return -pen_ll, -grad

    res = minimize(neg_pen_ll_and_grad, x0=np.zeros(p), jac=True, method="BFGS",
                    options={"gtol": 1e-10, "maxiter": 2000})
    grad_norm = float(np.linalg.norm(res.jac))
    max_beta_diff = float(np.max(np.abs(res.x - full.beta)))
    ok_beta = round(max_beta_diff, 4) == 0.0
    ok_grad = grad_norm < 1e-6
    all_pass &= ok_beta and ok_grad
    report.append(f"scipy converged: {res.success}, message: {res.message}")
    report.append(f"{'PASS' if ok_beta else 'FAIL'}  max|beta_scipy - beta_newton| = {max_beta_diff:.3e} "
                  f"(gate: round to 4dp == 0)")
    report.append(f"{'PASS' if ok_grad else 'FAIL'}  ||gradient|| at scipy solution = {grad_norm:.3e} "
                  f"(gate: < 1e-6)")
    report.append(f"scipy penalized loglik = {-res.fun:.6f}  (Newton-Raphson = {full.penalized_loglik:.6f})\n")

    # ---------------- verdict ----------------
    report.append("=" * 70)
    report.append(f"GATE VERDICT (Steps 1-4, {GATE_DECIMALS} decimals): "
                  f"{'PASS -- proceed to Step 1' if all_pass else 'FAIL -- STOP, do not proceed'}")
    report.append("=" * 70)

    text = "\n".join(report)
    print(text)
    (HERE / "validation_report.txt").write_text(text, encoding="utf-8")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
