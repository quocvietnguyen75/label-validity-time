"""XF-C7 estimators with segment-clustered (CR1) variances (protocol v1.0 section 5; pilot record 2026-10-21).

- weighted mean: mu = sum w y / sum w; V_CR1 = G/(G-1) sum_g (sum_{i in g} w_i e_i)^2 / (sum w)^2, e = y - mu.
- pooled slope with group intercepts: weighted within-group demeaning, beta = sum w x~ y~ / sum w x~^2;
  scores s_i = w_i x~_i (y~_i - beta x~_i); V_CR1 = G/(G-1) sum_g (sum s)^2 / (sum w x~^2)^2.
- independent-observation variances with per-observation variances `var`; design effect = V_CR1 / V_ind.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def _codes(cluster) -> Tuple[np.ndarray, int]:
    codes, uniq = pd.factorize(pd.Series(np.asarray(cluster)), sort=False)
    return codes, len(uniq)


def weighted_mean_cr1(y, w, cluster) -> Tuple[float, float]:
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    sw = w.sum()
    mu = float((w * y).sum() / sw)
    codes, G = _codes(cluster)
    s = np.bincount(codes, weights=w * (y - mu), minlength=G)
    v = (G / (G - 1)) * float((s ** 2).sum()) / sw ** 2 if G > 1 else float("nan")
    return mu, float(np.sqrt(v))


def ind_var_weighted_mean(w, var) -> float:
    w = np.asarray(w, dtype=float)
    return float((w ** 2 * np.asarray(var, dtype=float)).sum() / w.sum() ** 2)


def _demean(v, w, grp) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)
    codes, k = _codes(grp)
    m = np.bincount(codes, weights=w * v, minlength=k) / np.bincount(codes, weights=w, minlength=k)
    return v - m[codes]


def wls_slope_cr1(y, x, w, grp, cluster) -> Tuple[float, float]:
    w = np.asarray(w, dtype=float)
    xt = _demean(x, w, grp)
    yt = _demean(y, w, grp)
    sxx = float((w * xt ** 2).sum())
    b = float((w * xt * yt).sum() / sxx)
    codes, G = _codes(cluster)
    s = np.bincount(codes, weights=w * xt * (yt - b * xt), minlength=G)
    v = (G / (G - 1)) * float((s ** 2).sum()) / sxx ** 2 if G > 1 else float("nan")
    return b, float(np.sqrt(v))


def wls_cr1(y, X, w, grp, cluster) -> Tuple[np.ndarray, np.ndarray]:
    """Several regressors with group intercepts (columns demeaned within group); CR1 sandwich, as wls_slope_cr1."""
    w = np.asarray(w, dtype=float)
    Xt = np.column_stack([_demean(c, w, grp) for c in np.asarray(X, dtype=float).T])
    yt = _demean(y, w, grp)
    A = Xt.T @ (w[:, None] * Xt)
    b = np.linalg.solve(A, Xt.T @ (w * yt))
    codes, G = _codes(cluster)
    S = np.zeros((G, Xt.shape[1]))
    np.add.at(S, codes, (w * (yt - Xt @ b))[:, None] * Xt)
    Ai = np.linalg.inv(A)
    V = (G / (G - 1)) * Ai @ (S.T @ S) @ Ai if G > 1 else np.full(A.shape, np.nan)
    return b, np.sqrt(np.clip(np.diag(V), 0.0, None))


def wls_intercept_slope_cr1(y, x, w, cluster):
    """WLS of y on [1, x] (no group intercepts); returns ((alpha, se), (beta, se)) with the CR1 sandwich by cluster."""
    w = np.asarray(w, dtype=float)
    X = np.column_stack([np.ones(len(w)), np.asarray(x, dtype=float)])
    y = np.asarray(y, dtype=float)
    A = X.T @ (w[:, None] * X)
    b = np.linalg.solve(A, X.T @ (w * y))
    codes, G = _codes(cluster)
    S = np.zeros((G, 2))
    np.add.at(S, codes, (w * (y - X @ b))[:, None] * X)
    Ai = np.linalg.inv(A)
    V = (G / (G - 1)) * Ai @ (S.T @ S) @ Ai if G > 1 else np.full((2, 2), np.nan)
    se = np.sqrt(np.clip(np.diag(V), 0.0, None)) if G > 1 else np.array([np.nan, np.nan])
    return (float(b[0]), float(se[0])), (float(b[1]), float(se[1]))


def ind_var_wls_slope(x, w, grp, var) -> float:
    w = np.asarray(w, dtype=float)
    xt = _demean(x, w, grp)
    return float((w ** 2 * xt ** 2 * np.asarray(var, dtype=float)).sum() / (w * xt ** 2).sum() ** 2)


def design_effect_mean(resid, w, var, cluster) -> float:
    _, se = weighted_mean_cr1(resid, w, cluster)
    return float(se ** 2 / ind_var_weighted_mean(w, var))


def design_effect_slope(resid, x, w, var, grp, cluster) -> float:
    _, se = wls_slope_cr1(resid, x, w, grp, cluster)
    return float(se ** 2 / ind_var_wls_slope(x, w, grp, var))
