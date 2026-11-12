"""XF-C7 revision-2 post hoc helpers (pilot record, 2026-10-25, K1-K6; specified there before running).

- stratified_camera, split_camera: camera key 10 x camera + readout code, so that the frozen zero-line code bins by
  camera x readout direction without change (K1);
- cells_mean_cov_cr1: weighted means of several cells with their joint CR1 covariance by cluster (K1 Wald test);
- direction_z, wald_chi2: the between-direction heterogeneity statistics (K1);
- retention_lower_bound: rho >= (lo - bias) / tau_seed (K3);
- clearance: how far an interval lies outside +-tol (negative: the shift that would put it outside) (K1, K5);
- shift_each: shift_along with one delta per observation (K4);
- percentile_rank: share of values at or below a value, in % (K6).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2


def stratified_camera(camera, readout) -> np.ndarray:
    return 10 * np.asarray(camera, dtype=int) + np.asarray(readout, dtype=int)


def split_camera(code) -> Tuple[np.ndarray, np.ndarray]:
    c = np.asarray(code, dtype=int)
    return c // 10, c % 10


def cells_mean_cov_cr1(y, w, cell, cluster) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell weighted means and V_kl = G/(G-1) sum_g s_gk s_gl, s_gk = sum_{i in g, k} w_i (y_i - mu_k) / W_k."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    cc, cells = pd.factorize(pd.Series(np.asarray(cell)), sort=True)
    gc, gs = pd.factorize(pd.Series(np.asarray(cluster)), sort=False)
    K, G = len(cells), len(gs)
    W = np.bincount(cc, weights=w, minlength=K)
    mu = np.bincount(cc, weights=w * y, minlength=K) / W
    S = np.zeros((G, K))
    np.add.at(S, (gc, cc), w * (y - mu[cc]) / W[cc])
    V = (G / (G - 1)) * S.T @ S if G > 1 else np.full((K, K), np.nan)
    return np.asarray(cells), mu, V


def direction_z(est_a: float, se_a: float, est_b: float, se_b: float) -> float:
    return float((est_a - est_b) / np.sqrt(se_a ** 2 + se_b ** 2))


def wald_chi2(diff, cov) -> Tuple[float, int, float]:
    d = np.asarray(diff, dtype=float)
    stat = float(d @ np.linalg.solve(np.asarray(cov, dtype=float), d))
    return stat, int(len(d)), float(chi2.sf(stat, len(d)))


def retention_lower_bound(lo: float, bias: float, tau_seed: float) -> float:
    return float((lo - bias) / tau_seed)


def clearance(lo: float, hi: float, tol: float) -> float:
    """max(lo - tol, -tol - hi): > 0 when the interval is entirely outside +-tol; for an interval clear of zero it is the
    change of tol at which the rejection flips (negative: tol must shrink by that much)."""
    return float(max(lo - tol, -tol - hi))


def shift_each(points, d_hat, sdot, delta_ms) -> np.ndarray:
    k = np.asarray(delta_ms, dtype=float) * np.asarray(sdot, dtype=float)
    return np.asarray(points, dtype=float) - k[:, None] * np.asarray(d_hat, dtype=float)


def percentile_rank(x, value: float) -> float:
    return float(100.0 * (np.asarray(x, dtype=float) <= value).mean())
