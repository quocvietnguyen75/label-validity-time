"""XF-C7 gate G3b: capture time of each top-LiDAR range-image column, from the per-pixel vehicle pose.

Waymo releases no per-point LiDAR timestamps, but `lidar_pose` gives, for every top-LiDAR pixel with a return,
the vehicle pose at the moment that pixel was captured (channels roll, pitch, yaw, x, y, z). Matching the
per-column vehicle position to the ego trajectory from the timestamped image poses recovers the column's time.
Range-image column j points at vehicle azimuth pi - 2 pi (j + 0.5) / W (range_image_utils), so under the
Sun et al. convention column j is captured at T (j + 0.5) / W after the frame start.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def column_positions(pose_img: np.ndarray) -> np.ndarray:
    """Median vehicle position (W, 3) per column over rows with a return (non-zero xyz); NaN if none."""
    xyz = np.asarray(pose_img, dtype=float)[..., 3:6]
    valid = np.abs(xyz).sum(axis=-1) > 0
    out = np.full((xyz.shape[1], 3), np.nan)
    masked = np.where(valid[..., None], xyz, np.nan)
    has = valid.any(axis=0)
    if has.any():
        out[has] = np.nanmedian(masked[:, has, :], axis=0)
    return out


def column_times(col_xyz: np.ndarray, image_t: np.ndarray, image_xyz: np.ndarray,
                 min_speed: float = 2.0) -> np.ndarray:
    """Time of closest approach of a linear fit to the image-pose trajectory, for each column position.

    Returns NaN for columns without a position, and everywhere when the fitted ego speed is below min_speed.
    """
    p = np.asarray(col_xyz, dtype=float)
    t = np.asarray(image_t, dtype=float)
    x = np.asarray(image_xyz, dtype=float)
    out = np.full(len(p), np.nan)
    if len(t) < 2:
        return out
    tm = t.mean()
    dt = t - tm
    denom = float((dt ** 2).sum())
    if denom <= 0:
        return out
    b = (dt[:, None] * (x - x.mean(axis=0))).sum(axis=0) / denom
    if np.linalg.norm(b) < min_speed:
        return out
    ok = np.isfinite(p).all(axis=1)
    out[ok] = tm + (p[ok] - x.mean(axis=0)) @ b / float(b @ b)
    return out


def fit_sweep(cols: np.ndarray, t: np.ndarray) -> Tuple[float, float, float, int]:
    """Least-squares t = a + b * col over finite entries: (a, b, MAD of residuals, n used)."""
    c = np.asarray(cols, dtype=float)
    y = np.asarray(t, dtype=float)
    ok = np.isfinite(c) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan"), float("nan"), float("nan"), int(ok.sum())
    b, a = np.polyfit(c[ok], y[ok], 1)
    r = y[ok] - (a + b * c[ok])
    return float(a), float(b), float(np.median(np.abs(r - np.median(r)))), int(ok.sum())
