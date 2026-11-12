"""XF-C7 label-only kinematic checks (no human image label, no residual).

- pose_time_offset: when the frame's vehicle pose is taken, from the five per-image poses and their timestamps.
- track_velocity_world / rotate_to_world / speed_agreement: finite-difference velocity of box centres in the world
  frame versus the label `speed`, under a stated frame convention.
- implied_capture_offset: the capture-time offset Waymo used for `camera_synced_box`, from c_s - c = v (t_cap - t_frame).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def pose_time_offset(frame_xyz: np.ndarray, image_t: np.ndarray, image_xyz: np.ndarray,
                     min_speed: float = 1.0) -> float:
    """Time (same origin as image_t) at which a linear fit of the image-pose positions passes closest to frame_xyz.

    Returns NaN when the fitted ego speed is below min_speed (m/s), where the time is not identifiable.
    """
    t = np.asarray(image_t, dtype=float)
    x = np.asarray(image_xyz, dtype=float)
    tm = t.mean()
    dt = t - tm
    denom = float((dt ** 2).sum())
    if denom <= 0:
        return float("nan")
    b = (dt[:, None] * (x - x.mean(axis=0))).sum(axis=0) / denom   # velocity (m/s)
    if np.linalg.norm(b) < min_speed:
        return float("nan")
    a = x.mean(axis=0)                                               # position at tm
    return float(tm + np.dot(np.asarray(frame_xyz, dtype=float) - a, b) / np.dot(b, b))


def track_velocity_world(t: np.ndarray, c_world: np.ndarray, max_gap: float = 0.15) -> np.ndarray:
    """Central-difference velocity (N, 3) of one track sorted by time; NaN where a neighbour is missing or too far."""
    t = np.asarray(t, dtype=float)
    c = np.asarray(c_world, dtype=float)
    out = np.full(c.shape, np.nan)
    if len(t) < 3:
        return out
    left = t[1:-1] - t[:-2]
    right = t[2:] - t[1:-1]
    ok = (left > 0) & (right > 0) & (left <= max_gap) & (right <= max_gap)
    v = (c[2:] - c[:-2]) / (t[2:] - t[:-2])[:, None]
    out[1:-1][ok] = v[ok]
    return out


def rotate_to_world(rot: np.ndarray, v_vehicle: np.ndarray) -> np.ndarray:
    """Rotate vehicle-frame vectors (N, 3) into the world frame with per-row rotations (N, 3, 3)."""
    return np.einsum("nij,nj->ni", np.asarray(rot, dtype=float), np.asarray(v_vehicle, dtype=float))


def speed_agreement(v_ref: np.ndarray, v_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Planar magnitude ratio |v_test| / |v_ref| and planar angle between them in degrees (rows)."""
    a = np.asarray(v_ref, dtype=float)[:, :2]
    b = np.asarray(v_test, dtype=float)[:, :2]
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    ratio = nb / na
    cosang = np.clip((a * b).sum(axis=1) / (na * nb), -1.0, 1.0)
    return ratio, np.degrees(np.arccos(cosang))


def implied_capture_offset(c: np.ndarray, c_synced: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Offset (s) with c_synced - c = v * offset, by projection on v, and the perpendicular residual (m)."""
    c, cs, v = (np.asarray(x, dtype=float) for x in (c, c_synced, v))
    d = cs - c
    vv = (v * v).sum(axis=1)
    off = (d * v).sum(axis=1) / vv
    perp = np.linalg.norm(d - off[:, None] * v, axis=1)
    return off, perp
