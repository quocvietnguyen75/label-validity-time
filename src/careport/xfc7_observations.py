"""XF-C7 label-only observation logic (protocol v1.0 sections 4-5; pure NumPy, no Waymo op).

Definitions pinned here (the protocol leaves them to the implementation; recorded in the pilot record):
- range: distance in world axes from the camera position to the box centre;
- camera position: image pose (world_from_vehicle at pose_timestamp) composed with the camera extrinsic translation;
- viewpoint: box heading relative to the camera-to-box ray, signed, in [-180, 180) deg, 12 bins of 30 deg
  (bin 6 = moving straight away from the camera along the ray, bin 0 = towards it);
- ego-speed class: magnitude of the image's linear velocity, classes < 2, 2-8, >= 8 m/s;
- track: laser_object_id within a segment;
- range bins: [0, 15), [15, 30), [30, 50), [50, inf) m; the protocol's 50-80 m bin is extended to the label range;
- column third: floor(3 u / W) of the hull centre, clipped to 0..2.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def motion_sign(v: np.ndarray, heading_unit: np.ndarray) -> np.ndarray:
    """+1 if the velocity points along the body-forward axis, -1 if against (reversing)."""
    d = (np.asarray(v, dtype=float) * np.asarray(heading_unit, dtype=float)).sum(axis=1)
    return np.where(d >= 0, 1, -1)


def image_speed(p0: np.ndarray, p10: np.ndarray, d_hat: np.ndarray) -> np.ndarray:
    """Image speed along d_hat in px/ms, from the projections at tau = 0 and tau = 10 ms."""
    return ((np.asarray(p0, dtype=float) - np.asarray(p10, dtype=float)) * np.asarray(d_hat, dtype=float)).sum(axis=1) / 10.0


def anchor_and_gate(p0: np.ndarray, pmax: np.ndarray, w0: np.ndarray, wmax: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Association anchor (midpoint) and gate radius in px (protocol section 4)."""
    p0 = np.asarray(p0, dtype=float)
    pmax = np.asarray(pmax, dtype=float)
    anchor = 0.5 * (p0 + pmax)
    gate = 0.5 * np.linalg.norm(p0 - pmax, axis=1) + 40.0 + 0.25 * 0.5 * (np.asarray(w0, dtype=float) + np.asarray(wmax, dtype=float))
    return anchor, gate


def translate_hull(hull: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Axis-aligned hull (u0, v0, u1, v1) moved so that its centre is `centre` (size kept)."""
    h = np.asarray(hull, dtype=float)
    c = np.asarray(centre, dtype=float)
    half = 0.5 * np.stack([h[:, 2] - h[:, 0], h[:, 3] - h[:, 1]], axis=1)
    return np.concatenate([c - half, c + half], axis=1)


def hull_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix (N, M) between axis-aligned boxes (u0, v0, u1, v1)."""
    a = np.asarray(a, dtype=float)[:, None, :]
    b = np.asarray(b, dtype=float)[None, :, :]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def overlap_flags(hulls: np.ndarray, image_ids: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """True where another hull in the same image overlaps with IoU > threshold."""
    hulls = np.asarray(hulls, dtype=float)
    ids = np.asarray(image_ids)
    out = np.zeros(len(hulls), dtype=bool)
    for img in np.unique(ids):
        idx = np.where(ids == img)[0]
        if len(idx) < 2:
            continue
        m = hull_iou(hulls[idx], hulls[idx])
        np.fill_diagonal(m, 0.0)
        out[idx] = (m > threshold).any(axis=1)
    return out


def along_axis_angle_deg(v: np.ndarray, cam_pos: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Acute 3D angle (deg) between the velocity and the camera-to-centre line of sight."""
    v = np.asarray(v, dtype=float)
    los = np.asarray(centre, dtype=float) - np.asarray(cam_pos, dtype=float)
    cosang = np.abs((v * los).sum(axis=1)) / (np.linalg.norm(v, axis=1) * np.linalg.norm(los, axis=1))
    return np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0)))


def viewpoint_bin(heading_xy: np.ndarray, ray_xy: np.ndarray) -> np.ndarray:
    """12 bins of 30 deg of the signed heading relative to the camera-to-box ray, in [-180, 180)."""
    h = np.asarray(heading_xy, dtype=float)
    r = np.asarray(ray_xy, dtype=float)
    rel = np.degrees(np.arctan2(h[:, 1], h[:, 0]) - np.arctan2(r[:, 1], r[:, 0]))
    rel = (rel + 180.0) % 360.0 - 180.0
    return np.clip(np.floor((rel + 180.0) / 30.0).astype(int), 0, 11)


def range_bin(dist_m: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(dist_m, dtype=float), [15.0, 30.0, 50.0])


def column_third(u: np.ndarray, width: float) -> np.ndarray:
    return np.clip(np.floor(3.0 * np.asarray(u, dtype=float) / width).astype(int), 0, 2)


def ego_speed_class(speed: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(speed, dtype=float), [2.0, 8.0])


def frustum_margin_deg(u, v, f_u, f_v, c_u, c_v, width, height) -> np.ndarray:
    """Angular margin (deg) of image points to the nearest frustum edge, pinhole approximation (robustness run)."""
    au = np.degrees(np.arctan(np.abs(np.asarray(u, dtype=float) - c_u) / f_u))
    av = np.degrees(np.arctan(np.abs(np.asarray(v, dtype=float) - c_v) / f_v))
    hu = np.degrees(np.arctan((width / 2.0) / f_u))
    hv = np.degrees(np.arctan((height / 2.0) / f_v))
    return np.minimum(hu - au, hv - av)
