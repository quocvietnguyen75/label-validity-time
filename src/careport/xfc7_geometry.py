"""XF-C7 box geometry: upright 3D box corners, rigid transforms and image-space hulls (pure NumPy).

Waymo conventions: box centre and heading in the frame's vehicle coordinates; size = (length along heading,
width, height); heading rotates +x about +z. Transforms are 4x4 row-major matrices (Waymo's `transform` lists).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

# Unit-cube corner signs, ordered: 4 bottom then 4 top.
_SIGNS = np.array([[1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
                   [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]], dtype=float) * 0.5


def box_corners(center: np.ndarray, size: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """Corners (N, 8, 3) of upright boxes with centres (N, 3), sizes (N, 3) = (l, w, h) and headings (N,)."""
    center = np.asarray(center, dtype=float)
    size = np.asarray(size, dtype=float)
    heading = np.asarray(heading, dtype=float)
    local = _SIGNS[None, :, :] * size[:, None, :]
    c, s = np.cos(heading)[:, None], np.sin(heading)[:, None]
    x = c * local[..., 0] - s * local[..., 1]
    y = s * local[..., 0] + c * local[..., 1]
    return np.stack([x, y, local[..., 2]], axis=-1) + center[:, None, :]


def transform_points(t: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply one 4x4 transform to (..., 3) points, or N transforms (N, 4, 4) to (N, K, 3) points."""
    t = np.asarray(t, dtype=float)
    points = np.asarray(points, dtype=float)
    if t.ndim == 2:
        return points @ t[:3, :3].T + t[:3, 3]
    return np.einsum("nij,nkj->nki", t[:, :3, :3], points) + t[:, None, :3, 3]


def hull_2d(uv: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned hull (N, 4) = (u_min, v_min, u_max, v_max) of projected corners (N, 8, 2).

    The second output is True only where all eight corners projected validly.
    """
    uv = np.asarray(uv, dtype=float)
    h = np.concatenate([uv.min(axis=1), uv.max(axis=1)], axis=1)
    return h, np.asarray(valid, dtype=bool).all(axis=1)


def hull_inside_image(hull: np.ndarray, width: float, height: float, margin: float = 0.0) -> np.ndarray:
    """True where the hull lies inside the image with at least `margin` pixels to every border."""
    hull = np.asarray(hull, dtype=float)
    return ((hull[:, 0] >= margin) & (hull[:, 1] >= margin)
            & (hull[:, 2] <= width - margin) & (hull[:, 3] <= height - margin))
