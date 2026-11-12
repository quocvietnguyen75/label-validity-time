"""XF-C7 post hoc helpers (pilot record, 2026-10-22; specified there before running).

- tipping_factor: the largest factor k on B_c at which an interval is still entirely outside +-(5 + k B_c);
- ruled_out_up_to: the factor below which a hypothesis stays rejected in >= 2 cameras (second-largest k);
- n_in_gate: number of human box centres inside each observation's gate (the G5 crowding count);
- compose_budget: B_c from its three components under max (frozen v2.0), quadrature or linear sum (revision B1);
- clip_boxes, paired_iou: row-wise IoU of axis-aligned boxes clipped to the image (revision B5);
- shift_along: move image points by -delta * sdot along d_hat, which raises tau-hat by delta (revision B4);
- cell_mean_ci, closest_row, crop_window: the paper displays D2 and D3 (pilot record, 2026-10-24).
"""
from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np

from careport.xfc7_estimators import weighted_mean_cr1


def tipping_factor(lo: float, hi: float, budget: float, eps: float = 5.0) -> float:
    if lo > eps:
        return (lo - eps) / budget
    if hi < -eps:
        return (-hi - eps) / budget
    return 0.0


def ruled_out_up_to(factors: Mapping[str, float]) -> float:
    k = sorted(factors.values(), reverse=True)
    return float(k[1]) if len(k) >= 2 else 0.0


def n_in_gate(anchors: np.ndarray, gates: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    a = np.asarray(anchors, dtype=float)
    b = np.asarray(boxes, dtype=float).reshape(-1, 2)
    if len(b) == 0:
        return np.zeros(len(a), dtype=int)
    d = np.hypot(b[None, :, 0] - a[:, None, 0], b[None, :, 1] - a[:, None, 1])
    return (d <= np.asarray(gates, dtype=float)[:, None]).sum(axis=1)


def compose_budget(components, mode: str) -> float:
    c = np.asarray(components, dtype=float)
    if mode == "max":
        return float(c.max())
    if mode == "quadrature":
        return float(np.sqrt((c ** 2).sum()))
    if mode == "sum":
        return float(c.sum())
    raise ValueError(mode)


def clip_boxes(boxes: np.ndarray, width, height) -> np.ndarray:
    b = np.asarray(boxes, dtype=float).copy()
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0.0, np.asarray(width, dtype=float)[..., None] if np.ndim(width) else width)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0.0, np.asarray(height, dtype=float)[..., None] if np.ndim(height) else height)
    return b


def paired_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    iw = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def shift_along(points: np.ndarray, d_hat: np.ndarray, sdot: np.ndarray, delta_ms: float) -> np.ndarray:
    return np.asarray(points, dtype=float) - delta_ms * np.asarray(sdot, dtype=float)[:, None] * np.asarray(d_hat, dtype=float)


def cell_mean_ci(y, w, cluster, min_clusters: int = 5) -> Dict[str, float]:
    """Weighted mean with its 95 % CR1 interval; the interval is NaN when the cell has fewer than min_clusters clusters."""
    est, se = weighted_mean_cr1(y, w, cluster)
    g = int(len(np.unique(np.asarray(cluster))))
    ok = g >= min_clusters
    return {"est": est, "lo": est - 1.96 * se if ok else float("nan"), "hi": est + 1.96 * se if ok else float("nan"),
            "obs": int(len(np.asarray(y))), "clusters": g}


def closest_row(values, target: float, *tie_keys) -> int:
    """Position of the value closest to target; ties go to the smallest tie keys, in the order given."""
    dist = np.abs(np.asarray(values, dtype=float) - target)
    order = np.lexsort(tuple(np.asarray(k) for k in reversed(tie_keys)) + (dist,))
    return int(order[0])


def crop_window(boxes, pad: float, min_width: int, width: int, height: int) -> Tuple[int, int, int, int]:
    """Integer crop around the union of boxes, padded by pad x its size on each side, widened to min_width, kept in the image."""
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    u0, v0, u1, v1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    pu, pv = pad * (u1 - u0), pad * (v1 - v0)
    u0, u1, v0, v1 = u0 - pu, u1 + pu, v0 - pv, v1 + pv
    if u1 - u0 < min_width:
        c = 0.5 * (u0 + u1)
        u0, u1 = c - 0.5 * min_width, c + 0.5 * min_width
    u0, v0, u1, v1 = (int(round(x)) for x in (u0, v0, u1, v1))
    if u0 < 0:
        u0, u1 = 0, u1 - u0
    if u1 > width:
        u0, u1 = u0 - (u1 - width), width
    return max(u0, 0), max(v0, 0), min(u1, width), min(v1, height)
