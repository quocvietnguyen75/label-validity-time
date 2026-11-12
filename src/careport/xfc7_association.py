"""XF-C7 association of projected 3D boxes with human 2D boxes (protocol v1.0 section 4).

cost = |u_ref - anchor| / gate + |ln(w_ref / w_hull)|, infinite outside the gate; Hungarian per image over the full
roster; infinite costs are replaced by a large constant (so the number of finite matches is maximised first) and
post-filtered. Margin = second-best finite cost of the row minus its assigned cost (+inf if no other finite candidate).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

BIG = 1.0e6


def cost_matrix(anchors: np.ndarray, gates: np.ndarray, hull_w: np.ndarray,
                refs: np.ndarray, ref_w: np.ndarray) -> np.ndarray:
    a = np.asarray(anchors, dtype=float)
    r = np.asarray(refs, dtype=float)
    g = np.asarray(gates, dtype=float)
    d = np.linalg.norm(a[:, None, :] - r[None, :, :], axis=2)
    c = d / g[:, None] + np.abs(np.log(np.asarray(ref_w, dtype=float)[None, :] / np.asarray(hull_w, dtype=float)[:, None]))
    c[d > g[:, None]] = np.inf
    return c


def associate_images(roster, boxes, gate_scale: float = 1.0):
    """Match roster rows (segment, frame, camera, object_id, anchor_u, anchor_v, gate, hull_w0, hull_wmax) to human
    boxes (segment, frame, camera, camera_object_id, u, v, w) image by image. Returns one row per roster row with
    camera_object_id (NaN if unassigned), u_ref, v_ref, w_ref, cost and margin."""
    import pandas as pd
    key = ["segment", "frame", "camera"]
    bgroups = {k: g for k, g in boxes.groupby(key)}
    out = []
    for k, g in roster.groupby(key):
        res = pd.DataFrame({"segment": k[0], "frame": k[1], "camera": k[2], "object_id": g["object_id"].to_numpy(),
                            "camera_object_id": np.nan, "u_ref": np.nan, "v_ref": np.nan, "w_ref": np.nan,
                            "cost": np.nan, "margin": np.nan})
        res["camera_object_id"] = res["camera_object_id"].astype(object)
        b = bgroups.get(k)
        if b is not None and len(b):
            hw = 0.5 * (g["hull_w0"].to_numpy() + g["hull_wmax"].to_numpy())
            c = cost_matrix(g[["anchor_u", "anchor_v"]].to_numpy(), gate_scale * g["gate"].to_numpy(), hw,
                            b[["u", "v"]].to_numpy(), b["w"].to_numpy())
            col, margin = assign_image(c)
            ok = col >= 0
            bi = b.reset_index(drop=True)
            res.loc[ok, "camera_object_id"] = bi.loc[col[ok], "camera_object_id"].to_numpy()
            res.loc[ok, "u_ref"] = bi.loc[col[ok], "u"].to_numpy()
            res.loc[ok, "v_ref"] = bi.loc[col[ok], "v"].to_numpy()
            res.loc[ok, "w_ref"] = bi.loc[col[ok], "w"].to_numpy()
            res.loc[ok, "cost"] = c[np.where(ok)[0], col[ok]]
            res.loc[ok, "margin"] = margin[ok]
        out.append(res)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def assign_image(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Column index per row (-1 if unassigned) and the cost margin (NaN if unassigned)."""
    cost = np.asarray(cost, dtype=float)
    n = cost.shape[0]
    m = cost.shape[1] if cost.ndim == 2 else 0
    col = np.full(n, -1, dtype=int)
    margin = np.full(n, np.nan)
    if n == 0 or m == 0:
        return col, margin
    finite = np.isfinite(cost)
    rows, cols = linear_sum_assignment(np.where(finite, cost, BIG))
    for i, j in zip(rows, cols):
        if finite[i, j]:
            col[i] = j
    for i in np.where(col >= 0)[0]:
        others = np.delete(cost[i], col[i])
        others = others[np.isfinite(others)]
        margin[i] = (others.min() - cost[i, col[i]]) if len(others) else np.inf
    return col, margin
