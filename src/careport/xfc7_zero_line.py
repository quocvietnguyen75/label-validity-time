"""XF-C7 static zero line (protocol v1.0 section 5) and the observed time of validity tau-hat.

Bins: camera x viewpoint (12) x range (4) x column third (3) x ego-speed class (3). z_b = median, sigma_b = 1.4826 MAD
of the static residual e_j = (p_j(0) - u_ref,j) . e_j. A moving observation uses its full bin if it holds >= 30 static
observations; otherwise the ego-speed class is collapsed (level 1), then the column third (level 2); if still < 30 the
observation is excluded with `no_zero_line` (pilot-record decision of 2026-10-21).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

LEVELS = [
    ["camera", "viewpoint_bin", "range_bin", "column_third", "ego_class"],
    ["camera", "viewpoint_bin", "range_bin", "column_third"],
    ["camera", "viewpoint_bin", "range_bin"],
]


def _mad(x: pd.Series) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def zero_line_tables(static: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Median, scaled MAD and count of the static residual `e` at each collapse level."""
    out = {}
    for lev, keys in enumerate(LEVELS):
        g = static.groupby(keys)["e"]
        out[lev] = pd.DataFrame({"z_b": g.median(), "sigma_b": g.apply(_mad), "n_b": g.size()}).reset_index()
    return out


def lookup_zero_line(moving: pd.DataFrame, tables: Dict[int, pd.DataFrame], min_n: int = 30) -> pd.DataFrame:
    """z_b, sigma_b, n_b and the level used for each moving observation (row order kept)."""
    res = pd.DataFrame(index=moving.index, data={"z_b": np.nan, "sigma_b": np.nan, "n_b": 0, "zl_level": -1})
    todo = pd.Series(True, index=moving.index)
    for lev, keys in enumerate(LEVELS):
        t = tables[lev]
        t = t[t["n_b"] >= min_n]
        m = moving.loc[todo, keys].reset_index().merge(t, on=keys, how="inner").set_index("index")
        if len(m):
            res.loc[m.index, ["z_b", "sigma_b", "n_b"]] = m[["z_b", "sigma_b", "n_b"]].to_numpy()
            res.loc[m.index, "zl_level"] = lev
            todo.loc[m.index] = False
    res["zl_level"] = res["zl_level"].astype(int)
    res["zl_flagged"] = res["zl_level"] > 0
    res["no_zero_line"] = res["zl_level"] < 0
    return res


def tau_hat(p0: np.ndarray, u_ref: np.ndarray, d_hat: np.ndarray, s: np.ndarray, z_b: np.ndarray,
            sdot: np.ndarray) -> np.ndarray:
    """tau-hat = ((p0 - u_ref) . d_hat - s z_b) / sdot, in ms (protocol v1.0 section 5)."""
    num = ((np.asarray(p0, dtype=float) - np.asarray(u_ref, dtype=float)) * np.asarray(d_hat, dtype=float)).sum(axis=1)
    return (num - np.asarray(s, dtype=float) * np.asarray(z_b, dtype=float)) / np.asarray(sdot, dtype=float)


def segment_intercepts(static_resid: pd.DataFrame, min_obs: int = 20, min_tracks: int = 3) -> pd.DataFrame:
    """Segment x camera offset c_seg = median of the static residual r = e - z_b (secondary S8, protocol v2.0).

    A cell needs at least min_obs static observations from at least min_tracks tracks.
    """
    g = static_resid.groupby(["segment", "camera"])
    t = pd.DataFrame({"c_seg": g["r"].median(), "n": g.size(), "tracks": g["object_id"].nunique()}).reset_index()
    return t[(t["n"] >= min_obs) & (t["tracks"] >= min_tracks)].reset_index(drop=True)


def apply_segment_correction(moving: pd.DataFrame, intercepts: pd.DataFrame) -> pd.DataFrame:
    """tau_hat_seg = tau_hat - s * c_seg / sdot; NaN where the segment x camera cell has no intercept."""
    m = moving.merge(intercepts[["segment", "camera", "c_seg"]], on=["segment", "camera"], how="left")
    m["tau_hat_seg"] = m["tau_hat"] - m["s"] * m["c_seg"] / m["sdot"]
    return m
