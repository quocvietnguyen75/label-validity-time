"""XF-C7 systematic-error budget components from the static zero line (pilot record, 2026-10-21).

(i) neighbour_spread: for an observation's full bin, the largest |z_b - z_b(neighbour)| over the neighbouring viewpoint
    bins (+-1, circular over the 12 bins) and range bins (+-1) of the same camera, column third and ego class, using
    bins with at least min_n static observations.
(ii) ego_spread: within the observation's (camera, viewpoint, range, column third) bin, max minus min z_b across the
    ego-speed classes with at least min_n static observations (NaN if fewer than two).
Both are in pixels; the caller converts them to ms with the camera's median image speed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K0 = ["camera", "viewpoint_bin", "range_bin", "column_third", "ego_class"]


def neighbour_spread(table0: pd.DataFrame, obs: pd.DataFrame, min_n: int = 30) -> np.ndarray:
    t = table0[table0["n_b"] >= min_n].set_index(K0)["z_b"].to_dict()
    out = np.full(len(obs), np.nan)
    for i, (cam, vp, rg, col, ego) in enumerate(obs[K0].itertuples(index=False, name=None)):
        z = t.get((cam, vp, rg, col, ego))
        if z is None:
            continue
        nbrs = [(cam, (vp - 1) % 12, rg, col, ego), (cam, (vp + 1) % 12, rg, col, ego),
                (cam, vp, rg - 1, col, ego), (cam, vp, rg + 1, col, ego)]
        diffs = [abs(z - t[k]) for k in nbrs if k in t]
        if diffs:
            out[i] = max(diffs)
    return out


def ego_spread(table0: pd.DataFrame, obs: pd.DataFrame, min_n: int = 30) -> np.ndarray:
    t = table0[table0["n_b"] >= min_n]
    g = t.groupby(K0[:4])["z_b"].agg(lambda z: (z.max() - z.min()) if len(z) >= 2 else np.nan).to_dict()
    return np.array([g.get(k, np.nan) for k in obs[K0[:4]].itertuples(index=False, name=None)], dtype=float)
