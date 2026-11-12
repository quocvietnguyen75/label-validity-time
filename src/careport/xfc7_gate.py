"""XF-C7 proposed amendment A3: a track-length-matched static design effect for the G5 precision gate.

The pre-registered proxy computes the design effect on static tracks whose frames are near-duplicates, so it scales
with static track length (often 100+ frames) rather than with the moving tracks (about 10 frames) it stands in for.
A3 subsamples each static track to a contiguous block whose length is drawn from the moving tracks' length
distribution of the same camera (seeded), keeping segment clustering as it is. Outcome-blind: static data only.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def subsample_tracks_to_lengths(df: pd.DataFrame, track_keys: Sequence[str], lengths: np.ndarray, seed: int,
                                order_col: str = "frame") -> pd.DataFrame:
    """Keep, per track, a contiguous block (by `order_col`) of a length drawn from `lengths`; shorter tracks are kept whole."""
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths, dtype=int)
    parts = []
    for _, g in df.sort_values(list(track_keys) + [order_col]).groupby(list(track_keys), sort=True):
        m = int(rng.choice(lengths))
        n = len(g)
        if n <= m:
            parts.append(g)
            continue
        start = int(rng.integers(0, n - m + 1))
        parts.append(g.iloc[start:start + m])
    return pd.concat(parts) if parts else df.iloc[0:0]
