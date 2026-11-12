"""Instrument property for the letter: per segment x camera median static residual (px) after the zero line.

Usage: PYTHONPATH=src python segment_offsets.py <vehicle_label_only.parquet> <g5_summary.json> <out_json>
Static (parked-vehicle) data only; no moving residual. Aggregates per segment are written (no per-object values).
"""
import json
import sys

import numpy as np
import pandas as pd

from careport.xfc7_g5_core import build_sets
from careport.xfc7_g5_run import sha
from careport.xfc7_primary import CAM


def main(lab_path, g5_path, out):
    g5 = json.load(open(g5_path, encoding="utf-8"))
    assert sha(lab_path) == g5["inputs"]["label_only_sha256"]
    st, _, _ = build_sets(pd.read_parquet(lab_path), pd.read_parquet(g5["association"]["table"]))
    g = st.groupby(["camera", "segment"])
    seg = pd.DataFrame({"median_px": g["r"].median(), "n": g.size(), "tracks": g["object_id"].nunique()}).reset_index()
    seg = seg[seg["tracks"] >= 3]
    res = {"inputs": {"label_only_sha256": g5["inputs"]["label_only_sha256"], "association_sha256": sha(g5["association"]["table"])},
           "rule": "segment x camera cells with >= 3 parked tracks; median of the static residual after the zero line (px)", "cameras": {}}
    for c, x in seg.groupby("camera"):
        within = st[st["camera"] == c].groupby("segment")["r"].apply(lambda r: float(1.4826 * np.median(np.abs(r - np.median(r)))))
        res["cameras"][CAM[int(c)]] = {"cells": int(len(x)), "median_px_quantiles": {q: float(x["median_px"].quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
                                       "segment_medians_px": sorted(round(float(v), 2) for v in x["median_px"]),
                                       "within_segment_scaled_mad_px_median": float(within.median())}
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    for k, v in res["cameras"].items():
        q = v["median_px_quantiles"]
        print(k, v["cells"], {kk: round(vv, 1) for kk, vv in q.items()}, "within-seg MAD", round(v["within_segment_scaled_mad_px_median"], 1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
