"""XF-C7 secondary S8 (segment-corrected tau-hat): outcome-blind coverage and predicted precision, before unblinding.

Usage: PYTHONPATH=src python -m careport.xfc7_s8_capacity_run <vehicle_label_only.parquet> <g5_summary.json> <out_dir>
Rebuilds the static residuals as G5 (hash-checked inputs). Honest split: within each segment x camera, static tracks are
split by a seeded coin into halves A and B; intercepts from A are subtracted from B, and the design effect is measured on
B after track-length matching to the moving tracks (as A3). Coverage = share of moving rows whose segment x camera has
an intercept from all static tracks. Reads from the moving set only camera, segment, track keys, n_track, weights.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

from careport.xfc7_estimators import design_effect_mean
from careport.xfc7_gate import subsample_tracks_to_lengths
from careport.xfc7_zero_line import LEVELS, lookup_zero_line, segment_intercepts, zero_line_tables

CAM = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}
SEED = 20261022


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(argv):
    lab_path, g5_path, out_dir = argv[1:4]
    g5 = json.load(open(g5_path, encoding="utf-8"))
    assert sha(lab_path) == g5["inputs"]["label_only_sha256"]
    assert sha(g5["association"]["table"]) == g5["association"]["table_sha256"]
    lab = pd.read_parquet(lab_path)
    assoc = pd.read_parquet(g5["association"]["table"])
    k4 = ["segment", "frame", "camera", "object_id"]
    d = lab.merge(assoc[k4 + ["camera_object_id", "u_ref", "v_ref"]], on=k4, how="left")
    st = d[d["incl_static"] & d["camera_object_id"].notna()].copy()
    st["e"] = (st["p0_u"] - st["u_ref"]) * st["e_u"] + (st["p0_v"] - st["v_ref"]) * st["e_v"]
    tables = zero_line_tables(st)
    st = st.join(lookup_zero_line(st[LEVELS[0]], tables))
    st = st[~st["no_zero_line"]].copy()
    st["r"] = st["e"] - st["z_b"]
    ic_all = segment_intercepts(st)
    mo = pd.read_parquet(g5["moving_analysis_set"]["file"], columns=["segment", "object_id", "camera", "frame", "n_track", "w", "var_tau"])
    cov = mo.merge(ic_all[["segment", "camera"]].assign(has=True), on=["segment", "camera"], how="left")
    rng = np.random.default_rng(SEED)
    tracks = st[["segment", "camera", "object_id"]].drop_duplicates()
    tracks["half"] = rng.integers(0, 2, len(tracks))
    st = st.merge(tracks, on=["segment", "camera", "object_id"])
    icA = segment_intercepts(st[st["half"] == 0])
    B = st[st["half"] == 1].merge(icA[["segment", "camera", "c_seg"]], on=["segment", "camera"], how="inner")
    B["r_corr"] = B["r"] - B["c_seg"]
    out = {"seed": SEED, "intercept_cells": int(len(ic_all)), "cameras": {}}
    for c, g in B.groupby("camera"):
        name = CAM[int(c)]
        if name not in g5["cameras"]:
            continue
        lens = mo[mo["camera"] == c].drop_duplicates(["segment", "object_id"])["n_track"].to_numpy()
        ss = subsample_tracks_to_lengths(g, ["segment", "object_id"], lens, seed=SEED + int(c))
        ss["n_sub"] = ss.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
        ss["w_s"] = 1.0 / (ss["sigma_b"] ** 2 * ss["n_sub"])
        de_raw = design_effect_mean(ss["r"], ss["w_s"], ss["sigma_b"] ** 2, ss["segment"])
        de_corr = design_effect_mean(ss["r_corr"], ss["w_s"], ss["sigma_b"] ** 2, ss["segment"])
        mc = cov[(cov["camera"] == c) & cov["has"].fillna(False)].copy()
        mc["n_track"] = mc.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
        mc["w2"] = mc["w"] * 0 + 1.0
        # weights recomputed on the covered subset (same formula: sdot^2/(sigma_b^2 n_track) = 1/(var_tau n_track))
        mc["w2"] = 1.0 / (mc["var_tau"] * mc["n_track"])
        V = float((mc["w2"] ** 2 * mc["var_tau"]).sum() / mc["w2"].sum() ** 2) if len(mc) else float("nan")
        out["cameras"][name] = {
            "moving_rows": int((cov["camera"] == c).sum()), "moving_rows_covered": int(len(mc)),
            "moving_tracks_covered": int(mc[["segment", "object_id"]].drop_duplicates().shape[0]) if len(mc) else 0,
            "DE_split_half_uncorrected": de_raw, "DE_split_half_corrected": de_corr,
            "predicted_halfwidth_S8_ms": float(1.96 * np.sqrt(de_corr * V)) if len(mc) else float("nan"),
            "predicted_halfwidth_uncorrected_same_rows_ms": float(1.96 * np.sqrt(de_raw * V)) if len(mc) else float("nan")}
    with open(os.path.join(out_dir, "s8_capacity.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
