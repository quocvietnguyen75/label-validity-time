"""XF-C7 proposed amendment A3 (outcome-blind): G5 precision gate with a track-length-matched static design effect.

Usage: PYTHONPATH=src python -m careport.xfc7_g5_a3_run <vehicle_label_only.parquet> <g5_summary.json> <out_dir>
Rebuilds the static residuals exactly as G5 (association table and zero-line tables from G5, hash-checked), subsamples
each static track to a contiguous block whose length is drawn from the same camera's moving-track lengths
(seed 20261021 + camera), and recomputes DE and the predicted half-widths with G5's V_ind. Reads from the moving
analysis set only label-derived columns (camera, readout, track keys, n_track); never u_ref, never tau-hat.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

from careport.xfc7_estimators import design_effect_mean, design_effect_slope
from careport.xfc7_gate import subsample_tracks_to_lengths
from careport.xfc7_zero_line import LEVELS, lookup_zero_line, zero_line_tables

CAM = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}
RO = {2: "L2R", 4: "R2L"}
SEED = 20261021


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(argv):
    lab_path, g5_path, out_dir = argv[1:4]
    g5 = json.load(open(g5_path, encoding="utf-8"))
    a_path = g5["association"]["table"]
    assert sha(a_path) == g5["association"]["table_sha256"], "association table changed"
    assert sha(lab_path) == g5["inputs"]["label_only_sha256"], "label-only file changed"
    lab = pd.read_parquet(lab_path)
    assoc = pd.read_parquet(a_path)
    k4 = ["segment", "frame", "camera", "object_id"]
    d = lab.merge(assoc[k4 + ["camera_object_id", "u_ref", "v_ref"]], on=k4, how="left")
    st = d[d["incl_static"] & d["camera_object_id"].notna()].copy()
    st["e"] = (st["p0_u"] - st["u_ref"]) * st["e_u"] + (st["p0_v"] - st["v_ref"]) * st["e_v"]
    tables = zero_line_tables(st)
    zt = pd.concat([t.assign(level=lev) for lev, t in tables.items()], ignore_index=True)
    tmp = os.path.join(out_dir, "_zl_check.parquet")
    zt.to_parquet(tmp, index=False)
    same = sha(tmp) == g5["zero_line"]["tables_sha256"]
    os.remove(tmp)
    st = st.join(lookup_zero_line(st[LEVELS[0]], tables))
    st = st[~st["no_zero_line"]].copy()
    st["r"] = st["e"] - st["z_b"]
    mo = pd.read_parquet(g5["moving_analysis_set"]["file"], columns=["segment", "object_id", "camera", "readout", "n_track"])
    subs = []
    for c, g in st.groupby("camera"):
        lens = mo[mo["camera"] == c].drop_duplicates(["segment", "object_id"])["n_track"].to_numpy()
        if len(lens) == 0:
            continue
        subs.append(subsample_tracks_to_lengths(g, ["segment", "object_id"], lens, seed=SEED + int(c)))
    ss = pd.concat(subs)
    ss["n_sub"] = ss.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
    ss["w"] = 1.0 / (ss["sigma_b"] ** 2 * ss["n_sub"])
    out = {"seed": SEED, "zero_line_tables_rebuilt_identical": bool(same), "static_rows_full": int(len(st)),
           "static_rows_subsampled": int(len(ss)), "cameras": {}, "slope_cells": {}}
    for c, g in ss.groupby("camera"):
        name = CAM[int(c)]
        if name not in g5["cameras"]:
            continue
        de = design_effect_mean(g["r"], g["w"], g["sigma_b"] ** 2, g["segment"])
        V = g5["cameras"][name]["V_ind"]
        hw = 1.96 * np.sqrt(de * V)
        tracks = g5["cameras"][name]["tracks"]
        out["cameras"][name] = {"DE_preregistered": g5["cameras"][name]["DE"], "DE_A3": de,
                                "halfwidth_preregistered_ms": g5["cameras"][name]["predicted_halfwidth_ms"],
                                "halfwidth_A3_ms": float(hw), "tracks": tracks, "decision_camera_A3": bool(tracks >= 30 and hw < 2.5)}
    for r, g in ss.groupby("readout"):
        name = RO[int(r)]
        de = design_effect_slope(g["r"], g["xi"], g["w"], g["sigma_b"] ** 2, g["camera"], g["segment"])
        V = g5["slope_cells"][name]["V_ind"]
        hw = 1.96 * np.sqrt(de * V)
        out["slope_cells"][name] = {"DE_preregistered": g5["slope_cells"][name]["DE"], "DE_A3": de,
                                    "halfwidth_preregistered_ms_per_xi": g5["slope_cells"][name]["predicted_halfwidth_ms_per_xi"],
                                    "halfwidth_A3_ms_per_xi": float(hw), "slope_cell_A3": bool(hw < 5.0)}
    out["decision_cameras_A3"] = [k for k, v in out["cameras"].items() if v["decision_camera_A3"]]
    out["eligible_slope_cells_A3"] = [k for k, v in out["slope_cells"].items() if v["slope_cell_A3"]]
    with open(os.path.join(out_dir, "g5_a3_proposal.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
