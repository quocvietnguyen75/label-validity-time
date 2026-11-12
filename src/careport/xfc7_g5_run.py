"""XF-C7 G5 (protocol v1.0 section 3): vehicle association, static zero line, design effects, precision gate, budget.

Usage: PYTHONPATH=src python -m careport.xfc7_g5_run <vehicle_label_only.parquet> <expected_sha256> <summary_dir>
Reads WAYMO_V2_ROOT and camera_box rows of type 1 (vehicle). Writes per-object tables to $WAYMO_V2_ROOT/derived/g5/
(outside Git) with their SHA-256, and an aggregate summary JSON to <summary_dir>.
This step NEVER computes the observed time of validity (tau-hat) or any residual of a moving vehicle: the moving rows
get their association, zero line and weights only. The primary analysis is a separate script, run once.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from careport.xfc7_association import associate_images
from careport.xfc7_budget import ego_spread, neighbour_spread
from careport.xfc7_estimators import design_effect_mean, design_effect_slope, ind_var_wls_slope
from careport.xfc7_zero_line import LEVELS, lookup_zero_line, zero_line_tables

CB = "[CameraBoxComponent]"
CAM = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}
RO = {2: "L2R", 4: "R2L"}
HYP = {"frame": "tau_frame", "lidar": "tau_lidar", "pose": "tau_pose", "cam": "tau_cap"}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_boxes(vdir):
    out = []
    for f in sorted(glob.glob(os.path.join(vdir, "camera_box", "*.parquet"))):
        b = pq.read_table(f, columns=["key.segment_context_name", "key.frame_timestamp_micros", "key.camera_name", "key.camera_object_id",
                                      f"{CB}.box.center.x", f"{CB}.box.center.y", f"{CB}.box.size.x", f"{CB}.type"]).to_pandas()
        b = b[b[f"{CB}.type"] == 1]
        out.append(pd.DataFrame({"segment": b["key.segment_context_name"], "frame": b["key.frame_timestamp_micros"],
                                 "camera": b["key.camera_name"].astype(int), "camera_object_id": b["key.camera_object_id"],
                                 "u": b[f"{CB}.box.center.x"], "v": b[f"{CB}.box.center.y"], "w": b[f"{CB}.box.size.x"]}))
    return pd.concat(out, ignore_index=True)


def attrition(lab, assoc):
    m = lab[lab["incl_moving"] & (lab["speed"] >= 8.0)][["segment", "frame", "camera", "object_id"]]
    m = m.merge(assoc[["segment", "frame", "camera", "object_id", "camera_object_id"]], on=["segment", "frame", "camera", "object_id"], how="left")
    return {CAM[c]: float(g["camera_object_id"].isna().mean()) for c, g in m.groupby("camera")}


def main(argv):
    if len(argv) != 4:
        print(__doc__)
        return 64
    lab_path, expected, summ_dir = argv[1], argv[2], argv[3]
    got = sha(lab_path)
    if got != expected:
        raise SystemExit(f"label-only file hash mismatch: {got} != {expected}")
    root = os.environ["WAYMO_V2_ROOT"]
    odir = os.path.join(root, "derived", "g5")
    os.makedirs(odir, exist_ok=True)
    lab = pd.read_parquet(lab_path)
    boxes = load_boxes(os.path.join(root, "validation"))
    # 1. association with the pre-specified single widening
    runs = {1.0: associate_images(lab, boxes, 1.0)}
    att = {1.0: attrition(lab, runs[1.0])}
    widened = any(v >= 0.30 for v in att[1.0].values())
    if widened:
        runs[1.5] = associate_images(lab, boxes, 1.5)
        att[1.5] = attrition(lab, runs[1.5])
    scale = 1.5 if widened else 1.0
    assoc = runs[scale]
    a_path = os.path.join(odir, "association_primary.parquet")
    assoc.to_parquet(a_path, index=False)
    k4 = ["segment", "frame", "camera", "object_id"]
    d = lab.merge(assoc[k4 + ["camera_object_id", "u_ref", "v_ref", "w_ref", "cost", "margin"]], on=k4, how="left")
    d["assigned"] = d["camera_object_id"].notna()
    # 2. crowding check (not a gate): human vehicle boxes inside the gate of included moving vehicles
    mv = d[d["incl_moving"]]
    bg = {kk: g for kk, g in boxes.groupby(["segment", "frame", "camera"])}
    crowd = []
    for kk, g in mv.groupby(["segment", "frame", "camera"]):
        b = bg.get(kk)
        if b is None:
            crowd.append(pd.Series(0, index=g.index)); continue
        dist = np.hypot(b["u"].to_numpy()[None, :] - g["anchor_u"].to_numpy()[:, None], b["v"].to_numpy()[None, :] - g["anchor_v"].to_numpy()[:, None])
        crowd.append(pd.Series((dist <= scale * g["gate"].to_numpy()[:, None]).sum(axis=1), index=g.index))
    n_in_gate = pd.concat(crowd)
    # 3. static zero line
    st = d[d["incl_static"] & d["assigned"]].copy()
    st["e"] = (st["p0_u"] - st["u_ref"]) * st["e_u"] + (st["p0_v"] - st["v_ref"]) * st["e_v"]
    tables = zero_line_tables(st)
    z_path = os.path.join(odir, "zero_line_tables.parquet")
    pd.concat([t.assign(level=lev) for lev, t in tables.items()], ignore_index=True).to_parquet(z_path, index=False)
    zs = lookup_zero_line(st[LEVELS[0]], tables)
    st = st.join(zs)
    st = st[~st["no_zero_line"]].copy()
    st["r"] = st["e"] - st["z_b"]
    st["n_track"] = st.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
    st["w"] = 1.0 / (st["sigma_b"] ** 2 * st["n_track"])
    # 4. moving analysis set: association + zero line + weights (no tau-hat here)
    mo = d[d["incl_moving"] & d["assigned"]].copy()
    zm = lookup_zero_line(mo[LEVELS[0]], tables)
    mo = mo.join(zm)
    n_no_zl = int(mo["no_zero_line"].sum())
    mo = mo[~mo["no_zero_line"] & (mo["sigma_b"] > 0)].copy()
    mo["n_track"] = mo.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
    mo["w"] = mo["sdot"] ** 2 / (mo["sigma_b"] ** 2 * mo["n_track"])
    mo["var_tau"] = mo["sigma_b"] ** 2 / mo["sdot"] ** 2
    m_path = os.path.join(odir, "moving_analysis_set.parquet")
    keep = k4 + ["readout", "speed", "sdot", "s", "d_u", "d_v", "p0_u", "p0_v", "u_ref", "v_ref", "xi", "margin", "z_b", "sigma_b",
                 "zl_level", "zl_flagged", "n_track", "w", "var_tau", "tau_frame", "tau_lidar", "tau_pose", "tau_cap",
                 "viewpoint_bin", "range_bin", "column_third", "ego_class", "ego_speed", "range_m", "top_points", "all_points"]
    mo[keep].to_parquet(m_path, index=False)
    # 5. design effects, predicted half-widths, decision cameras and slope cells
    cams, cells = {}, {}
    for c, g in mo.groupby("camera"):
        sc = st[st["camera"] == c]
        de = design_effect_mean(sc["r"], sc["w"], sc["sigma_b"] ** 2, sc["segment"])
        V = float((g["w"] ** 2 * g["var_tau"]).sum() / g["w"].sum() ** 2)
        tracks = int(g[["segment", "object_id"]].drop_duplicates().shape[0])
        hw = 1.96 * np.sqrt(de * V)
        cams[CAM[c]] = {"obs": int(len(g)), "tracks": tracks, "segments": int(g["segment"].nunique()), "DE": de, "V_ind": V,
                        "predicted_halfwidth_ms": float(hw), "decision_camera": bool(tracks >= 30 and hw < 2.5),
                        "median_sdot": float(g["sdot"].median()),
                        "predicted_tau_bar": {h: float((g["w"] * g[col]).sum() / g["w"].sum()) for h, col in HYP.items()}}
    for r, g in mo.groupby("readout"):
        sr = st[st["readout"] == r]
        de = design_effect_slope(sr["r"], sr["xi"], sr["w"], sr["sigma_b"] ** 2, sr["camera"], sr["segment"])
        V = ind_var_wls_slope(g["xi"], g["w"], g["camera"], g["var_tau"])
        hw = 1.96 * np.sqrt(de * V)
        cells[RO[r]] = {"obs": int(len(g)), "tracks": int(g[["segment", "object_id"]].drop_duplicates().shape[0]), "DE": de,
                        "V_ind": float(V), "predicted_halfwidth_ms_per_xi": float(hw), "slope_cell": bool(hw < 5.0)}
    # 6. systematic budget components (static-only quantities and label-only bin shares), in ms
    t0 = tables[0]
    budget = {}
    for c, g in mo.groupby("camera"):
        sdot_med = float(g["sdot"].median())
        nb = np.nanmean(neighbour_spread(t0, g[LEVELS[0]]))
        es = np.nanmean(ego_spread(t0, g[LEVELS[0]]))
        sc = st[st["camera"] == c]
        X = np.column_stack([np.ones(len(sc)), sc["ego_speed"].to_numpy()])
        W = sc["w"].to_numpy()
        beta = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (W * sc["r"].to_numpy()))[1]
        dspeed = float(g["ego_speed"].median() - sc["ego_speed"].median())
        budget[CAM[c]] = {"neighbour_bins_ms": float(nb / sdot_med), "ego_class_spread_ms": float(es / sdot_med),
                          "ego_speed_slope_ms": float(abs(beta * dspeed) / sdot_med),
                          "static_slope_px_per_mps": float(beta), "ego_speed_median_diff_mps": dspeed}
    res = {
        "inputs": {"label_only_sha256": got, "human_vehicle_boxes": int(len(boxes))},
        "association": {"attrition_ge8ms_scale1": att[1.0], "widened": widened,
                        "attrition_ge8ms_scale1_5": att.get(1.5), "primary_gate_scale": scale,
                        "table": a_path, "table_sha256": sha(a_path)},
        "crowding_check": {"included_moving": int(len(mv)), "share_ge2_human_boxes_in_gate": float((n_in_gate >= 2).mean()),
                           "share_0_in_gate": float((n_in_gate == 0).mean())},
        "assigned": {"included_moving": int(len(mv)), "moving_assigned": int(mv["assigned"].sum()),
                     "static_included": int(d["incl_static"].sum()), "static_assigned": int((d["incl_static"] & d["assigned"]).sum())},
        "zero_line": {"tables": z_path, "tables_sha256": sha(z_path), "static_used": int(len(st)),
                      "moving_levels": {int(k): int(v) for k, v in zm["zl_level"].value_counts().items()},
                      "moving_no_zero_line_excluded": n_no_zl,
                      "sigma_b_median_px": float(mo["sigma_b"].median())},
        "moving_analysis_set": {"file": m_path, "sha256": sha(m_path), "rows": int(len(mo)),
                                "tracks": int(mo[["segment", "object_id"]].drop_duplicates().shape[0])},
        "cameras": cams, "slope_cells": cells, "budget": budget,
        "decision_cameras": [k for k, v in cams.items() if v["decision_camera"]],
        "eligible_slope_cells": [k for k, v in cells.items() if v["slope_cell"]],
    }
    os.makedirs(summ_dir, exist_ok=True)
    with open(os.path.join(summ_dir, "g5_summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
