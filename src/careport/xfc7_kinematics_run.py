"""Run the XF-C7 label-only kinematic checks across the 202 validation segments (pre-freeze; protocol audit A06, A09).

Usage: PYTHONPATH=src python -m careport.xfc7_kinematics_run <output_dir>
Reads WAYMO_V2_ROOT. Uses vehicle_pose, the metadata columns of camera_image (no image bytes), lidar_box and
lidar_camera_synced_box. Never reads camera_box, projected_lidar_box or masks. Writes aggregate tables only.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from careport.xfc7_kinematics import (
    implied_capture_offset,
    pose_time_offset,
    rotate_to_world,
    speed_agreement,
    track_velocity_world,
)
from careport.xfc7_pregate import CAMERA_NAMES

CI, LB, SB, VP = "[CameraImageComponent]", "[LiDARBoxComponent]", "[LiDARCameraSyncedBoxComponent]", "[VehiclePoseComponent]"
MOVING, MIN_TOP_POINTS = 3.0, 20


def _q(x, qs=(0.05, 0.5, 0.95)):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, q)) for q in qs] if len(x) else [float("nan")] * len(qs)


def run_segment(vdir: str, seg: str):
    rd = lambda comp, cols: pq.read_table(os.path.join(vdir, comp, seg + ".parquet"), columns=cols).to_pandas()
    vp = rd("vehicle_pose", ["key.frame_timestamp_micros", f"{VP}.world_from_vehicle.transform"])
    T = {int(k): np.array(v, dtype=float).reshape(4, 4) for k, v in zip(vp["key.frame_timestamp_micros"], vp[f"{VP}.world_from_vehicle.transform"])}
    ci = rd("camera_image", ["key.frame_timestamp_micros", "key.camera_name", f"{CI}.pose.transform", f"{CI}.pose_timestamp",
                             f"{CI}.rolling_shutter_params.camera_trigger_time", f"{CI}.rolling_shutter_params.camera_readout_done_time"])
    # (1) frame-pose time
    pose_off = []
    for fr, g in ci.groupby("key.frame_timestamp_micros"):
        if int(fr) not in T:
            continue
        xyz = np.stack([np.array(p, dtype=float).reshape(4, 4)[:3, 3] for p in g[f"{CI}.pose.transform"]])
        t = g[f"{CI}.pose_timestamp"].to_numpy(dtype=float) - fr / 1e6
        pose_off.append(pose_time_offset(T[int(fr)][:3, 3], t, xyz, min_speed=2.0) * 1000.0)
    # (2) label velocity versus finite differences
    lb = rd("lidar_box", ["key.frame_timestamp_micros", "key.laser_object_id", f"{LB}.type", f"{LB}.num_top_lidar_points_in_box",
                          f"{LB}.box.center.x", f"{LB}.box.center.y", f"{LB}.box.center.z", f"{LB}.speed.x", f"{LB}.speed.y", f"{LB}.speed.z"])
    lb = lb[lb[f"{LB}.type"] == 1].copy()
    fr = lb["key.frame_timestamp_micros"].to_numpy()
    rot = np.stack([T[int(f)][:3, :3] for f in fr]) if len(lb) else np.zeros((0, 3, 3))
    tr = np.stack([T[int(f)][:3, 3] for f in fr]) if len(lb) else np.zeros((0, 3))
    c_v = lb[[f"{LB}.box.center.{a}" for a in "xyz"]].to_numpy(dtype=float)
    v_v = lb[[f"{LB}.speed.{a}" for a in "xyz"]].to_numpy(dtype=float)
    lb["cwx"], lb["cwy"], lb["cwz"] = (np.einsum("nij,nj->ni", rot, c_v) + tr).T
    v_w = rotate_to_world(rot, v_v)
    lb["vwx"], lb["vwy"], lb["vwz"] = v_w.T
    lb["vvx"], lb["vvy"], lb["vvz"] = v_v.T
    lb["t"] = fr / 1e6
    fd_rows = []
    for _, g in lb.sort_values("t").groupby("key.laser_object_id"):
        fd = track_velocity_world(g["t"].to_numpy(), g[["cwx", "cwy", "cwz"]].to_numpy())
        sel = (np.isfinite(fd).all(axis=1) & (np.hypot(g[f"{LB}.speed.x"], g[f"{LB}.speed.y"]).to_numpy() >= MOVING)
               & (g[f"{LB}.num_top_lidar_points_in_box"].to_numpy() >= MIN_TOP_POINTS))
        if sel.any():
            r_v, a_v = speed_agreement(fd[sel], g[["vwx", "vwy", "vwz"]].to_numpy()[sel])
            r_w, a_w = speed_agreement(fd[sel], g[["vvx", "vvy", "vvz"]].to_numpy()[sel])
            fd_rows.append(np.column_stack([r_v, a_v, r_w, a_w]))
    fd_arr = np.vstack(fd_rows) if fd_rows else np.zeros((0, 4))
    # (3) implied capture offset of camera_synced_box
    sb = rd("lidar_camera_synced_box", ["key.frame_timestamp_micros", "key.laser_object_id", f"{SB}.most_visible_camera_name",
                                        f"{SB}.camera_synced_box.center.x", f"{SB}.camera_synced_box.center.y", f"{SB}.camera_synced_box.center.z"])
    m = lb.merge(sb, on=["key.frame_timestamp_micros", "key.laser_object_id"], how="inner")
    m = m[(np.hypot(m[f"{LB}.speed.x"], m[f"{LB}.speed.y"]) >= MOVING) & (m[f"{LB}.num_top_lidar_points_in_box"] >= MIN_TOP_POINTS)]
    cap = pd.DataFrame()
    if len(m):
        # The label speed is in world axes (finite-difference check); express it in the frame's vehicle axes, where
        # c and c_s live: v_veh = R^T v. The raw (unrotated) variant is kept to test whether Waymo mixed frames.
        rot_m = np.stack([T[int(f)][:3, :3] for f in m["key.frame_timestamp_micros"]])
        v_raw = m[["vvx", "vvy", "vvz"]].to_numpy()
        v_veh = np.einsum("nji,nj->ni", rot_m, v_raw)
        c_m = m[[f"{LB}.box.center.{a}" for a in "xyz"]].to_numpy()
        cs_m = m[[f"{SB}.camera_synced_box.center.{a}" for a in "xyz"]].to_numpy()
        off, perp = implied_capture_offset(c_m, cs_m, v_veh)
        off_raw, perp_raw = implied_capture_offset(c_m, cs_m, v_raw)
        win = ci.set_index(["key.frame_timestamp_micros", "key.camera_name"])
        key = list(zip(m["key.frame_timestamp_micros"], m[f"{SB}.most_visible_camera_name"]))
        w = win.reindex(key)
        f_s = m["key.frame_timestamp_micros"].to_numpy() / 1e6
        cap = pd.DataFrame({"camera": m[f"{SB}.most_visible_camera_name"].map(CAMERA_NAMES).to_numpy(),
                            "off_ms": off * 1000.0, "perp_m": perp,
                            "off_raw_ms": off_raw * 1000.0, "perp_raw_m": perp_raw,
                            "win_lo_ms": (w[f"{CI}.rolling_shutter_params.camera_trigger_time"].to_numpy() - f_s) * 1000.0,
                            "win_hi_ms": (w[f"{CI}.rolling_shutter_params.camera_readout_done_time"].to_numpy() - f_s) * 1000.0})
    return np.array(pose_off, dtype=float), fd_arr, cap


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 64
    vdir = os.path.join(os.environ["WAYMO_V2_ROOT"], "validation")
    segs = sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(vdir, "lidar_box", "*.parquet")))
    assert len(segs) == 202, len(segs)
    pose_all, fd_all, cap_all = [], [], []
    for s in segs:
        p, f, c = run_segment(vdir, s)
        pose_all.append(p), fd_all.append(f), cap_all.append(c)
    pose = np.concatenate(pose_all)
    fd = np.vstack(fd_all)
    cap = pd.concat(cap_all, ignore_index=True)
    cap["inside"] = (cap["off_ms"] >= cap["win_lo_ms"] - 2.0) & (cap["off_ms"] <= cap["win_hi_ms"] + 2.0)
    cap_tab = cap.groupby("camera").agg(n=("off_ms", "size"), off_ms_p05=("off_ms", lambda x: np.quantile(x, 0.05)),
                                        off_ms_median=("off_ms", "median"), off_ms_p95=("off_ms", lambda x: np.quantile(x, 0.95)),
                                        inside_readout_window=("inside", "mean"), perp_m_median=("perp_m", "median"),
                                        perp_m_p95=("perp_m", lambda x: np.quantile(x, 0.95)),
                                        raw_off_ms_median=("off_raw_ms", "median"),
                                        raw_perp_m_median=("perp_raw_m", "median")).reset_index()
    res = {
        "frame_pose_time_ms": {"n_frames_ego_ge_2ms": int(np.isfinite(pose).sum()), "p05_median_p95": _q(pose)},
        "label_velocity_vs_finite_difference": {
            "n": int(len(fd)),
            "vehicle_frame_convention": {"ratio_p05_median_p95": _q(fd[:, 0]), "angle_deg_p05_median_p95": _q(fd[:, 1]),
                                         "frac_ratio_outside_0.8_1.25": float(np.mean((fd[:, 0] < 0.8) | (fd[:, 0] > 1.25)))},
            "world_frame_convention": {"ratio_p05_median_p95": _q(fd[:, 2]), "angle_deg_p05_median_p95": _q(fd[:, 3]),
                                       "frac_ratio_outside_0.8_1.25": float(np.mean((fd[:, 2] < 0.8) | (fd[:, 2] > 1.25)))}},
        "implied_capture_offset": cap_tab.to_dict(orient="records"),
    }
    os.makedirs(argv[1], exist_ok=True)
    with open(os.path.join(argv[1], "pregate_kinematics.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "implied_capture_offset"}, indent=1, default=float))
    pd.set_option("display.width", 200)
    print(cap_tab.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
