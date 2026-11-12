"""Which formula did Waymo use to build camera_synced_box? Label-only reproduction check (pre-freeze).

For moving vehicles (planar speed >= 3 m/s, >= 20 top-LiDAR points), the shift Waymo applied is
dW = (c_s - c) . v_veh / |v_veh|^2 (ms), with the label speed rotated from world axes into the frame's vehicle axes.
It is compared with predictions from timing alone:
  frame-start formula (label.proto):  t_cap - 0
  LiDAR-time formula (LET-3D-AP text): t_cap - t_lidar(azimuth of the box centre in the top-LiDAR frame)
  frame-pose formula:                 t_cap - t_pose (49.85 ms, measured)
t_cap = column time of the centre of Waymo's projected_lidar_box in the most-visible camera (camera_model.cc).
Reads lidar_box, lidar_camera_synced_box, projected_lidar_box, vehicle_pose, lidar_calibration, camera_calibration
and camera_image metadata. Never reads camera_box (human boxes) or masks. Aggregates only.
Usage: PYTHONPATH=src python -m careport.xfc7_synced_check_run <output_dir>
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from careport.xfc7_kinematics import implied_capture_offset
from careport.xfc7_pregate import CAMERA_NAMES
from careport.xfc7_timing_model import azimuth_about, column_time, lidar_sweep_time_ms

CI, CC, LB, SB, PB, VP, LC = ("[CameraImageComponent]", "[CameraCalibrationComponent]", "[LiDARBoxComponent]",
                              "[LiDARCameraSyncedBoxComponent]", "[ProjectedLiDARBoxComponent]", "[VehiclePoseComponent]",
                              "[LiDARCalibrationComponent]")
T_POSE_MS = 49.85
RSD = {2: "LEFT_TO_RIGHT", 4: "RIGHT_TO_LEFT"}


def run_segment(vdir, seg):
    rd = lambda comp, cols=None: pq.read_table(os.path.join(vdir, comp, seg + ".parquet"), columns=cols).to_pandas()
    vp = rd("vehicle_pose")
    R = {int(k): np.array(v, dtype=float).reshape(4, 4)[:3, :3] for k, v in zip(vp["key.frame_timestamp_micros"], vp[f"{VP}.world_from_vehicle.transform"])}
    lc = rd("lidar_calibration")
    top = np.array(lc.loc[lc["key.laser_name"] == 1, f"{LC}.extrinsic.transform"].iloc[0], dtype=float).reshape(4, 4)
    top_inv = np.linalg.inv(top)
    cal = rd("camera_calibration").set_index("key.camera_name")
    lb = rd("lidar_box", ["key.frame_timestamp_micros", "key.laser_object_id", f"{LB}.type", f"{LB}.num_top_lidar_points_in_box",
                          f"{LB}.box.center.x", f"{LB}.box.center.y", f"{LB}.box.center.z", f"{LB}.speed.x", f"{LB}.speed.y", f"{LB}.speed.z"])
    lb = lb[(lb[f"{LB}.type"] == 1) & (lb[f"{LB}.num_top_lidar_points_in_box"] >= 20)
            & (np.hypot(lb[f"{LB}.speed.x"], lb[f"{LB}.speed.y"]) >= 3.0)]
    sb = rd("lidar_camera_synced_box", ["key.frame_timestamp_micros", "key.laser_object_id", f"{SB}.most_visible_camera_name",
                                        f"{SB}.camera_synced_box.center.x", f"{SB}.camera_synced_box.center.y", f"{SB}.camera_synced_box.center.z"])
    pb = rd("projected_lidar_box", ["key.frame_timestamp_micros", "key.camera_name", "key.laser_object_id", f"{PB}.box.center.x"])
    ci = rd("camera_image", ["key.frame_timestamp_micros", "key.camera_name", f"{CI}.rolling_shutter_params.shutter",
                             f"{CI}.rolling_shutter_params.camera_trigger_time", f"{CI}.rolling_shutter_params.camera_readout_done_time"])
    k = ["key.frame_timestamp_micros", "key.laser_object_id"]
    m = lb.merge(sb, on=k).merge(pb.rename(columns={"key.camera_name": f"{SB}.most_visible_camera_name"}),
                                 on=k + [f"{SB}.most_visible_camera_name"])
    m = m.merge(ci.rename(columns={"key.camera_name": f"{SB}.most_visible_camera_name"}),
                on=["key.frame_timestamp_micros", f"{SB}.most_visible_camera_name"])
    if not len(m):
        return pd.DataFrame()
    fr = m["key.frame_timestamp_micros"].to_numpy()
    rot = np.stack([R[int(f)] for f in fr])
    v_veh = np.einsum("nji,nj->ni", rot, m[[f"{LB}.speed.{a}" for a in "xyz"]].to_numpy(dtype=float))
    c = m[[f"{LB}.box.center.{a}" for a in "xyz"]].to_numpy(dtype=float)
    cs = m[[f"{SB}.camera_synced_box.center.{a}" for a in "xyz"]].to_numpy(dtype=float)
    dW, perp = implied_capture_offset(c, cs, v_veh)
    theta = azimuth_about(c, top[:2, 3])   # vehicle axes about the top-LiDAR position (Sun et al. convention)
    t_lidar = lidar_sweep_time_ms(theta)
    f_ms = fr / 1000.0
    cams = m[f"{SB}.most_visible_camera_name"].to_numpy()
    t_cap = np.empty(len(m))
    rsd = np.empty(len(m), dtype=int)
    for cam in np.unique(cams):
        sel = cams == cam
        w = int(cal.loc[cam, f"{CC}.width"])
        d = int(cal.loc[cam, f"{CC}.rolling_shutter_direction"])
        rsd[sel] = d
        t_cap[sel] = column_time(m.loc[sel, f"{PB}.box.center.x"].to_numpy(), w,
                                 m.loc[sel, f"{CI}.rolling_shutter_params.camera_trigger_time"].to_numpy() * 1000.0 - f_ms[sel],
                                 m.loc[sel, f"{CI}.rolling_shutter_params.shutter"].to_numpy() * 1000.0,
                                 m.loc[sel, f"{CI}.rolling_shutter_params.camera_readout_done_time"].to_numpy() * 1000.0 - f_ms[sel], d)
    return pd.DataFrame({"camera": pd.Series(cams).map(CAMERA_NAMES).to_numpy(), "readout": pd.Series(rsd).map(RSD).to_numpy(),
                         "dW_ms": dW * 1000.0, "perp_m": perp, "t_cap_ms": t_cap, "t_lidar_ms": t_lidar,
                         "res_frame": dW * 1000.0 - t_cap, "res_lidar": dW * 1000.0 - (t_cap - t_lidar),
                         "res_pose": dW * 1000.0 - (t_cap - T_POSE_MS), "speed": np.linalg.norm(v_veh[:, :2], axis=1)})


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 64
    vdir = os.path.join(os.environ["WAYMO_V2_ROOT"], "validation")
    segs = sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(vdir, "lidar_box", "*.parquet")))
    assert len(segs) == 202
    df = pd.concat([run_segment(vdir, s) for s in segs], ignore_index=True)
    rows = []
    for (cam, ro), g in df.groupby(["camera", "readout"]):
        row = {"camera": cam, "readout": ro, "n": int(len(g))}
        for h in ("frame", "lidar", "pose"):
            r = g[f"res_{h}"]
            row[f"{h}_median_ms"] = float(r.median())
            row[f"{h}_mad_ms"] = float((r - r.median()).abs().median())
            row[f"{h}_within_2ms"] = float((r.abs() <= 2.0).mean())
        rows.append(row)
    tab = pd.DataFrame(rows)
    allrow = {h: {"median_ms": float(df[f"res_{h}"].median()), "mad_ms": float((df[f"res_{h}"] - df[f"res_{h}"].median()).abs().median()),
                  "within_2ms": float((df[f"res_{h}"].abs() <= 2.0).mean())} for h in ("frame", "lidar", "pose")}
    res = {"n": int(len(df)), "pooled": allrow, "by_camera_readout": tab.to_dict(orient="records"),
           "perp_m_median": float(df["perp_m"].median())}
    os.makedirs(argv[1], exist_ok=True)
    with open(os.path.join(argv[1], "pregate_synced_box_formula.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    pd.set_option("display.width", 250)
    print("n =", len(df)); print(json.dumps(allrow, indent=1))
    print(tab.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
