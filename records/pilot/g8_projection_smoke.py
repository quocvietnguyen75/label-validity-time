"""G8 projection check, pre-freeze part: static vehicles only.

Projects the original 3D boxes of parked vehicles (planar label speed <= 0.5 m/s) with Waymo's official,
unmodified `world_to_image` op and compares the axis-aligned hull of the eight corners with Waymo's
`projected_lidar_box`, under two settings: rolling shutter with the per-image pose, velocity and timing
(how the study projects original boxes) and global shutter with the frame pose (the tutorial's setting).
Reads lidar_box, projected_lidar_box, vehicle_pose, camera_calibration and the metadata columns of
camera_image. Never reads camera_box (human boxes) or masks. Moving objects are not touched before the freeze.

Run inside WSL:  ~/envs/wod/bin/python g8_projection_smoke.py <validation_dir> <n_segments> <out_json>
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "/mnt/d/paper/src")
from careport.xfc7_geometry import box_corners, hull_2d, hull_inside_image, transform_points  # noqa: E402

import tensorflow as tf  # noqa: E402
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops as ops  # noqa: E402

GLOBAL_SHUTTER = 5
CAM = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}
CI, LB, PB, CC, VP = ("[CameraImageComponent]", "[LiDARBoxComponent]", "[ProjectedLiDARBoxComponent]",
                      "[CameraCalibrationComponent]", "[VehiclePoseComponent]")


def read(vdir, comp, seg, cols):
    return pq.read_table(os.path.join(vdir, comp, seg + ".parquet"), columns=cols).to_pandas()


def project(extr, intr, meta, cim, pts):
    out = ops.world_to_image(tf.constant(extr, tf.float64), tf.constant(intr, tf.float64),
                             tf.constant(meta, tf.int32), tf.constant(cim, tf.float64),
                             tf.constant(pts, tf.float64)).numpy()
    return out[:, :2], out[:, 2] > 0.5


def main(vdir, n_seg, out_json):
    segs = sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(vdir, "lidar_box", "*.parquet")))
    assert len(segs) == 202, len(segs)
    rows, timing = [], []
    for seg in segs[:n_seg]:
        t0 = time.time()
        lb = read(vdir, "lidar_box", seg, ["key.frame_timestamp_micros", "key.laser_object_id", f"{LB}.type",
                                           f"{LB}.speed.x", f"{LB}.speed.y", f"{LB}.box.center.x", f"{LB}.box.center.y",
                                           f"{LB}.box.center.z", f"{LB}.box.size.x", f"{LB}.box.size.y",
                                           f"{LB}.box.size.z", f"{LB}.box.heading"])
        lb = lb[(lb[f"{LB}.type"] == 1) & (np.hypot(lb[f"{LB}.speed.x"], lb[f"{LB}.speed.y"]) <= 0.5)]
        pb = read(vdir, "projected_lidar_box", seg, ["key.frame_timestamp_micros", "key.camera_name", "key.laser_object_id",
                                                     f"{PB}.box.center.x", f"{PB}.box.center.y", f"{PB}.box.size.x",
                                                     f"{PB}.box.size.y"])
        m = lb.merge(pb, on=["key.frame_timestamp_micros", "key.laser_object_id"], how="inner")
        cal = read(vdir, "camera_calibration", seg, None).set_index("key.camera_name")
        vp = read(vdir, "vehicle_pose", seg, None).set_index("key.frame_timestamp_micros")
        ci = read(vdir, "camera_image", seg, ["key.frame_timestamp_micros", "key.camera_name", f"{CI}.pose.transform",
                                              f"{CI}.velocity.linear_velocity.x", f"{CI}.velocity.linear_velocity.y",
                                              f"{CI}.velocity.linear_velocity.z", f"{CI}.velocity.angular_velocity.x",
                                              f"{CI}.velocity.angular_velocity.y", f"{CI}.velocity.angular_velocity.z",
                                              f"{CI}.pose_timestamp", f"{CI}.rolling_shutter_params.shutter",
                                              f"{CI}.rolling_shutter_params.camera_trigger_time",
                                              f"{CI}.rolling_shutter_params.camera_readout_done_time"])
        ci = ci.set_index(["key.frame_timestamp_micros", "key.camera_name"])
        for (frame, cam), g in m.groupby(["key.frame_timestamp_micros", "key.camera_name"]):
            c = cal.loc[cam]
            extr = np.array(c[f"{CC}.extrinsic.transform"], dtype=float).reshape(4, 4)
            intr = np.array([c[f"{CC}.intrinsic.{k}"] for k in ["f_u", "f_v", "c_u", "c_v", "k1", "k2", "p1", "p2", "k3"]])
            w, h, rs = int(c[f"{CC}.width"]), int(c[f"{CC}.height"]), int(c[f"{CC}.rolling_shutter_direction"])
            fpose = np.array(vp.loc[frame, f"{VP}.world_from_vehicle.transform"], dtype=float).reshape(4, 4)
            im = ci.loc[(frame, cam)]
            cim_rs = np.concatenate([np.array(im[f"{CI}.pose.transform"], dtype=float),
                                     [im[f"{CI}.velocity.linear_velocity.{a}"] for a in "xyz"],
                                     [im[f"{CI}.velocity.angular_velocity.{a}"] for a in "xyz"],
                                     [im[f"{CI}.pose_timestamp"], im[f"{CI}.rolling_shutter_params.shutter"],
                                      im[f"{CI}.rolling_shutter_params.camera_trigger_time"],
                                      im[f"{CI}.rolling_shutter_params.camera_readout_done_time"]]])
            cim_gs = np.concatenate([fpose.reshape(-1), np.zeros(10)])
            corners = box_corners(g[[f"{LB}.box.center.{a}" for a in "xyz"]].to_numpy(),
                                  g[[f"{LB}.box.size.{a}" for a in "xyz"]].to_numpy(), g[f"{LB}.box.heading"].to_numpy())
            world = transform_points(fpose, corners).reshape(-1, 3)
            waymo = np.stack([g[f"{PB}.box.center.x"] - g[f"{PB}.box.size.x"] / 2, g[f"{PB}.box.center.y"] - g[f"{PB}.box.size.y"] / 2,
                              g[f"{PB}.box.center.x"] + g[f"{PB}.box.size.x"] / 2, g[f"{PB}.box.center.y"] + g[f"{PB}.box.size.y"] / 2], axis=1)
            for model, meta, cim in (("rolling", [w, h, rs], cim_rs), ("global", [w, h, GLOBAL_SHUTTER], cim_gs)):
                uv, ok = project(extr, intr, meta, cim, world)
                hull, allok = hull_2d(uv.reshape(-1, 8, 2), ok.reshape(-1, 8))
                keep = allok & hull_inside_image(hull, w, h, margin=1.0)
                d = hull[keep] - waymo[keep]
                for row in d:
                    rows.append({"camera": CAM[int(cam)], "model": model, "du_min": row[0], "dv_min": row[1],
                                 "du_max": row[2], "dv_max": row[3]})
        timing.append(time.time() - t0)
    df = pd.DataFrame(rows)
    df["max_abs_edge"] = df[["du_min", "dv_min", "du_max", "dv_max"]].abs().max(axis=1)
    summary = (df.groupby(["model", "camera"])["max_abs_edge"]
               .agg(n="size", median="median", p95=lambda x: float(np.quantile(x, 0.95)), max="max").reset_index())
    signed = df.groupby(["model", "camera"])[["du_min", "du_max", "dv_min", "dv_max"]].median().reset_index()
    res = {"segments": segs[:n_seg], "seconds_per_segment": timing, "summary": summary.to_dict(orient="records"),
           "median_signed_edge_diff_px": signed.to_dict(orient="records")}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    print(signed.round(3).to_string(index=False))
    print("seconds per segment:", [round(t, 1) for t in timing])


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
