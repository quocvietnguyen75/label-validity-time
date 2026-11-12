"""G8 (b), (c), (d) for XF-C7 protocol v1.0 (post-freeze, label-only; never reads camera_box or masks).

(b) world_to_image (static point) of the original box corners against projected_lidar_box, moving vehicles.
(c) camera_synced_box reproduction with world_to_image_moving_point: dW = (c_s - c).v_veh/|v_veh|^2 against
    t_cap,op - t_lidar, where t_cap,op is the column time of the moving-point projection of the box centre
    assumed valid at t_lidar; t_lidar under amendment A1 and under the nominal model.
(d) invariance of the op to a world translation of 1e5 m.

Run inside WSL:  ~/envs/wod/bin/python g8_checks.py <validation_dir> <n_segments|all> <out_json>
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "/mnt/d/paper/src")
from careport.xfc7_geometry import box_corners, hull_2d, hull_inside_image, transform_points  # noqa: E402
from careport.xfc7_timing_model import (A1_PERIOD_MS, A1_PHASE_MS, azimuth_about, column_time,  # noqa: E402
                                        lidar_sweep_time_ms)

import tensorflow as tf  # noqa: E402
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops as ops  # noqa: E402

CAM = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}
RSD = {2: "L2R", 4: "R2L"}
CI, LB, PB, CC, VP, SB, LC = ("[CameraImageComponent]", "[LiDARBoxComponent]", "[ProjectedLiDARBoxComponent]",
                              "[CameraCalibrationComponent]", "[VehiclePoseComponent]", "[LiDARCameraSyncedBoxComponent]",
                              "[LiDARCalibrationComponent]")
F64 = tf.float64


def rd(vdir, comp, seg, cols=None):
    return pq.read_table(os.path.join(vdir, comp, seg + ".parquet"), columns=cols).to_pandas()


def calib(c):
    extr = np.array(c[f"{CC}.extrinsic.transform"], dtype=float).reshape(4, 4)
    intr = np.array([c[f"{CC}.intrinsic.{k}"] for k in ["f_u", "f_v", "c_u", "c_v", "k1", "k2", "p1", "p2", "k3"]], dtype=float)
    return extr, intr, int(c[f"{CC}.width"]), int(c[f"{CC}.height"]), int(c[f"{CC}.rolling_shutter_direction"])


def cim_of(im):
    return np.concatenate([np.array(im[f"{CI}.pose.transform"], dtype=float),
                           [im[f"{CI}.velocity.linear_velocity.{a}"] for a in "xyz"],
                           [im[f"{CI}.velocity.angular_velocity.{a}"] for a in "xyz"],
                           [im[f"{CI}.pose_timestamp"], im[f"{CI}.rolling_shutter_params.shutter"],
                            im[f"{CI}.rolling_shutter_params.camera_trigger_time"],
                            im[f"{CI}.rolling_shutter_params.camera_readout_done_time"]]]).astype(float)


def w2i(extr, intr, meta, cim, pts):
    o = ops.world_to_image(tf.constant(extr, F64), tf.constant(intr, F64), tf.constant(meta, tf.int32),
                           tf.constant(cim, F64), tf.constant(pts, F64)).numpy()
    return o[:, :2], o[:, 2] > 0.5


def w2i_moving(extr, intr, meta, cim, pts, vel):
    o = ops.world_to_image_moving_point(tf.constant(extr, F64), tf.constant(intr, F64), tf.constant(meta, tf.int32),
                                        tf.constant(cim, F64), tf.constant(pts, F64), tf.constant(vel, F64)).numpy()
    return o[:, :2], o[:, 2] > 0.5


IMG_COLS = ["key.frame_timestamp_micros", "key.camera_name", f"{CI}.pose.transform",
            f"{CI}.velocity.linear_velocity.x", f"{CI}.velocity.linear_velocity.y", f"{CI}.velocity.linear_velocity.z",
            f"{CI}.velocity.angular_velocity.x", f"{CI}.velocity.angular_velocity.y", f"{CI}.velocity.angular_velocity.z",
            f"{CI}.pose_timestamp", f"{CI}.rolling_shutter_params.shutter",
            f"{CI}.rolling_shutter_params.camera_trigger_time", f"{CI}.rolling_shutter_params.camera_readout_done_time"]


def run_segment(vdir, seg):
    lb = rd(vdir, "lidar_box", seg, ["key.frame_timestamp_micros", "key.laser_object_id", f"{LB}.type",
                                     f"{LB}.num_top_lidar_points_in_box", f"{LB}.speed.x", f"{LB}.speed.y", f"{LB}.speed.z",
                                     f"{LB}.box.center.x", f"{LB}.box.center.y", f"{LB}.box.center.z",
                                     f"{LB}.box.size.x", f"{LB}.box.size.y", f"{LB}.box.size.z", f"{LB}.box.heading"])
    lb = lb[(lb[f"{LB}.type"] == 1) & (np.hypot(lb[f"{LB}.speed.x"], lb[f"{LB}.speed.y"]) >= 3.0)]
    pb = rd(vdir, "projected_lidar_box", seg, ["key.frame_timestamp_micros", "key.camera_name", "key.laser_object_id",
                                               f"{PB}.box.center.x", f"{PB}.box.center.y", f"{PB}.box.size.x", f"{PB}.box.size.y"])
    sb = rd(vdir, "lidar_camera_synced_box", seg, ["key.frame_timestamp_micros", "key.laser_object_id",
                                                   f"{SB}.most_visible_camera_name", f"{SB}.camera_synced_box.center.x",
                                                   f"{SB}.camera_synced_box.center.y", f"{SB}.camera_synced_box.center.z"])
    cal = rd(vdir, "camera_calibration", seg).set_index("key.camera_name")
    lc = rd(vdir, "lidar_calibration", seg)
    top_xy = np.array(lc.loc[lc["key.laser_name"] == 1, f"{LC}.extrinsic.transform"].iloc[0], dtype=float).reshape(4, 4)[:2, 3]
    vp = rd(vdir, "vehicle_pose", seg).set_index("key.frame_timestamp_micros")
    ci = rd(vdir, "camera_image", seg, IMG_COLS).set_index(["key.frame_timestamp_micros", "key.camera_name"])
    b_rows, c_rows = [], []
    key = ["key.frame_timestamp_micros", "key.laser_object_id"]
    # (b) projected boxes, all cameras in which Waymo projected the box
    mb = lb.merge(pb, on=key)
    for (fr, cam), g in mb.groupby(["key.frame_timestamp_micros", "key.camera_name"]):
        extr, intr, w, h, rs = calib(cal.loc[cam])
        T = np.array(vp.loc[fr, f"{VP}.world_from_vehicle.transform"], dtype=float).reshape(4, 4)
        corners = box_corners(g[[f"{LB}.box.center.{a}" for a in "xyz"]].to_numpy(), g[[f"{LB}.box.size.{a}" for a in "xyz"]].to_numpy(),
                              g[f"{LB}.box.heading"].to_numpy())
        im = ci.loc[(fr, cam)]
        cim = cim_of(im)
        f_s = fr / 1e6
        cw8 = transform_points(T, corners)                                   # (N, 8, 3)
        v_w = g[[f"{LB}.speed.{a}" for a in "xyz"]].to_numpy(dtype=float)
        c_v = g[[f"{LB}.box.center.{a}" for a in "xyz"]].to_numpy(dtype=float)
        tl = lidar_sweep_time_ms(azimuth_about(c_v, top_xy), A1_PERIOD_MS, A1_PHASE_MS)
        uc, okc = w2i(extr, intr, [w, h, rs], cim, transform_points(T, c_v))
        trig = (im[f"{CI}.rolling_shutter_params.camera_trigger_time"] - f_s) * 1000.0
        shut = im[f"{CI}.rolling_shutter_params.shutter"] * 1000.0
        done = (im[f"{CI}.rolling_shutter_params.camera_readout_done_time"] - f_s) * 1000.0
        tcap = column_time(uc[:, 0], w, trig, shut, done, rs)
        wb = np.stack([g[f"{PB}.box.center.x"] - g[f"{PB}.box.size.x"] / 2, g[f"{PB}.box.center.y"] - g[f"{PB}.box.size.y"] / 2,
                       g[f"{PB}.box.center.x"] + g[f"{PB}.box.size.x"] / 2, g[f"{PB}.box.center.y"] + g[f"{PB}.box.size.y"] / 2], axis=1)
        sp = np.hypot(g[f"{LB}.speed.x"], g[f"{LB}.speed.y"]).to_numpy()
        variants = {"static": None, "tau_0": np.zeros(len(g)), "tau_lidar": tl, "tau_pose": np.full(len(g), 49.85), "tau_cap": tcap}
        for name, tau in variants.items():
            if tau is None:
                uv, ok = w2i(extr, intr, [w, h, rs], cim, cw8.reshape(-1, 3))
            else:
                x_in = cw8 + (v_w * (im[f"{CI}.pose_timestamp"] - (f_s + tau / 1000.0))[:, None])[:, None, :]
                uv, ok = w2i_moving(extr, intr, [w, h, rs], cim, x_in.reshape(-1, 3), np.repeat(v_w, 8, axis=0))
            hull, allok = hull_2d(uv.reshape(-1, 8, 2), ok.reshape(-1, 8))
            keep = allok & hull_inside_image(hull, w, h, margin=1.0)
            d = hull - wb
            for k in np.where(keep)[0]:
                b_rows.append({"variant": name, "camera": CAM[int(cam)], "readout": RSD[rs], "du_min": d[k, 0], "dv_min": d[k, 1],
                               "du_max": d[k, 2], "dv_max": d[k, 3], "speed": sp[k]})
    # (c) synced-box reproduction, most-visible camera
    mc = lb[lb[f"{LB}.num_top_lidar_points_in_box"] >= 20].merge(sb, on=key)
    for (fr, cam), g in mc.groupby(["key.frame_timestamp_micros", f"{SB}.most_visible_camera_name"]):
        extr, intr, w, h, rs = calib(cal.loc[cam])
        T = np.array(vp.loc[fr, f"{VP}.world_from_vehicle.transform"], dtype=float).reshape(4, 4)
        im = ci.loc[(fr, cam)]
        cim = cim_of(im)
        f_s = fr / 1e6
        c = g[[f"{LB}.box.center.{a}" for a in "xyz"]].to_numpy(dtype=float)
        cs = g[[f"{SB}.camera_synced_box.center.{a}" for a in "xyz"]].to_numpy(dtype=float)
        v_w = g[[f"{LB}.speed.{a}" for a in "xyz"]].to_numpy(dtype=float)
        v_veh = v_w @ T[:3, :3]                                  # R^T v, row form
        dW = ((cs - c) * v_veh).sum(1) / (v_veh * v_veh).sum(1) * 1000.0
        theta = azimuth_about(c, top_xy)
        c_w = transform_points(T, c)
        trig = (im[f"{CI}.rolling_shutter_params.camera_trigger_time"] - f_s) * 1000.0
        shut = im[f"{CI}.rolling_shutter_params.shutter"] * 1000.0
        done = (im[f"{CI}.rolling_shutter_params.camera_readout_done_time"] - f_s) * 1000.0
        for model, tl in (("A1", lidar_sweep_time_ms(theta, A1_PERIOD_MS, A1_PHASE_MS)), ("nominal", lidar_sweep_time_ms(theta, 100.0))):
            x_in = c_w + v_w * (im[f"{CI}.pose_timestamp"] - (f_s + tl / 1000.0))[:, None]
            uv, ok = w2i_moving(extr, intr, [w, h, rs], cim, x_in, v_w)
            tcap = column_time(uv[:, 0], w, trig, shut, done, rs)
            res = dW - (tcap - tl)
            for k in np.where(ok)[0]:
                c_rows.append({"model": model, "segment": seg, "camera": CAM[int(cam)], "readout": RSD[rs], "res_ms": float(res[k]),
                               "speed": float(np.hypot(v_w[k, 0], v_w[k, 1]))})
    return b_rows, c_rows, (cal, vp, ci)


def check_d(cal, vp, ci):
    cam = 1
    extr, intr, w, h, rs = calib(cal.loc[cam])
    fr = vp.index[len(vp) // 2]
    cim = cim_of(ci.loc[(fr, cam)])
    T = np.array(vp.loc[fr, f"{VP}.world_from_vehicle.transform"], dtype=float).reshape(4, 4)
    rng = np.random.default_rng(0)
    pts_v = np.column_stack([rng.uniform(8, 60, 200), rng.uniform(-15, 15, 200), rng.uniform(-1, 3, 200)])
    pts = transform_points(T, pts_v)
    uv0, ok0 = w2i(extr, intr, [w, h, rs], cim, pts)
    shift = np.array([1e5, 1e5, 0.0])
    cim2 = cim.copy()
    pose = cim2[:16].reshape(4, 4)
    pose[:3, 3] += shift
    cim2[:16] = pose.reshape(-1)
    uv1, ok1 = w2i(extr, intr, [w, h, rs], cim2, pts + shift)
    both = ok0 & ok1
    return float(np.max(np.abs(uv1[both] - uv0[both]))), int(both.sum())


def main(vdir, n, out):
    segs = sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(vdir, "lidar_box", "*.parquet")))
    assert len(segs) == 202
    if n != "all":
        segs = segs[:int(n)]
    B, C, ctx = [], [], None
    for s in segs:
        b, c, ctx_s = run_segment(vdir, s)
        B += b
        C += c
        ctx = ctx or ctx_s
    dmax, nd = check_d(*ctx)
    b = pd.DataFrame(B)
    b["max_abs_edge"] = b[["du_min", "dv_min", "du_max", "dv_max"]].abs().max(axis=1)
    bt = b.groupby(["variant", "camera"]).agg(n=("max_abs_edge", "size"), median=("max_abs_edge", "median"),
                                              p95=("max_abs_edge", lambda x: float(np.quantile(x, 0.95))),
                                              dv_median=("dv_min", "median"), du_median=("du_min", "median")).reset_index()
    b["speed_class"] = pd.cut(b["speed"], [3, 8, 15, 60])
    bs = b.groupby(["variant", "camera", "speed_class"], observed=True)["max_abs_edge"].median().reset_index()
    c = pd.DataFrame(C)
    mad = lambda x: float(np.median(np.abs(x - np.median(x))))
    ct = c.groupby(["model", "camera", "readout"]).agg(n=("res_ms", "size"), median_ms=("res_ms", "median"), mad_ms=("res_ms", mad),
                                                       within_1ms=("res_ms", lambda x: float((np.abs(x) <= 1.0).mean()))).reset_index()
    seg_med = (c[c["model"] == "A1"].groupby(["segment", "camera", "readout"])["res_ms"].median().reset_index())
    seg_dist = seg_med.groupby(["camera", "readout"])["res_ms"].agg(
        n_segments="size", p05=lambda x: float(np.quantile(x, 0.05)), median="median", p95=lambda x: float(np.quantile(x, 0.95)),
        frac_abs_gt_5ms=lambda x: float((np.abs(x) > 5).mean())).reset_index()
    seg_med.to_csv(out.replace(".json", "_segment_medians.csv"), index=False)
    res = {"segments": len(segs), "b_projected_box": bt.to_dict(orient="records"), "b_by_speed": bs.astype({"speed_class": str}).to_dict(orient="records"),
           "c_synced_reproduction": ct.to_dict(orient="records"), "c_segment_median_distribution": seg_dist.to_dict(orient="records"),
           "d_translation_invariance_max_px": dmax, "d_points": nd}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    pd.set_option("display.width", 220)
    print(bt.round(3).to_string(index=False))
    print(ct.round(3).to_string(index=False))
    print(seg_dist.round(2).to_string(index=False))
    print("(d) max pixel change under 1e5 m translation:", dmax, "over", nd, "points")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
