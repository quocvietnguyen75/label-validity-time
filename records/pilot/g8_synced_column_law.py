"""G8(c) diagnostic: which column-time law do Waymo's camera_synced_box shifts follow? (label-only; no camera_box)

For moving vehicles (planar speed >= 3 m/s, >= 20 top-LiDAR points) with a synced box, Waymo's implied capture time is
t_W = t_lidar + dW, with dW = (c_s - c).v_veh/|v_veh|^2 and t_lidar from amendment A1. If the synced box is
self-consistent, t_W equals the column time of the synced box's own projection, u_s = world_to_image(T_frame c_s),
under the segment's readout direction (Waymo's GetPixelTimestamp). The script reports, per camera x readout:
median and WLS slope on xi = u_s/W - 0.5 of (t_W - t_col) for three laws: the segment's own direction, left-to-right
for every image, and the image-centre time (pose_timestamp).
Run inside WSL:  ~/envs/wod/bin/python g8_synced_column_law.py <validation_dir> <n|all> <out_json>
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/d/paper/newpaper/autonomous_driving_crossmodal_fit_selection_2026-10-13/pilot")
sys.path.insert(0, "/mnt/d/paper/src")
import g8_checks as G  # noqa: E402
from careport.xfc7_geometry import transform_points  # noqa: E402
from careport.xfc7_timing_model import (A1_PERIOD_MS, A1_PHASE_MS, LEFT_TO_RIGHT, azimuth_about,  # noqa: E402
                                        column_time, lidar_sweep_time_ms)


def run(vdir, seg):
    LB, SB = G.LB, G.SB
    lb = G.rd(vdir, "lidar_box", seg, ["key.frame_timestamp_micros", "key.laser_object_id", f"{LB}.type",
                                       f"{LB}.num_top_lidar_points_in_box", f"{LB}.speed.x", f"{LB}.speed.y", f"{LB}.speed.z",
                                       f"{LB}.box.center.x", f"{LB}.box.center.y", f"{LB}.box.center.z"])
    lb = lb[(lb[f"{LB}.type"] == 1) & (np.hypot(lb[f"{LB}.speed.x"], lb[f"{LB}.speed.y"]) >= 3.0)
            & (lb[f"{LB}.num_top_lidar_points_in_box"] >= 20)]
    sb = G.rd(vdir, "lidar_camera_synced_box", seg, ["key.frame_timestamp_micros", "key.laser_object_id",
                                                     f"{SB}.most_visible_camera_name", f"{SB}.camera_synced_box.center.x",
                                                     f"{SB}.camera_synced_box.center.y", f"{SB}.camera_synced_box.center.z"])
    m = lb.merge(sb, on=["key.frame_timestamp_micros", "key.laser_object_id"])
    if not len(m):
        return []
    cal = G.rd(vdir, "camera_calibration", seg).set_index("key.camera_name")
    vp = G.rd(vdir, "vehicle_pose", seg).set_index("key.frame_timestamp_micros")
    ci = G.rd(vdir, "camera_image", seg, G.IMG_COLS).set_index(["key.frame_timestamp_micros", "key.camera_name"])
    lc = G.rd(vdir, "lidar_calibration", seg)
    top_xy = np.array(lc.loc[lc["key.laser_name"] == 1, f"{G.LC}.extrinsic.transform"].iloc[0], dtype=float).reshape(4, 4)[:2, 3]
    out = []
    for (fr, cam), g in m.groupby(["key.frame_timestamp_micros", f"{SB}.most_visible_camera_name"]):
        extr, intr, w, h, rs = G.calib(cal.loc[cam])
        T = np.array(vp.loc[fr, f"{G.VP}.world_from_vehicle.transform"], dtype=float).reshape(4, 4)
        im = ci.loc[(fr, cam)]
        f_s = fr / 1e6
        c = g[[f"{LB}.box.center.{a}" for a in "xyz"]].to_numpy(dtype=float)
        cs = g[[f"{SB}.camera_synced_box.center.{a}" for a in "xyz"]].to_numpy(dtype=float)
        v = g[[f"{LB}.speed.{a}" for a in "xyz"]].to_numpy(dtype=float)
        v_veh = v @ T[:3, :3]
        dW = ((cs - c) * v_veh).sum(1) / (v_veh * v_veh).sum(1) * 1000.0
        tl = lidar_sweep_time_ms(azimuth_about(c, top_xy), A1_PERIOD_MS, A1_PHASE_MS)
        uv, ok = G.w2i(extr, intr, [w, h, rs], G.cim_of(im), transform_points(T, cs))
        trig = (im[f"{G.CI}.rolling_shutter_params.camera_trigger_time"] - f_s) * 1000.0
        shut = im[f"{G.CI}.rolling_shutter_params.shutter"] * 1000.0
        done = (im[f"{G.CI}.rolling_shutter_params.camera_readout_done_time"] - f_s) * 1000.0
        t_own = column_time(uv[:, 0], w, trig, shut, done, rs)
        t_l2r = column_time(uv[:, 0], w, trig, shut, done, LEFT_TO_RIGHT)
        t_ctr = (im[f"{G.CI}.pose_timestamp"] - f_s) * 1000.0
        tW = tl + dW
        inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] < w)
        for k in np.where(inside)[0]:
            out.append({"camera": G.CAM[int(cam)], "readout": G.RSD[rs], "xi": uv[k, 0] / w - 0.5,
                        "r_own": tW[k] - t_own[k], "r_l2r": tW[k] - t_l2r[k], "r_ctr": tW[k] - t_ctr,
                        "speed": float(np.hypot(v[k, 0], v[k, 1]))})
    return out


def main(vdir, n, out):
    segs = sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(vdir, "lidar_box", "*.parquet")))
    if n != "all":
        segs = segs[:int(n)]
    rows = []
    for s in segs:
        rows += run(vdir, s)
    d = pd.DataFrame(rows)
    tab = []
    for (cam, ro), g in d.groupby(["camera", "readout"]):
        row = {"camera": cam, "readout": ro, "n": len(g)}
        for law in ("own", "l2r", "ctr"):
            r = g[f"r_{law}"].to_numpy()
            slope, icpt = np.polyfit(g["xi"].to_numpy(), r, 1)
            res = r - (icpt + slope * g["xi"].to_numpy())
            row[f"{law}_median"] = float(np.median(r))
            row[f"{law}_slope_ms_per_width"] = float(slope)
            row[f"{law}_intercept"] = float(icpt)
            row[f"{law}_resid_mad"] = float(np.median(np.abs(res - np.median(res))))
        tab.append(row)
    t = pd.DataFrame(tab)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"segments": len(segs), "n": len(d), "table": t.to_dict(orient="records")}, f, indent=2, default=float)
    pd.set_option("display.width", 250)
    print("segments", len(segs), "observations", len(d))
    print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
