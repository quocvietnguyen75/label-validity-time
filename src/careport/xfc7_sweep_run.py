"""Run XF-C7 gate G3b on the 40-segment LiDAR subset (label-free; protocol v1.0 section 3).

Usage: PYTHONPATH=src python -m careport.xfc7_sweep_run <subset_list> <output_dir>
Reads WAYMO_V2_ROOT. Uses lidar_pose (top LiDAR, return 1), camera_image metadata (pose, pose_timestamp; no
image bytes) and vehicle_pose. Pass rule: the sweep reaches the vehicle's backward direction within +-3 ms of the
frame start, and the fitted period is within +-1 ms of the frame interval (medians over frames with ego >= 2 m/s).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from careport.xfc7_sweep import column_positions, column_times, fit_sweep

CI, LP, VP = "[CameraImageComponent]", "[LiDARPoseComponent]", "[VehiclePoseComponent]"


def run_segment(vdir: str, seg: str) -> pd.DataFrame:
    rd = lambda comp, cols=None: pq.read_table(os.path.join(vdir, comp, seg + ".parquet"), columns=cols).to_pandas()
    ci = rd("camera_image", ["key.frame_timestamp_micros", f"{CI}.pose.transform", f"{CI}.pose_timestamp"])
    img_t = ci[f"{CI}.pose_timestamp"].to_numpy(dtype=float)
    img_xyz = np.stack([np.asarray(p, dtype=float).reshape(4, 4)[:3, 3] for p in ci[f"{CI}.pose.transform"]])
    order = np.argsort(img_t)
    img_t, img_xyz = img_t[order], img_xyz[order]
    vp = rd("vehicle_pose", ["key.frame_timestamp_micros"])
    fts = np.sort(vp["key.frame_timestamp_micros"].to_numpy())
    interval_ms = float(np.median(np.diff(fts)) / 1000.0)
    lp = rd("lidar_pose")
    lp = lp[lp["key.laser_name"] == 1]
    rows = []
    for _, r in lp.iterrows():
        f_s = r["key.frame_timestamp_micros"] / 1e6
        shape = [int(x) for x in r[f"{LP}.range_image_return1.shape"]]
        img = np.asarray(r[f"{LP}.range_image_return1.values"], dtype=float).reshape(shape)
        cols = column_positions(img)
        win = (img_t >= f_s - 0.03) & (img_t <= f_s + 0.13)
        t = column_times(cols, img_t[win] - f_s, img_xyz[win], min_speed=2.0) * 1000.0
        a, b, mad, n = fit_sweep(np.arange(shape[1], dtype=float), t)
        if len(img_t[win]) >= 2:
            dtw = img_t[win] - img_t[win].mean()
            v = (dtw[:, None] * (img_xyz[win] - img_xyz[win].mean(axis=0))).sum(axis=0) / (dtw ** 2).sum()
            speed = float(np.linalg.norm(v))
        else:
            speed = float("nan")
        rows.append({"segment": seg, "frame": int(r["key.frame_timestamp_micros"]), "ego_speed": speed,
                     "t_backward_ms": a - 0.5 * b, "period_ms": b * shape[1], "resid_mad_ms": mad, "n_cols": n,
                     "frame_interval_ms": interval_ms})
    return pd.DataFrame(rows)


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 64
    with open(argv[1], encoding="utf-8") as f:
        segs = [ln.strip() for ln in f if ln.strip()]
    vdir = os.path.join(os.environ["WAYMO_V2_ROOT"], "validation")
    df = pd.concat([run_segment(vdir, s) for s in segs], ignore_index=True)
    ok = df[np.isfinite(df["t_backward_ms"]) & (df["n_cols"] >= 1000) & (df["ego_speed"] >= 2.0)]
    q = lambda x: [float(np.quantile(x, p)) for p in (0.05, 0.5, 0.95)]
    per_seg = ok.groupby("segment").agg(t_backward_ms=("t_backward_ms", "median"), period_ms=("period_ms", "median"),
                                         frame_interval_ms=("frame_interval_ms", "first"), n=("frame", "size"))
    med_back = float(ok["t_backward_ms"].median())
    med_period = float(ok["period_ms"].median())
    med_interval = float(df["frame_interval_ms"].median())
    res = {
        "segments": len(segs), "frames_total": int(len(df)), "frames_used": int(len(ok)),
        "segments_with_frames_used": int(per_seg.shape[0]),
        "t_backward_ms_p05_median_p95": q(ok["t_backward_ms"]),
        "period_ms_p05_median_p95": q(ok["period_ms"]),
        "frame_interval_ms_median": med_interval,
        "residual_mad_ms_median": float(ok["resid_mad_ms"].median()),
        "per_segment_t_backward_ms_p05_median_p95": q(per_seg["t_backward_ms"]),
        "per_segment_period_minus_interval_ms_p05_median_p95": q(per_seg["period_ms"] - per_seg["frame_interval_ms"]),
        "pass_phase_within_3ms": bool(abs(med_back) <= 3.0),
        "pass_period_within_1ms": bool(abs(med_period - med_interval) <= 1.0),
    }
    os.makedirs(argv[2], exist_ok=True)
    with open(os.path.join(argv[2], "g3b_sweep_phase.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
