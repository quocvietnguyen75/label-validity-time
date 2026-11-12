"""Tier 2, AV2 label-free timing check (G2/G3 analogue, 2026-11-01). Reads no annotation and no image pixel.

Inputs per log: calibration (intrinsics, egovehicle_SE3_sensor), sweep timestamps and per-point (x, y, laser_number,
offset_ns) from sensors/lidar/*.feather, and camera frame timestamps from the image FILE NAMES only.
Outputs (aggregates only):
- distortion coefficients and image sizes per ring camera;
- per LiDAR (laser_number < 32 vs >= 32): median offset_ns per 10-degree ego azimuth bin, so the sweep start
  azimuth and the spin direction can be read off;
- per ring camera: its optical-axis azimuth in the ego frame, each LiDAR's predicted crossing offset there, and the
  observed camera frame offsets relative to the sweep start (two frames per sweep expected).
Usage: python g3_av2_timing.py <av2_root> <out.json> <log_dir> [<log_dir> ...]"""
import glob
import json
import os
import sys

import numpy as np
import pyarrow.feather as feather

RING = ["ring_front_center", "ring_front_left", "ring_front_right", "ring_side_left", "ring_side_right",
        "ring_rear_left", "ring_rear_right"]


def quat_to_R(qw, qx, qy, qz):
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def camera_azimuth_deg(row):
    """Azimuth (deg, atan2(y, x) in the ego frame) of the camera's optical axis (+z of the camera frame)."""
    R = quat_to_R(row["qw"], row["qx"], row["qy"], row["qz"])
    z = R[:, 2]
    return float(np.degrees(np.arctan2(z[1], z[0])))


def main(argv):
    root, out, logs = argv[1], argv[2], argv[3:]
    res = {"logs": len(logs), "cameras": {}, "lidar_profile": {}, "frame_offsets_ms": {}}
    bins = np.arange(-180, 181, 10)
    prof = {0: [[] for _ in range(len(bins) - 1)], 1: [[] for _ in range(len(bins) - 1)]}
    offs = {c: [] for c in RING}
    cam_info = {}
    for L in logs:
        intr = feather.read_table(os.path.join(L, "calibration", "intrinsics.feather")).to_pandas().set_index("sensor_name")
        ext = feather.read_table(os.path.join(L, "calibration", "egovehicle_SE3_sensor.feather")).to_pandas().set_index("sensor_name")
        for c in RING:
            i = intr.loc[c]
            cam_info.setdefault(c, []).append({"k1": float(i["k1"]), "k2": float(i["k2"]), "k3": float(i["k3"]),
                                               "w": int(i["width_px"]), "h": int(i["height_px"]),
                                               "azimuth_deg": camera_azimuth_deg(ext.loc[c])})
        sweeps = sorted(int(os.path.basename(p)[:-8]) for p in glob.glob(os.path.join(L, "sensors", "lidar", "*.feather")))
        # LiDAR azimuth profile from every 10th sweep
        for t in sweeps[::10]:
            pts = feather.read_table(os.path.join(L, "sensors", "lidar", f"{t}.feather"),
                                     columns=["x", "y", "laser_number", "offset_ns"]).to_pandas()
            az = np.degrees(np.arctan2(pts["y"].to_numpy(), pts["x"].to_numpy()))
            k = np.digitize(az, bins) - 1
            for lid in (0, 1):
                m = (pts["laser_number"].to_numpy() >= 32) == bool(lid)
                for b in range(len(bins) - 1):
                    sel = m & (k == b)
                    if sel.any():
                        prof[lid][b].append(float(np.median(pts["offset_ns"].to_numpy()[sel])) / 1e6)
        sw = np.array(sweeps, dtype=np.int64)
        for c in RING:
            ts = np.array(sorted(int(os.path.basename(p)[:-4]) for p in glob.glob(os.path.join(L, "sensors", "cameras", c, "*.jpg"))),
                          dtype=np.int64)
            j = np.searchsorted(sw, ts, side="right") - 1          # the sweep that started last before the frame
            ok = (j >= 0) & (j < len(sw) - 1)
            offs[c].extend(((ts[ok] - sw[j[ok]]) / 1e6).tolist())
    for lid in (0, 1):
        res["lidar_profile"]["lower_laser_numbers" if lid == 0 else "upper_laser_numbers"] = {
            f"{bins[b]}..{bins[b + 1]}": (round(float(np.median(v)), 2) if v else None) for b, v in enumerate(prof[lid])}
    for c in RING:
        info = cam_info[c]
        o = np.array(offs[c])
        h = np.histogram(o, bins=np.arange(0, 101, 5))[0].tolist()
        res["cameras"][c] = {"k1_k2_k3_max_abs": max(max(abs(x["k1"]), abs(x["k2"]), abs(x["k3"])) for x in info),
                             "sizes": sorted({f"{x['w']}x{x['h']}" for x in info}),
                             "azimuth_deg_median": round(float(np.median([x["azimuth_deg"] for x in info])), 1)}
        res["frame_offsets_ms"][c] = {"n": int(len(o)), "hist_5ms_0_100": h,
                                      "modes_ms": [round(float(np.median(o[(o >= a) & (o < a + 50)])), 1)
                                                   for a in (0, 50) if ((o >= a) & (o < a + 50)).any()]}
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv)
