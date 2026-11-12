"""XF-C7 tier 2: the AV2 label-side table (T2 protocol v0.1 sections 5, 7 and 8). Never opens an image.

Per log, one row per (sweep s, ring-camera frame f, camera, vehicle box) whose projection under H_start is valid.
Columns mirror the Waymo label-only file (pilot/g4_label_only.py) so the frozen association, zero-line and budget
code can run unchanged: segment (log id), frame (image timestamp_ns), camera (RING index), object_id (track_uuid),
anchor_u/v, gate, hull_w0, hull_wmax, p0_u/v, e_u/v, s, d_u/v, sdot, the inclusion flags and the bin columns.

Time convention (protocol section 5): theta = t_validity - t_sweep_start. p0 is the hull centre projected under
H_start (theta = 0): the object at t_f sits at the box displaced by v * (t_f - t_s - theta). theta-hat follows from
the G5 estimator with p0 under H_start, so it estimates theta directly. Predictions per row: theta_start = 0,
theta_A / theta_B = median offset of the box's interior points from LiDAR A (laser_number < 32) / B, theta_mid = 50.

Usage: PYTHONPATH=src python -m careport.xfc7_t2_av2_labels run <out_dir> <log_dir> [<log_dir> ...]
       PYTHONPATH=src python -m careport.xfc7_t2_av2_labels combine <out_dir> <summary.json>"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

from careport.xfc7_geometry import box_corners, hull_2d, hull_inside_image
from careport.xfc7_observations import (along_axis_angle_deg, anchor_and_gate, column_third, ego_speed_class,
                                        image_speed, motion_sign, overlap_flags, range_bin, translate_hull, viewpoint_bin)
from careport.xfc7_t2_detect import RING

VEHICLE = ["REGULAR_VEHICLE", "LARGE_VEHICLE", "BUS", "SCHOOL_BUS", "ARTICULATED_BUS", "BOX_TRUCK", "TRUCK",
           "TRUCK_CAB", "MOTORCYCLE"]
LIDAR_A_MAX_LASER = 31
THETA_MID_MS = 50.0
MIN_LIDAR_PTS = 5
SEAM_MS = 50.0


def quat_to_R(qw, qx, qy, qz):
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def yaw_from_quat(qw, qx, qy, qz):
    return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))


def se3(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_se3(T):
    R, t = T[:3, :3], T[:3, 3]
    return se3(R.T, -R.T @ t)


def _slerp(q0, q1, a):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(q0 @ q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)


class PoseTrack:
    """city_SE3_egovehicle at any time: linear translation and slerp rotation between samples (clamped at ends)."""

    def __init__(self, ts_ns, q, t):
        o = np.argsort(ts_ns)
        self.ts = np.asarray(ts_ns, dtype=np.int64)[o]
        self.q = np.asarray(q, float)[o]
        self.t = np.asarray(t, float)[o]

    def at(self, t_ns):
        i = int(np.clip(np.searchsorted(self.ts, t_ns), 1, len(self.ts) - 1))
        t0, t1 = self.ts[i - 1], self.ts[i]
        a = float(np.clip((t_ns - t0) / (t1 - t0), 0.0, 1.0)) if t1 > t0 else 0.0
        q = _slerp(self.q[i - 1], self.q[i], a)
        return se3(quat_to_R(*q), (1 - a) * self.t[i - 1] + a * self.t[i])


def project_points(K, T_cam_from_x, pts):
    """Pinhole projection (images are undistorted; K only) of (N, 3) points given in frame x."""
    p = np.asarray(pts, float) @ T_cam_from_x[:3, :3].T + T_cam_from_x[:3, 3]
    z = p[:, 2]
    ok = z > 0.1
    zz = np.where(ok, z, 1.0)
    uv = np.stack([K[0, 0] * p[:, 0] / zz + K[0, 2], K[1, 1] * p[:, 1] / zz + K[1, 2]], axis=1)
    return uv, ok


def _inside_masks(pts, boxes):
    xyz = pts[["x", "y", "z"]].to_numpy(float)
    las = pts["laser_number"].to_numpy()
    for b in boxes:
        c, s = np.cos(b["yaw"]), np.sin(b["yaw"])
        d = xyz - [b["cx"], b["cy"], b["cz"]]
        lx = c * d[:, 0] + s * d[:, 1]
        ly = -s * d[:, 0] + c * d[:, 1]
        inside = (np.abs(lx) <= b["l"] / 2) & (np.abs(ly) <= b["w"] / 2) & (np.abs(d[:, 2]) <= b["h"] / 2)
        yield inside & (las <= LIDAR_A_MAX_LASER), inside & (las > LIDAR_A_MAX_LASER)


def interior_offsets(pts, boxes):
    """Per box (ego frame at the sweep time): ((median offset ms, n) for LiDAR A, (median, n) for LiDAR B)."""
    off = pts["offset_ns"].to_numpy(float) / 1e6
    return [tuple((float(np.median(off[sel])) if sel.any() else float("nan"), int(sel.sum())) for sel in pair)
            for pair in _inside_masks(pts, boxes)]


def interior_spreads(pts, boxes):
    """Per box: (max - min offset in ms) of its interior points for LiDAR A and B; > 50 ms means the box straddles
    that LiDAR's sweep seam, where the median offset is meaningless (T2 protocol section 5)."""
    off = pts["offset_ns"].to_numpy(float) / 1e6
    return [tuple(float(off[sel].max() - off[sel].min()) if sel.any() else float("nan") for sel in pair)
            for pair in _inside_masks(pts, boxes)]


def frames_in_sweeps(sweep_ns, cam_ns):
    """[(sweep, frame)]: each camera frame paired with the last sweep that started at or before it (within 100 ms)."""
    sw = np.asarray(sweep_ns, dtype=np.int64)
    out = []
    for f in np.asarray(cam_ns, dtype=np.int64):
        j = np.searchsorted(sw, f, side="right") - 1
        if j >= 0 and f - sw[j] < 100_000_000:
            out.append((int(sw[j]), int(f)))
    return out


def frame_type(dt_ms, cross_a, cross_b):
    """'A' or 'B': the LiDAR whose crossing of the camera axis (offset after the sweep start) is nearer, mod 100 ms."""
    da = abs((dt_ms - cross_a + 50.0) % 100.0 - 50.0)
    db = abs((dt_ms - cross_b + 50.0) % 100.0 - 50.0)
    return "A" if da <= db else "B"


def _terms_batch(K, T_cam_from_ego, T_ego_from_city, centres, sizes, yaws, v, dt_fs_ms, thetas):
    """Hulls and image terms for N boxes (city frame) seen in one frame. thetas: (N, k) validity offsets (ms)."""
    n = len(centres)
    T = T_cam_from_ego @ T_ego_from_city
    corners = box_corners(centres, sizes, yaws)                                   # (N, 8, 3)
    thetas = np.asarray(thetas, float).reshape(n, -1)
    allth = np.concatenate([np.zeros((n, 1)), np.full((n, 1), 10.0), thetas], axis=1)
    k = allth.shape[1]
    disp = v[:, None, :] * ((dt_fs_ms - allth) / 1000.0)[:, :, None]               # (N, k, 3)
    pts = corners[:, None, :, :] + disp[:, :, None, :]                             # (N, k, 8, 3)
    uv, ok = project_points(K, T, pts.reshape(-1, 3))
    hulls, valid = hull_2d(uv.reshape(n * k, 8, 2), ok.reshape(n * k, 8))
    hulls, valid = hulls.reshape(n, k, 4), valid.reshape(n, k)
    cen = np.stack([(hulls[..., 0] + hulls[..., 2]) / 2, (hulls[..., 1] + hulls[..., 3]) / 2], axis=-1)
    head = np.stack([np.cos(yaws), np.sin(yaws), np.zeros(n)], axis=1)
    uc, okc = project_points(K, T, centres)
    ue, oke = project_points(K, T, centres + head)
    e = ue - uc
    e_hat = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
    s = motion_sign(v, head)
    d_hat = s[:, None] * e_hat
    sdot = image_speed(cen[:, 0], cen[:, 1], d_hat)
    return {"cen": cen, "hulls": hulls, "valid": valid, "e_hat": e_hat, "s": s, "d_hat": d_hat, "sdot": sdot,
            "static_ok": okc & oke, "head": head}


def box_image_terms(K, T_cam_from_ego, T_ego_from_city, centre, size, yaw, v, dt_fs_ms, thetas):
    """One box: hull centres p for each theta, and d_hat and sdot (from theta = 0 and theta = 10 ms)."""
    r = _terms_batch(K, T_cam_from_ego, T_ego_from_city, np.atleast_2d(centre), np.atleast_2d(size),
                     np.atleast_1d(float(yaw)), np.atleast_2d(v), dt_fs_ms, np.atleast_2d(thetas))
    return {"p": r["cen"][0, 2:], "d_hat": r["d_hat"][0], "sdot": float(r["sdot"][0])}


def _read(path, cols=None):
    import pyarrow.feather as feather
    return feather.read_table(path, columns=cols).to_pandas()


def crossing_offsets(L, sweeps, cams_az):
    """Per camera: median offset (ms) of LiDAR A and B points within 3 degrees of the camera axis azimuth."""
    acc = {c: ([], []) for c in cams_az}
    for t in sweeps[::15]:
        p = _read(os.path.join(L, "sensors", "lidar", f"{t}.feather"), ["x", "y", "laser_number", "offset_ns"])
        az = np.degrees(np.arctan2(p["y"].to_numpy(), p["x"].to_numpy()))
        las, off = p["laser_number"].to_numpy(), p["offset_ns"].to_numpy() / 1e6
        for c, a0 in cams_az.items():
            near = np.abs((az - a0 + 180) % 360 - 180) < 3.0
            for k, sel in enumerate((near & (las <= LIDAR_A_MAX_LASER), near & (las > LIDAR_A_MAX_LASER))):
                if sel.any():
                    acc[c][k].append(float(np.median(off[sel])))
    return {c: (float(np.median(a)), float(np.median(b))) for c, (a, b) in acc.items()}


def build_log(L):
    log = os.path.basename(os.path.normpath(L))
    ann = _read(os.path.join(L, "annotations.feather"))
    ann = ann[ann["category"].isin(VEHICLE)].copy()
    pz = _read(os.path.join(L, "city_SE3_egovehicle.feather"))
    poses = PoseTrack(pz["timestamp_ns"].to_numpy(), pz[["qw", "qx", "qy", "qz"]].to_numpy(), pz[["tx_m", "ty_m", "tz_m"]].to_numpy())
    intr = _read(os.path.join(L, "calibration", "intrinsics.feather")).set_index("sensor_name")
    ext = _read(os.path.join(L, "calibration", "egovehicle_SE3_sensor.feather")).set_index("sensor_name")
    sweeps = sorted(int(os.path.basename(p)[:-8]) for p in glob.glob(os.path.join(L, "sensors", "lidar", "*.feather")))
    # city-frame centres, yaws and velocities
    Ts = {t: poses.at(t) for t in sweeps}
    ann["yaw_ego"] = [yaw_from_quat(*q) for q in ann[["qw", "qx", "qy", "qz"]].to_numpy()]
    cc, yc = [], []
    for t, c, y in zip(ann["timestamp_ns"].to_numpy(), ann[["tx_m", "ty_m", "tz_m"]].to_numpy(), ann["yaw_ego"].to_numpy()):
        T = Ts[int(t)]
        cc.append(T[:3, :3] @ c + T[:3, 3])
        yc.append(y + np.arctan2(T[1, 0], T[0, 0]))
    ann[["cx_c", "cy_c", "cz_c"]] = np.array(cc)
    ann["yaw_c"] = np.array(yc)
    ann = ann.sort_values(["track_uuid", "timestamp_ns"])
    vel = np.full((len(ann), 3), np.nan)
    for _, idx in ann.groupby("track_uuid").indices.items():
        if len(idx) < 2:
            continue
        g = ann.iloc[idx]
        t = g["timestamp_ns"].to_numpy() / 1e9
        x = g[["cx_c", "cy_c", "cz_c"]].to_numpy()
        vel[idx] = np.gradient(x, t, axis=0)
    ann[["vx", "vy", "vz"]] = vel
    # interior LiDAR offsets per box
    th = np.full((len(ann), 2), np.nan)
    spr = np.full((len(ann), 2), np.nan)
    npts = np.zeros((len(ann), 2), dtype=int)
    pos = {k: i for i, k in enumerate(ann.index)}
    for t, g in ann.groupby("timestamp_ns"):
        p = _read(os.path.join(L, "sensors", "lidar", f"{int(t)}.feather"), ["x", "y", "z", "laser_number", "offset_ns"])
        boxes = [{"cx": r.tx_m, "cy": r.ty_m, "cz": r.tz_m, "l": r.length_m, "w": r.width_m, "h": r.height_m, "yaw": r.yaw_ego}
                 for r in g.itertuples()]
        for k, ((ta, na), (tb, nb)), sp in zip(g.index, interior_offsets(p, boxes), interior_spreads(p, boxes)):
            th[pos[k]] = (ta, tb)
            npts[pos[k]] = (na, nb)
            spr[pos[k]] = sp
    ann["theta_A"], ann["theta_B"] = th[:, 0], th[:, 1]
    ann["n_pts_A"], ann["n_pts_B"] = npts[:, 0], npts[:, 1]
    ann["spread_A"], ann["spread_B"] = spr[:, 0], spr[:, 1]
    cams_az = {}
    for c in RING:
        R = quat_to_R(*ext.loc[c, ["qw", "qx", "qy", "qz"]].to_numpy(float))
        cams_az[c] = float(np.degrees(np.arctan2(R[1, 2], R[0, 2])))
    cross = crossing_offsets(L, sweeps, cams_az)
    # fallback hit offsets for boxes with few points of one LiDAR: that LiDAR's offset at the box centre azimuth
    prof = lidar_profile(L, sweeps)
    az_box = np.degrees(np.arctan2(ann["ty_m"].to_numpy(), ann["tx_m"].to_numpy()))
    for k, col, ncol in ((0, "theta_A", "n_pts_A"), (1, "theta_B", "n_pts_B")):
        few = ann[ncol].to_numpy() < MIN_LIDAR_PTS
        ann.loc[few, col] = np.interp(az_box[few], prof[0], prof[k + 1], period=360.0)
    ann["theta_fallback_A"] = ann["n_pts_A"] < MIN_LIDAR_PTS
    ann["theta_fallback_B"] = ann["n_pts_B"] < MIN_LIDAR_PTS
    by_sweep = {int(t): g for t, g in ann.groupby("timestamp_ns")}
    rows = []
    for ci, c in enumerate(RING, 1):
        i = intr.loc[c]
        K = np.array([[i["fx_px"], 0, i["cx_px"]], [0, i["fy_px"], i["cy_px"]], [0, 0, 1.0]])
        w, h = float(i["width_px"]), float(i["height_px"])
        T_ego_cam = se3(quat_to_R(*ext.loc[c, ["qw", "qx", "qy", "qz"]].to_numpy(float)), ext.loc[c, ["tx_m", "ty_m", "tz_m"]].to_numpy(float))
        T_cam_ego = inv_se3(T_ego_cam)
        cam_ts = sorted(int(os.path.basename(p)[:-4]) for p in glob.glob(os.path.join(L, "sensors", "cameras", c, "*.jpg")))
        for t_s, t_f in frames_in_sweeps(sweeps, cam_ts):
            g = by_sweep.get(t_s)
            if g is None or not len(g):
                continue
            T_city_ego_f = poses.at(t_f)
            T_ego_city_f = inv_se3(T_city_ego_f)
            dt = (t_f - t_s) / 1e6
            cen_c = g[["cx_c", "cy_c", "cz_c"]].to_numpy()
            v = g[["vx", "vy", "vz"]].to_numpy()
            vv = np.nan_to_num(v)
            thetas = np.stack([g["theta_A"].to_numpy(), g["theta_B"].to_numpy(), np.full(len(g), THETA_MID_MS)], axis=1)
            r = _terms_batch(K, T_cam_ego, T_ego_city_f, cen_c, g[["length_m", "width_m", "height_m"]].to_numpy(),
                             g["yaw_c"].to_numpy(), vv, dt, np.nan_to_num(thetas, nan=THETA_MID_MS))
            keep = r["valid"][:, 0]
            if not keep.any():
                continue
            cen, hulls = r["cen"], r["hulls"]
            p0, hull0 = cen[:, 0], hulls[:, 0]
            # extremes of the hypotheses (theta = 0 and the largest predicted theta) for the anchor and gate
            th_all = np.concatenate([np.zeros((len(g), 1)), np.nan_to_num(thetas, nan=THETA_MID_MS)], axis=1)
            imax = np.argmax(th_all, axis=1)
            col = np.where(imax == 0, 0, imax + 1)                     # index into cen/hulls (0, 10, A, B, mid)
            pmax = cen[np.arange(len(g)), col]
            hmax = hulls[np.arange(len(g)), col]
            wid0, widm = hull0[:, 2] - hull0[:, 0], hmax[:, 2] - hmax[:, 0]
            anchor, gate = anchor_and_gate(p0, pmax, wid0, widm)
            a_hull = translate_hull(hull0, anchor)
            inside_all = np.ones(len(g), dtype=bool)
            for j in (0, 2, 3, 4):
                inside_all &= r["valid"][:, j] & hull_inside_image(hulls[:, j], w, h, margin=10.0)
            clip = np.stack([np.clip(hull0[:, 0], 0, w), np.clip(hull0[:, 1], 0, h), np.clip(hull0[:, 2], 0, w), np.clip(hull0[:, 3], 0, h)], axis=1)
            vis_area = np.clip(clip[:, 2] - clip[:, 0], 0, None) * np.clip(clip[:, 3] - clip[:, 1], 0, None)
            cam_pos = (T_city_ego_f @ np.append(T_ego_cam[:3, 3], 1.0))[:3]
            ego_v = (poses.at(t_f + 50_000_000)[:3, 3] - poses.at(t_f - 50_000_000)[:3, 3]) / 0.1
            ego_speed = float(np.linalg.norm(ego_v[:2]))
            ftype = frame_type(dt, *cross[c])
            rec = pd.DataFrame({
                "segment": log, "frame": np.int64(t_f), "camera": ci, "object_id": g["track_uuid"].to_numpy(),
                "sweep_ns": np.int64(t_s), "dt_fs_ms": dt, "frame_type": ftype, "category": g["category"].to_numpy(),
                "width": w, "height": h, "image_id": f"{log}|{t_f}|{ci}",
                "num_pts": g["num_interior_pts"].to_numpy(), "n_pts_A": g["n_pts_A"].to_numpy(), "n_pts_B": g["n_pts_B"].to_numpy(),
                "speed": np.hypot(v[:, 0], v[:, 1]),
                "theta_start": 0.0, "theta_A": g["theta_A"].to_numpy(), "theta_B": g["theta_B"].to_numpy(),
                "theta_mid": THETA_MID_MS, "theta_fallback_A": g["theta_fallback_A"].to_numpy(),
                "theta_fallback_B": g["theta_fallback_B"].to_numpy(),
                "spread_A": g["spread_A"].to_numpy(), "spread_B": g["spread_B"].to_numpy(),
                "p0_u": p0[:, 0], "p0_v": p0[:, 1],
                "h0_u0": hull0[:, 0], "h0_v0": hull0[:, 1], "h0_u1": hull0[:, 2], "h0_v1": hull0[:, 3],
                "anchor_u": anchor[:, 0], "anchor_v": anchor[:, 1], "gate": gate, "hull_w0": wid0, "hull_wmax": widm,
                "ah_u0": a_hull[:, 0], "ah_v0": a_hull[:, 1], "ah_u1": a_hull[:, 2], "ah_v1": a_hull[:, 3],
                "e_u": r["e_hat"][:, 0], "e_v": r["e_hat"][:, 1], "s": r["s"], "d_u": r["d_hat"][:, 0], "d_v": r["d_hat"][:, 1],
                "sdot": r["sdot"], "inside_all": inside_all, "static_proj_ok": r["static_ok"], "vis_area": vis_area,
                "along_axis_deg": along_axis_angle_deg(vv, np.tile(cam_pos, (len(g), 1)), cen_c),
                "range_m": np.linalg.norm(cen_c - cam_pos, axis=1),
                "viewpoint_bin": viewpoint_bin(r["head"][:, :2], (cen_c - cam_pos)[:, :2]),
                "column_third": column_third(p0[:, 0], w), "ego_speed": ego_speed,
                "ego_class": int(ego_speed_class(np.array([ego_speed]))[0]),
            })
            rows.append(rec[keep])
    if not rows:
        return pd.DataFrame(), cross
    d = pd.concat(rows, ignore_index=True)
    d["range_bin"] = range_bin(d["range_m"].to_numpy())
    d["overlap"] = overlap_flags(d[["ah_u0", "ah_v0", "ah_u1", "ah_v1"]].to_numpy(), d["image_id"].to_numpy(), 0.1)
    best = d.groupby(["object_id", "sweep_ns", "frame_type"])["vis_area"].transform("max")
    d["most_visible"] = d["vis_area"] >= best
    d = d.drop_duplicates(["object_id", "sweep_ns", "frame_type", "camera"])
    unit = d["most_visible"] & (d["num_pts"] >= 20) & d["inside_all"] & ~d["overlap"] & d["static_proj_ok"] & d["speed"].notna()
    d["seam"] = (d["spread_A"] > SEAM_MS) | (d["spread_B"] > SEAM_MS)
    d["incl_moving"] = unit & (d["speed"] >= 3.0) & (d["sdot"] >= 0.25) & (d["along_axis_deg"] >= 20.0) & ~d["seam"]
    d["incl_static"] = unit & (d["speed"] <= 0.5)
    return d, cross


def lidar_profile(L, sweeps):
    """(azimuth bin centres, median offset of A, median offset of B) over 2-degree bins, from every 15th sweep."""
    bins = np.arange(-180, 181, 2)
    acc = [[[] for _ in range(len(bins) - 1)] for _ in range(2)]
    for t in sweeps[::15]:
        p = _read(os.path.join(L, "sensors", "lidar", f"{t}.feather"), ["x", "y", "laser_number", "offset_ns"])
        az = np.degrees(np.arctan2(p["y"].to_numpy(), p["x"].to_numpy()))
        k = np.clip(np.digitize(az, bins) - 1, 0, len(bins) - 2)
        las, off = p["laser_number"].to_numpy(), p["offset_ns"].to_numpy() / 1e6
        for li, sel in enumerate((las <= LIDAR_A_MAX_LASER, las > LIDAR_A_MAX_LASER)):
            for b in np.unique(k[sel]):
                m = sel & (k == b)
                acc[li][b].append(float(np.median(off[m])))
    cen = (bins[:-1] + bins[1:]) / 2
    a = np.array([np.median(x) if x else np.nan for x in acc[0]])
    b = np.array([np.median(x) if x else np.nan for x in acc[1]])
    ok = np.isfinite(a) & np.isfinite(b)
    return cen[ok], a[ok], b[ok]


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 64
    if argv[1] == "run":
        odir, logs = argv[2], argv[3:]
        os.makedirs(odir, exist_ok=True)
        for L in logs:
            log = os.path.basename(os.path.normpath(L))
            p = os.path.join(odir, log + ".parquet")
            if os.path.exists(p):
                continue
            d, cross = build_log(L)
            d.to_parquet(p + ".tmp", index=False)
            os.replace(p + ".tmp", p)
            json.dump({c: {"cross_A_ms": a, "cross_B_ms": b} for c, (a, b) in cross.items()},
                      open(os.path.join(odir, log + "_crossing.json"), "w", encoding="utf-8"), indent=1)
            print(log, len(d), int(d["incl_moving"].sum()) if len(d) else 0, int(d["incl_static"].sum()) if len(d) else 0, flush=True)
        return 0
    if argv[1] == "combine":
        odir, summ = argv[2], argv[3]
        files = sorted(glob.glob(os.path.join(odir, "*.parquet")))
        d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).sort_values(
            ["segment", "frame", "camera", "object_id"]).reset_index(drop=True)
        final = os.path.join(os.path.dirname(odir.rstrip("/")), os.path.basename(odir.rstrip("/")) + "_label_side.parquet")
        d.to_parquet(final, index=False)
        res = {"file": "$AV2_SENSOR_ROOT/derived/" + os.path.basename(final), "sha256": hashlib.sha256(open(final, "rb").read()).hexdigest(), "rows": int(len(d)),
               "logs": int(d["segment"].nunique()), "incl_moving": int(d["incl_moving"].sum()),
               "incl_static": int(d["incl_static"].sum()),
               "moving_tracks": int(d[d["incl_moving"]][["segment", "object_id"]].drop_duplicates().shape[0])}
        json.dump(res, open(summ, "w", encoding="utf-8"), indent=1)
        print(json.dumps(res, indent=1))
        return 0
    print(__doc__)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv))
