"""XF-C7 tier 2: the MAN TruckScenes label-side table (T2 protocol v0.1 section 11; descriptive by default).
Never opens an image.

theta = t_validity - t_sample. Boxes exist only on 2 Hz samples, so only keyframe images are used.
Hypotheses: H_sample (theta = 0) and one H per LiDAR: the median absolute time of the box's interior points from that
LiDAR's keyframe sweep, minus the sample time. Points are taken to the global frame with the ego translation at each
point's own time (ego pose table at about 100 Hz) and the rotation at the sweep's median time.
Rows mirror the AV2 table (xfc7_t2_av2_labels) so xfc7_t2_static runs on them; segment = scene name,
frame = image timestamp (microseconds), camera = index into TS_CAMERAS, frame_type = 'K' (keyframe).
Rolling shutter (section 3): the readout runs row by row from top to bottom and the trigger aligns at the image
centre (TruckScenes paper, Sec. 3.2 and App. A.2), so moving and static inclusion require the centre ROW third
(amended 1 Nov 2026 after the T2 audit; the first build used the centre column third). `reflag` re-derives the
inclusion flags of existing per-scene tables without re-reading any LiDAR file.

Usage: PYTHONPATH=src python -m careport.xfc7_t2_ts_labels run <truckscenes_root> <out_dir> [<scene_name> ...]
       PYTHONPATH=src python -m careport.xfc7_t2_ts_labels reflag <in_dir> <out_dir>
       PYTHONPATH=src python -m careport.xfc7_t2_ts_labels combine <out_dir> <summary.json>"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

from careport.xfc7_observations import (along_axis_angle_deg, anchor_and_gate, column_third, ego_speed_class,
                                        overlap_flags, range_bin, translate_hull, viewpoint_bin)
from careport.xfc7_geometry import hull_inside_image
from careport.xfc7_t2_av2_labels import PoseTrack, _terms_batch, inv_se3, quat_to_R, se3, yaw_from_quat
from careport.xfc7_t2_pcd import read_pcd

def row_third(v, height):
    return np.clip(np.floor(3.0 * np.asarray(v, dtype=float) / np.asarray(height, dtype=float)).astype(int), 0, 2)


def inclusion(d):
    """Unit conditions (section 7 as for AV2) plus the centre row third; moving and parked flags."""
    d = d.copy()
    d["row_third"] = row_third(d["p0_v"], d["height"])
    unit = (d["most_visible"] & (d["num_pts"] >= 20) & d["inside_all"] & ~d["overlap"] & d["static_proj_ok"]
            & d["speed"].notna() & (d["row_third"] == 1))
    d["incl_moving"] = unit & (d["speed"] >= 3.0) & (d["sdot"] >= 0.25) & (d["along_axis_deg"] >= 20.0)
    d["incl_static"] = unit & (d["speed"] <= 0.5)
    return d


TS_VEHICLE = ["vehicle.car", "vehicle.truck", "vehicle.bus.rigid", "vehicle.motorcycle"]
TS_CAMERAS = ["CAMERA_LEFT_FRONT", "CAMERA_RIGHT_FRONT", "CAMERA_LEFT_BACK", "CAMERA_RIGHT_BACK"]
TS_LIDARS = ["LIDAR_LEFT", "LIDAR_RIGHT", "LIDAR_TOP_FRONT", "LIDAR_TOP_LEFT", "LIDAR_TOP_RIGHT", "LIDAR_REAR"]


def points_to_global(xyz, t_us, T_ego_sensor, poses):
    pe = np.asarray(xyz, float) @ T_ego_sensor[:3, :3].T + T_ego_sensor[:3, 3]
    t = np.asarray(t_us, dtype=np.int64)
    tr = np.stack([np.interp(t, poses.ts, poses.t[:, k]) for k in range(3)], axis=1)
    R = poses.at(int(np.median(t)))[:3, :3]
    return pe @ R.T + tr


def box_hit_time_ms(g, t_us, box, sample_us):
    c, s = np.cos(box["yaw"]), np.sin(box["yaw"])
    d = np.asarray(g, float) - [box["cx"], box["cy"], box["cz"]]
    lx = c * d[:, 0] + s * d[:, 1]
    ly = -s * d[:, 0] + c * d[:, 1]
    inside = (np.abs(lx) <= box["l"] / 2) & (np.abs(ly) <= box["w"] / 2) & (np.abs(d[:, 2]) <= box["h"] / 2)
    n = int(inside.sum())
    return (float((np.median(np.asarray(t_us)[inside]) - sample_us) / 1000.0) if n else float("nan")), n


class Tables:
    def __init__(self, root):
        meta = glob.glob(os.path.join(root, "v1.*"))[0]
        L = lambda n: json.load(open(os.path.join(meta, n + ".json"), encoding="utf-8"))
        self.root = root
        self.sensor = {s["token"]: s["channel"] for s in L("sensor")}
        self.cal = {c["token"]: c for c in L("calibrated_sensor")}
        cat = {c["token"]: c["name"] for c in L("category")}
        self.inst_cat = {i["token"]: cat[i["category_token"]] for i in L("instance")}
        self.samples = {s["token"]: s for s in L("sample")}
        self.scenes = {s["name"]: s for s in L("scene")}
        self.ann = {a["token"]: a for a in L("sample_annotation")}
        self.ann_by_sample = {}
        for a in self.ann.values():
            self.ann_by_sample.setdefault(a["sample_token"], []).append(a)
        self.ego = {e["token"]: e for e in L("ego_pose")}
        ep = sorted(self.ego.values(), key=lambda e: e["timestamp"])
        self.poses = PoseTrack(np.array([e["timestamp"] for e in ep], dtype=np.int64),
                               np.array([e["rotation"] for e in ep]), np.array([e["translation"] for e in ep]))
        self.sd_by_sample = {}
        for sd in L("sample_data"):
            if sd["is_key_frame"]:
                ch = self.sensor[self.cal[sd["calibrated_sensor_token"]]["sensor_token"]]
                self.sd_by_sample.setdefault(sd["sample_token"], {})[ch] = sd

    def T_ego_sensor(self, sd):
        c = self.cal[sd["calibrated_sensor_token"]]
        return se3(quat_to_R(*c["rotation"]), np.array(c["translation"], float)), c["camera_intrinsic"]

    def ego_T(self, token):
        e = self.ego[token]
        return se3(quat_to_R(*e["rotation"]), np.array(e["translation"], float))


def _velocity(tb, a):
    """Central difference over the neighbouring samples' annotations of the same instance (global frame)."""
    nb = [tb.ann[k] for k in (a["prev"], a["next"]) if k]
    if not nb:
        return np.full(3, np.nan)
    pts = [a] + nb
    ts = np.array([tb.samples[p["sample_token"]]["timestamp"] for p in pts], float) / 1e6
    xs = np.array([p["translation"] for p in pts], float)
    o = np.argsort(ts)
    return (xs[o][-1] - xs[o][0]) / (ts[o][-1] - ts[o][0])


def build_scene(tb, scene_name):
    sc = tb.scenes[scene_name]
    rows = []
    tok = sc["first_sample_token"]
    while tok:
        smp = tb.samples[tok]
        t_s = smp["timestamp"]
        anns = [a for a in tb.ann_by_sample.get(tok, []) if tb.inst_cat[a["instance_token"]] in TS_VEHICLE]
        sds = tb.sd_by_sample.get(tok, {})
        if anns:
            cen = np.array([a["translation"] for a in anns], float)
            size = np.array([[a["size"][1], a["size"][0], a["size"][2]] for a in anns], float)   # (l, w, h)
            yaw = np.array([yaw_from_quat(*a["rotation"]) for a in anns])
            v = np.array([_velocity(tb, a) for a in anns])
            th = np.full((len(anns), len(TS_LIDARS)), np.nan)
            npts = np.zeros((len(anns), len(TS_LIDARS)), dtype=int)
            for k, ch in enumerate(TS_LIDARS):
                sd = sds.get(ch)
                if sd is None:
                    continue
                pc = read_pcd(os.path.join(tb.root, sd["filename"]))
                T_es, _ = tb.T_ego_sensor(sd)
                g = points_to_global(np.stack([pc["x"], pc["y"], pc["z"]], 1), pc["timestamp"].astype(np.int64), T_es, tb.poses)
                for j in range(len(anns)):
                    box = {"cx": cen[j, 0], "cy": cen[j, 1], "cz": cen[j, 2], "l": size[j, 0], "w": size[j, 1], "h": size[j, 2], "yaw": yaw[j]}
                    th[j, k], npts[j, k] = box_hit_time_ms(g, pc["timestamp"], box, t_s)
            for ci, cam in enumerate(TS_CAMERAS, 1):
                sd = sds.get(cam)
                if sd is None:
                    continue
                T_ec, K = tb.T_ego_sensor(sd)
                K = np.array(K, float)
                w, h = float(sd["width"]), float(sd["height"])
                T_ge = tb.ego_T(sd["ego_pose_token"])
                t_f = sd["timestamp"]
                dt = (t_f - t_s) / 1000.0
                vv = np.nan_to_num(v)
                thz = np.nan_to_num(th, nan=0.0)
                r = _terms_batch(K, inv_se3(T_ec), inv_se3(T_ge), cen, size, yaw, vv, dt, thz)
                keep = r["valid"][:, 0]
                if not keep.any():
                    continue
                cenp, hulls = r["cen"], r["hulls"]
                p0, hull0 = cenp[:, 0], hulls[:, 0]
                allth = np.concatenate([np.zeros((len(anns), 1)), thz], axis=1)
                idx = lambda j: np.where(j == 0, 0, j + 1)                          # into cen: 0, 10 ms, lidars
                jmin, jmax = np.argmin(allth, axis=1), np.argmax(allth, axis=1)
                pmin, pmax = cenp[np.arange(len(anns)), idx(jmin)], cenp[np.arange(len(anns)), idx(jmax)]
                hmin, hmax = hulls[np.arange(len(anns)), idx(jmin)], hulls[np.arange(len(anns)), idx(jmax)]
                anchor, gate = anchor_and_gate(pmin, pmax, hmin[:, 2] - hmin[:, 0], hmax[:, 2] - hmax[:, 0])
                a_hull = translate_hull(hull0, anchor)
                inside_all = r["valid"].all(axis=1) & np.all([hull_inside_image(hulls[:, j], w, h, margin=10.0)
                                                             for j in range(hulls.shape[1])], axis=0)
                clip = np.stack([np.clip(hull0[:, 0], 0, w), np.clip(hull0[:, 1], 0, h), np.clip(hull0[:, 2], 0, w), np.clip(hull0[:, 3], 0, h)], 1)
                vis = np.clip(clip[:, 2] - clip[:, 0], 0, None) * np.clip(clip[:, 3] - clip[:, 1], 0, None)
                cam_pos = (T_ge @ np.append(T_ec[:3, 3], 1.0))[:3]
                ego_v = (tb.poses.at(t_f + 50_000)[:3, 3] - tb.poses.at(t_f - 50_000)[:3, 3]) / 0.1
                ego_speed = float(np.linalg.norm(ego_v[:2]))
                rec = {"segment": scene_name, "frame": np.int64(t_f), "camera": ci,
                       "object_id": [a["instance_token"] for a in anns], "sweep_ns": np.int64(t_s), "dt_fs_ms": dt,
                       "frame_type": "K", "category": [tb.inst_cat[a["instance_token"]] for a in anns],
                       "width": w, "height": h, "image_id": f"{scene_name}|{t_f}|{ci}",
                       "num_pts": [a["num_lidar_pts"] for a in anns], "speed": np.hypot(v[:, 0], v[:, 1]),
                       "theta_sample": 0.0,
                       "p0_u": p0[:, 0], "p0_v": p0[:, 1], "anchor_u": anchor[:, 0], "anchor_v": anchor[:, 1], "gate": gate,
                       "hull_w0": hull0[:, 2] - hull0[:, 0], "hull_wmax": hmax[:, 2] - hmax[:, 0],
                       "ah_u0": a_hull[:, 0], "ah_v0": a_hull[:, 1], "ah_u1": a_hull[:, 2], "ah_v1": a_hull[:, 3],
                       "e_u": r["e_hat"][:, 0], "e_v": r["e_hat"][:, 1], "s": r["s"], "d_u": r["d_hat"][:, 0], "d_v": r["d_hat"][:, 1],
                       "sdot": r["sdot"], "inside_all": inside_all, "static_proj_ok": r["static_ok"], "vis_area": vis,
                       "along_axis_deg": along_axis_angle_deg(vv, np.tile(cam_pos, (len(anns), 1)), cen),
                       "range_m": np.linalg.norm(cen - cam_pos, axis=1),
                       "viewpoint_bin": viewpoint_bin(r["head"][:, :2], (cen - cam_pos)[:, :2]),
                       "column_third": column_third(p0[:, 0], w), "ego_speed": ego_speed,
                       "ego_class": int(ego_speed_class(np.array([ego_speed]))[0])}
                for k, ch in enumerate(TS_LIDARS):
                    rec[f"theta_{ch}"] = th[:, k]
                    rec[f"n_pts_{ch}"] = npts[:, k]
                rows.append(pd.DataFrame(rec)[keep])
        tok = smp["next"]
    if not rows:
        return pd.DataFrame()
    d = pd.concat(rows, ignore_index=True)
    d["range_bin"] = range_bin(d["range_m"].to_numpy())
    d["overlap"] = overlap_flags(d[["ah_u0", "ah_v0", "ah_u1", "ah_v1"]].to_numpy(), d["image_id"].to_numpy(), 0.1)
    best = d.groupby(["object_id", "sweep_ns"])["vis_area"].transform("max")
    d["most_visible"] = d["vis_area"] >= best
    d = d.drop_duplicates(["object_id", "sweep_ns", "camera"])
    return inclusion(d)


def main(argv):
    if len(argv) < 4:
        print(__doc__)
        return 64
    if argv[1] == "run":
        root, odir = argv[2], argv[3]
        os.makedirs(odir, exist_ok=True)
        tb = Tables(root)
        names = argv[4:] or sorted(tb.scenes)
        for n in names:
            p = os.path.join(odir, n + ".parquet")
            if os.path.exists(p):
                continue
            d = build_scene(tb, n)
            d.to_parquet(p + ".tmp", index=False)
            os.replace(p + ".tmp", p)
            print(n, len(d), int(d["incl_moving"].sum()) if len(d) else 0, int(d["incl_static"].sum()) if len(d) else 0, flush=True)
        return 0
    if argv[1] == "reflag":
        idir, odir = argv[2], argv[3]
        os.makedirs(odir, exist_ok=True)
        for f in sorted(glob.glob(os.path.join(idir, "*.parquet"))):
            d = inclusion(pd.read_parquet(f))
            d.to_parquet(os.path.join(odir, os.path.basename(f)), index=False)
            print(os.path.basename(f)[:-8], len(d), int(d["incl_moving"].sum()), int(d["incl_static"].sum()), flush=True)
        return 0
    if argv[1] == "combine":
        odir, summ = argv[2], argv[3]
        files = sorted(glob.glob(os.path.join(odir, "*.parquet")))
        d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).sort_values(
            ["segment", "frame", "camera", "object_id"]).reset_index(drop=True)
        final = os.path.join(os.path.dirname(odir.rstrip("/")), os.path.basename(odir.rstrip("/")) + "_label_side.parquet")
        d.to_parquet(final, index=False)
        res = {"file": "$TRUCKSCENES_DERIVED/" + os.path.basename(final),
               "sha256": hashlib.sha256(open(final, "rb").read()).hexdigest(), "rows": int(len(d)),
               "scenes": int(d["segment"].nunique()), "incl_moving": int(d["incl_moving"].sum()),
               "incl_static": int(d["incl_static"].sum()),
               "moving_tracks": int(d[d["incl_moving"]][["segment", "object_id"]].drop_duplicates().shape[0])}
        json.dump(res, open(summ, "w", encoding="utf-8"), indent=1)
        print(json.dumps(res, indent=1))
        return 0
    print(__doc__)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv))
