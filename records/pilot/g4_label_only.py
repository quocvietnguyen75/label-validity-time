"""G4 for XF-C7 protocol v1.0: the label-only file (inclusion flags, predictions, anchors, gates, e, s, d, sdot).

Roster: every (vehicle box, camera) pair present in projected_lidar_box (label-derived, hypothesis-neutral); the
most-visible-camera rows carry the estimand, the others exist for association and the overlap rule.
Never opens camera_box or masks. Per-segment Parquet files go to <out_dir>; `combine` writes one file and its hash.
Run inside WSL:  ~/envs/wod/bin/python g4_label_only.py run <validation_dir> <out_dir> [n|all] [type: 1 vehicle (default), 2 pedestrian]
                 ~/envs/wod/bin/python g4_label_only.py combine <out_dir> <summary_json>
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "/mnt/d/paper/newpaper/autonomous_driving_crossmodal_fit_selection_2026-10-13/pilot")
sys.path.insert(0, "/mnt/d/paper/src")
import g8_checks as G  # noqa: E402  (reuses the op wrappers and readers)
from careport.xfc7_geometry import box_corners, hull_2d, hull_inside_image, transform_points  # noqa: E402
from careport.xfc7_kinematics import pose_time_offset  # noqa: E402
from careport.xfc7_observations import (along_axis_angle_deg, anchor_and_gate, column_third, ego_speed_class,  # noqa: E402
                                        image_speed, motion_sign, overlap_flags, range_bin, translate_hull, viewpoint_bin)
from careport.xfc7_timing_model import A1_PERIOD_MS, A1_PHASE_MS, azimuth_about, column_time, lidar_sweep_time_ms  # noqa: E402

T_POSE_FALLBACK_MS = 49.8
TAUS = ["0", "lidar", "pose", "cap", "10"]


def segment(vdir, seg, obj_type=1):
    LB, SB, PB, CI, VP, LC = G.LB, G.SB, G.PB, G.CI, G.VP, G.LC
    lb = G.rd(vdir, "lidar_box", seg, ["key.frame_timestamp_micros", "key.laser_object_id", f"{LB}.type",
                                       f"{LB}.num_lidar_points_in_box", f"{LB}.num_top_lidar_points_in_box",
                                       f"{LB}.speed.x", f"{LB}.speed.y", f"{LB}.speed.z", f"{LB}.box.center.x", f"{LB}.box.center.y",
                                       f"{LB}.box.center.z", f"{LB}.box.size.x", f"{LB}.box.size.y", f"{LB}.box.size.z", f"{LB}.box.heading"])
    lb = lb[lb[f"{LB}.type"] == obj_type]
    pb = G.rd(vdir, "projected_lidar_box", seg, ["key.frame_timestamp_micros", "key.camera_name", "key.laser_object_id"])
    sb = G.rd(vdir, "lidar_camera_synced_box", seg, ["key.frame_timestamp_micros", "key.laser_object_id", f"{SB}.most_visible_camera_name"])
    k = ["key.frame_timestamp_micros", "key.laser_object_id"]
    r = lb.merge(pb, on=k).merge(sb, on=k, how="left")
    if not len(r):
        return pd.DataFrame()
    cal = G.rd(vdir, "camera_calibration", seg).set_index("key.camera_name")
    vp = G.rd(vdir, "vehicle_pose", seg).set_index("key.frame_timestamp_micros")
    ci = G.rd(vdir, "camera_image", seg, G.IMG_COLS)
    lc = G.rd(vdir, "lidar_calibration", seg)
    top_xy = np.array(lc.loc[lc["key.laser_name"] == 1, f"{LC}.extrinsic.transform"].iloc[0], dtype=float).reshape(4, 4)[:2, 3]
    # frame-pose time per frame (H_pose)
    tpose = {}
    for fr, g in ci.groupby("key.frame_timestamp_micros"):
        xyz = np.stack([np.asarray(p, dtype=float).reshape(4, 4)[:3, 3] for p in g[f"{CI}.pose.transform"]])
        t = g[f"{CI}.pose_timestamp"].to_numpy(dtype=float) - fr / 1e6
        T = np.asarray(vp.loc[fr, f"{VP}.world_from_vehicle.transform"], dtype=float).reshape(4, 4) if fr in vp.index else None
        val = pose_time_offset(T[:3, 3], t, xyz, min_speed=2.0) * 1000.0 if T is not None else float("nan")
        tpose[int(fr)] = (val, bool(np.isfinite(val)))
    ci = ci.set_index(["key.frame_timestamp_micros", "key.camera_name"])
    out = []
    for (fr, cam), g in r.groupby(["key.frame_timestamp_micros", "key.camera_name"]):
        if (fr, cam) not in ci.index or fr not in vp.index:
            continue
        extr, intr, w, h, rs = G.calib(cal.loc[cam])
        im = ci.loc[(fr, cam)]
        cim = G.cim_of(im)
        f_s = fr / 1e6
        T = np.asarray(vp.loc[fr, f"{VP}.world_from_vehicle.transform"], dtype=float).reshape(4, 4)
        Pimg = np.asarray(im[f"{CI}.pose.transform"], dtype=float).reshape(4, 4)
        cam_pos = Pimg[:3, :3] @ extr[:3, 3] + Pimg[:3, 3]
        ego_speed = float(np.linalg.norm([im[f"{CI}.velocity.linear_velocity.{a}"] for a in "xyz"]))
        n = len(g)
        c = g[[f"{LB}.box.center.{a}" for a in "xyz"]].to_numpy(dtype=float)
        size = g[[f"{LB}.box.size.{a}" for a in "xyz"]].to_numpy(dtype=float)
        hd = g[f"{LB}.box.heading"].to_numpy(dtype=float)
        v_w = g[[f"{LB}.speed.{a}" for a in "xyz"]].to_numpy(dtype=float)
        cw = transform_points(T, c)
        h_w = np.stack([np.cos(hd), np.sin(hd), np.zeros(n)], axis=1) @ T[:3, :3].T
        cw8 = transform_points(T, box_corners(c, size, hd))
        # static projections of the centre and of centre + 1 m along the heading (one op call)
        uvs, oks = G.w2i(extr, intr, [w, h, rs], cim, np.vstack([cw, cw + h_w]))
        uc, ue, okc, oke = uvs[:n], uvs[n:], oks[:n], oks[n:]
        trig = (im[f"{CI}.rolling_shutter_params.camera_trigger_time"] - f_s) * 1000.0
        shut = im[f"{CI}.rolling_shutter_params.shutter"] * 1000.0
        done = (im[f"{CI}.rolling_shutter_params.camera_readout_done_time"] - f_s) * 1000.0
        t_cap = column_time(uc[:, 0], w, trig, shut, done, rs)
        t_lid = lidar_sweep_time_ms(azimuth_about(c, top_xy), A1_PERIOD_MS, A1_PHASE_MS)
        tp, tp_fit = tpose.get(int(fr), (float("nan"), False))
        t_pose = np.full(n, tp if tp_fit else T_POSE_FALLBACK_MS)
        taus = {"0": np.zeros(n), "lidar": t_lid, "pose": t_pose, "cap": t_cap, "10": np.full(n, 10.0)}
        # all moving-point hulls in one op call: (5 taus x n boxes x 8 corners)
        pts = np.concatenate([(cw8 + (v_w * (im[f"{CI}.pose_timestamp"] - (f_s + taus[t] / 1000.0))[:, None])[:, None, :]).reshape(-1, 3)
                              for t in TAUS])
        vel = np.tile(np.repeat(v_w, 8, axis=0), (len(TAUS), 1))
        uvm, okm = G.w2i_moving(extr, intr, [w, h, rs], cim, pts, vel)
        hulls, valid = {}, {}
        for j, t in enumerate(TAUS):
            sl = slice(j * n * 8, (j + 1) * n * 8)
            hulls[t], valid[t] = hull_2d(uvm[sl].reshape(n, 8, 2), okm[sl].reshape(n, 8))
        cen = {t: np.stack([(hulls[t][:, 0] + hulls[t][:, 2]) / 2, (hulls[t][:, 1] + hulls[t][:, 3]) / 2], axis=1) for t in TAUS}
        e_vec = ue - uc
        e_hat = e_vec / np.linalg.norm(e_vec, axis=1, keepdims=True)
        s = motion_sign(v_w, h_w)
        d_hat = s[:, None] * e_hat
        sdot = image_speed(cen["0"], cen["10"], d_hat)
        stack = np.stack([t_lid, t_pose, t_cap], axis=1)
        imax = np.argmax(stack, axis=1)
        tau_max = stack[np.arange(n), imax]
        p_max = np.stack([cen["lidar"], cen["pose"], cen["cap"]], axis=1)[np.arange(n), imax]
        h_max = np.stack([hulls["lidar"], hulls["pose"], hulls["cap"]], axis=1)[np.arange(n), imax]
        wid0 = hulls["0"][:, 2] - hulls["0"][:, 0]
        widm = h_max[:, 2] - h_max[:, 0]
        anchor, gate = anchor_and_gate(cen["0"], p_max, wid0, widm)
        a_hull = translate_hull(hulls["0"], anchor)
        inside_all = np.ones(n, dtype=bool)
        for t in ("0", "lidar", "pose", "cap"):
            inside_all &= valid[t] & hull_inside_image(hulls[t], w, h, margin=10.0)
        mv = g[f"{SB}.most_visible_camera_name"].to_numpy()
        rec = pd.DataFrame({
            "segment": seg, "frame": int(fr), "object_id": g["key.laser_object_id"].to_numpy(), "camera": int(cam), "readout": rs,
            "width": w, "height": h, "image_id": f"{seg}|{int(fr)}|{int(cam)}",
            "most_visible": (mv == cam), "synced_present": ~pd.isna(mv),
            "speed": np.hypot(v_w[:, 0], v_w[:, 1]), "top_points": g[f"{LB}.num_top_lidar_points_in_box"].to_numpy(),
            "all_points": g[f"{LB}.num_lidar_points_in_box"].to_numpy(),
            "tau_frame": 0.0, "tau_lidar": t_lid, "tau_pose": t_pose, "tau_pose_fitted": tp_fit, "tau_cap": t_cap, "tau_max": tau_max,
            "p0_u": cen["0"][:, 0], "p0_v": cen["0"][:, 1], "pl_u": cen["lidar"][:, 0], "pl_v": cen["lidar"][:, 1],
            "pp_u": cen["pose"][:, 0], "pp_v": cen["pose"][:, 1], "pc_u": cen["cap"][:, 0], "pc_v": cen["cap"][:, 1],
            "h0_u0": hulls["0"][:, 0], "h0_v0": hulls["0"][:, 1], "h0_u1": hulls["0"][:, 2], "h0_v1": hulls["0"][:, 3],
            "anchor_u": anchor[:, 0], "anchor_v": anchor[:, 1], "gate": gate, "hull_w0": wid0, "hull_wmax": widm,
            "ah_u0": a_hull[:, 0], "ah_v0": a_hull[:, 1], "ah_u1": a_hull[:, 2], "ah_v1": a_hull[:, 3],
            "e_u": e_hat[:, 0], "e_v": e_hat[:, 1], "s": s, "d_u": d_hat[:, 0], "d_v": d_hat[:, 1], "sdot": sdot,
            "inside_all": inside_all, "static_proj_ok": okc & oke,
            "along_axis_deg": along_axis_angle_deg(v_w, np.tile(cam_pos, (n, 1)), cw),
            "range_m": np.linalg.norm(cw - cam_pos, axis=1),
            "viewpoint_bin": viewpoint_bin(h_w[:, :2], (cw - cam_pos)[:, :2]),
            "column_third": column_third(cen["0"][:, 0], w), "ego_speed": ego_speed, "ego_class": int(ego_speed_class(np.array([ego_speed]))[0]),
            "xi": anchor[:, 0] / w - 0.5,
        })
        out.append(rec)
    if not out:
        return pd.DataFrame()
    d = pd.concat(out, ignore_index=True)
    d["range_bin"] = range_bin(d["range_m"].to_numpy())
    d["overlap"] = overlap_flags(d[["ah_u0", "ah_v0", "ah_u1", "ah_v1"]].to_numpy(), d["image_id"].to_numpy(), 0.1)
    unit = d["most_visible"] & d["synced_present"] & (d["top_points"] >= 20) & d["inside_all"] & ~d["overlap"] & d["static_proj_ok"]
    d["incl_moving"] = unit & (d["speed"] >= 3.0) & (d["sdot"] >= 0.25) & (d["along_axis_deg"] >= 20.0)
    d["incl_along_axis_S5"] = unit & (d["speed"] >= 3.0) & (d["along_axis_deg"] < 20.0)
    d["incl_static"] = unit & (d["speed"] <= 0.5)
    return d


def main(argv):
    if argv[1] == "run":
        vdir, odir = argv[2], argv[3]
        n = argv[4] if len(argv) > 4 else "all"
        obj_type = int(argv[5]) if len(argv) > 5 else 1
        os.makedirs(odir, exist_ok=True)
        segs = sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(vdir, "lidar_box", "*.parquet")))
        assert len(segs) == 202
        if n != "all":
            segs = segs[:int(n)]
        for s in segs:
            p = os.path.join(odir, s + ".parquet")
            if os.path.exists(p):
                continue
            d = segment(vdir, s, obj_type)
            d.to_parquet(p + ".tmp", index=False)
            os.replace(p + ".tmp", p)
            print(s, len(d), int(d["incl_moving"].sum()) if len(d) else 0, flush=True)
    elif argv[1] == "combine":
        odir, summ = argv[2], argv[3]
        files = sorted(glob.glob(os.path.join(odir, "*.parquet")))
        d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).sort_values(
            ["segment", "frame", "camera", "object_id"]).reset_index(drop=True)
        # file named after the folder: derived/g4 -> g4_label_only.parquet (unchanged), derived/g4_ped -> g4_ped_label_only.parquet
        final = os.path.join(os.path.dirname(odir.rstrip("/")), os.path.basename(odir.rstrip("/")) + "_label_only.parquet")
        d.to_parquet(final, index=False)
        sha = hashlib.sha256(open(final, "rb").read()).hexdigest()
        cams = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}
        mvg = d[d["incl_moving"]].copy()
        mvg["n_track"] = mvg.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
        mvg["w_prelim"] = mvg["sdot"] ** 2 / mvg["n_track"]
        pred = {}
        for cam, g in mvg.groupby("camera"):
            pred[cams[cam]] = {h: float((g["w_prelim"] * g[f"tau_{h}"]).sum() / g["w_prelim"].sum()) for h in ("frame", "lidar", "pose", "cap")}
            pred[cams[cam]]["n_obs"] = int(len(g))
            pred[cams[cam]]["n_tracks"] = int(g[["segment", "object_id"]].drop_duplicates().shape[0])
        counts = {"rows": int(len(d)), "segments": int(d["segment"].nunique()), "most_visible_rows": int(d["most_visible"].sum()),
                  "incl_moving": int(d["incl_moving"].sum()), "incl_static": int(d["incl_static"].sum()),
                  "incl_along_axis_S5": int(d["incl_along_axis_S5"].sum()),
                  "moving_tracks": int(d[d["incl_moving"]][["segment", "object_id"]].drop_duplicates().shape[0]),
                  "incl_moving_by_camera_readout": {f"{cams[c]}|{'L2R' if r == 2 else 'R2L'}": int(v)
                                                    for (c, r), v in d[d["incl_moving"]].groupby(["camera", "readout"]).size().items()}}
        res = {"file": final, "sha256": sha, "counts": counts, "predicted_tau_bar_prelim_weights_sdot2_over_ntrack": pred}
        with open(summ, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv)
