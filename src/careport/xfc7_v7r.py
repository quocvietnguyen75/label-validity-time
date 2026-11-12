"""XF-C7 V7R post hoc checks (plan V7R_PLAN_2026-11-08.md, SHA-256 a4cbff92..., with amendment A1 a6bdf591...).

All four checks are post hoc and descriptive; none changes a planned verdict.
- V7R-1 (R32, R47): per-segment deviations of WOD's camera-synchronized boxes, from the G8 per-segment medians.
- V7R-2 (R38): what the AV2 replication's two-frame check bounds for a velocity-scale error.
- V7R-3 (R55): AV2 box length: is it fixed per track, and does it grow with speed?
- V7R-4 (R44): which LiDAR copy the faces of a moving AV2 box follow.

Usage (PYTHONPATH=src; AV2 data through $AV2_SENSOR_ROOT, never printed):
  python -m careport.xfc7_v7r v7r1 <g8_segment_medians.csv> <out.json>
  python -m careport.xfc7_v7r v7r2 <T3_RESULTS.json> <out.json>
  python -m careport.xfc7_v7r v7r3 <label_side.parquet> <out.json>
  python -m careport.xfc7_v7r v7r4 <label_side.parquet> <per_box_dir> <out.json>
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

SEED = 20261108
N_BOOT = 2000
LIDAR_A_MAX_LASER = 31
MIN_PTS = 10
SEAM_MS = 50.0

# PREFREEZE_LABEL_CHECKS_2026-10-20.md section 5: |dW - frame-start prediction| median per camera x readout (ms)
COMMENT_ERROR_MS = {("FRONT", "L2R"): 50.4, ("FRONT", "R2L"): 49.4, ("FRONT_LEFT", "L2R"): 39.4, ("FRONT_LEFT", "R2L"): 38.6,
                    ("FRONT_RIGHT", "L2R"): 62.0, ("FRONT_RIGHT", "R2L"): 63.3, ("SIDE_LEFT", "L2R"): 29.8, ("SIDE_LEFT", "R2L"): 26.9,
                    ("SIDE_RIGHT", "L2R"): 73.8, ("SIDE_RIGHT", "R2L"): 77.9}

# (G_earlier, G_later) per route and class (plan, V7R-4)
SIGNATURES = {
    "earlier copy": {"approaching": (0.0, 1.0), "receding": (0.0, -1.0)},
    "later copy": {"approaching": (-1.0, 0.0), "receding": (1.0, 0.0)},
    "midpoint": {"approaching": (-0.5, 0.5), "receding": (0.5, -0.5)},
    "stretched or near face": {"approaching": (-1.0, 0.0), "receding": (0.0, -1.0)},
}


# ---------------------------------------------------------------- V7R-1

def worst_camera_counts(df: pd.DataFrame):
    """(segments, segments whose worst camera |res_ms| exceeds 5 ms, exceeds 10 ms)."""
    worst = df.assign(a=df["res_ms"].abs()).groupby("segment")["a"].max()
    return int(len(worst)), int((worst > 5).sum()), int((worst > 10).sum())


def _q(a, p):
    return float(np.percentile(np.asarray(a, float), p))


def _stats(a):
    a = np.abs(np.asarray(a, float))
    return {"n": int(len(a)), "median_abs": _q(a, 50), "p90_abs": _q(a, 90),
            "share_gt5": float(np.mean(a > 5)), "share_gt10": float(np.mean(a > 10))}


def deviation_summary(df: pd.DataFrame) -> dict:
    a = df["res_ms"].abs()
    worst = df.assign(a=a).groupby("segment")["a"].max()
    by_cell = {f"{c}|{r}": _stats(g["res_ms"]) for (c, r), g in df.groupby(["camera", "readout"])}
    return {"cells": int(len(df)), "overall": _stats(df["res_ms"]),
            "worst": {"median_abs": _q(worst, 50), "p90_abs": _q(worst, 90)},
            "by_cell": by_cell,
            "by_camera_share_gt5": {c: float((g["res_ms"].abs() > 5).mean()) for c, g in df.groupby("camera")},
            "by_readout_share_gt5": {r: float((g["res_ms"].abs() > 5).mean()) for r, g in df.groupby("readout")}}


def share_below_comparator(df: pd.DataFrame, comparator: dict) -> float:
    comp = np.array([comparator[(c, r)] for c, r in zip(df["camera"], df["readout"])], float)
    return float(np.mean(df["res_ms"].abs().to_numpy() < comp))


# ---------------------------------------------------------------- V7R-2

def two_frame_bound(lo, hi, theta_bar, theta_a, theta_b) -> dict:
    gap = abs(theta_b - theta_a)
    kappa = max(abs(lo), abs(hi)) / gap
    return {"gap_ms": gap, "kappa": kappa, "shift_ms": kappa * max(abs(theta_a - theta_bar), abs(theta_b - theta_bar))}


# ---------------------------------------------------------------- shared

def cluster_bootstrap(values, clusters, stat, n=N_BOOT, seed=SEED):
    """(estimate, 2.5th, 97.5th percentile) of stat(values) under resampling of whole clusters."""
    values = np.asarray(values, float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    idx = {c: np.flatnonzero(clusters == c) for c in uniq}
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        reps.append(stat(values[np.concatenate([idx[c] for c in pick])]))
    return float(stat(values)), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def cluster_bootstrap_frame(df: pd.DataFrame, cluster_col: str, fn, n=N_BOOT, seed=SEED):
    """As cluster_bootstrap, for a statistic of a whole DataFrame (e.g. a regression slope)."""
    groups = {c: g for c, g in df.groupby(cluster_col)}
    uniq = np.array(sorted(groups))
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        reps.append(fn(pd.concat([groups[c] for c in pick], ignore_index=True)))
    return float(fn(df)), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def ols_slope(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc * xc).sum())


# ---------------------------------------------------------------- V7R-3

def track_length_ranges(ann: pd.DataFrame) -> pd.DataFrame:
    g = ann.groupby(["log", "track_uuid"])
    return pd.DataFrame({"category": g["category"].first(), "n_frames": g.size(),
                         "length_range": g["length_m"].max() - g["length_m"].min()}).reset_index()


def moving_tracks(ann: pd.DataFrame, min_frames=5, category="REGULAR_VEHICLE", min_speed=3.0) -> pd.DataFrame:
    a = ann[ann["category"] == category].dropna(subset=["speed"])
    g = a.groupby(["log", "track_uuid"])
    t = pd.DataFrame({"n": g.size(), "min_speed": g["speed"].min(), "speed": g["speed"].median(),
                      "length_m": g["length_m"].median()}).reset_index()
    return t[(t["n"] >= min_frames) & (t["min_speed"] >= min_speed)].reset_index(drop=True)


# ---------------------------------------------------------------- V7R-4

def _box_frame(xyz, box):
    c, s = np.cos(box["yaw"]), np.sin(box["yaw"])
    d = np.asarray(xyz, float) - [box["cx"], box["cy"], box["cz"]]
    return c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]


def _window(lx, ly, dz, box, margin):
    """Amendment A1: lateral within the half-width; vertical from -h/2 + 0.3 m to h/2 + 0.2 m; along the heading
    within the box length plus margin on each side."""
    return ((np.abs(ly) <= box["w"] / 2) & (dz >= -box["h"] / 2 + 0.3) & (dz <= box["h"] / 2 + 0.2)
            & (np.abs(lx) <= box["l"] / 2 + margin))


def copy_face_gaps(xyz, laser, off_ms, box, v_xy, min_pts=MIN_PTS, seam_ms=SEAM_MS):
    """Per moving box (ego frame at the sweep time): each LiDAR copy's visible-face gap beyond the box face, or None."""
    v_xy = np.asarray(v_xy, float)
    speed = float(np.hypot(*v_xy))
    head = np.array([np.cos(box["yaw"]), np.sin(box["yaw"])])
    sdir = np.sign(v_xy @ head)
    if speed <= 0 or sdir == 0:
        return None
    lx, ly, dz = _box_frame(xyz, box)
    xm = sdir * lx
    win = _window(lx, ly, dz, box, speed * 0.07 + 0.5)
    laser, off_ms = np.asarray(laser), np.asarray(off_ms, float)
    sel = {"A": win & (laser <= LIDAR_A_MAX_LASER), "B": win & (laser > LIDAR_A_MAX_LASER)}
    if min(int(s.sum()) for s in sel.values()) < min_pts:
        return None
    if any(np.ptp(off_ms[s]) > seam_ms for s in sel.values()):
        return None
    t = {k: float(np.median(off_ms[s])) for k, s in sel.items()}
    earlier, later = ("A", "B") if t["A"] <= t["B"] else ("B", "A")
    dt = t[later] - t[earlier]
    D = speed * dt / 1000.0
    if D <= 0:
        return None
    approaching = float(v_xy @ np.array([box["cx"], box["cy"]])) < 0
    pct = 95 if approaching else 5
    face = {k: float(np.percentile(xm[s], pct)) for k, s in sel.items()}
    half = box["l"] / 2

    def gap(f):
        return f - half if approaching else -half - f

    return {"cls": "approaching" if approaching else "receding", "earlier": earlier, "dt_ms": dt, "D": D, "speed": speed,
            "g_earlier": gap(face[earlier]), "g_later": gap(face[later]),
            "separation": (face[later] - face[earlier]) / D, "n_A": int(sel["A"].sum()), "n_B": int(sel["B"].sum())}


def parked_near_face_gap(xyz, box, min_pts=MIN_PTS):
    """Parked box: the gap of the face nearest the sensor along the heading axis (all points pooled), or None."""
    lx, ly, dz = _box_frame(xyz, box)
    win = _window(lx, ly, dz, box, 0.5)
    if int(win.sum()) < min_pts:
        return None
    head = np.array([np.cos(box["yaw"]), np.sin(box["yaw"])])
    front_near = float(-np.array([box["cx"], box["cy"]]) @ head) > 0
    if front_near:
        return float(np.percentile(lx[win], 95)) - box["l"] / 2
    return -box["l"] / 2 - float(np.percentile(lx[win], 5))


def consistent_routes(intervals: dict, signatures=SIGNATURES) -> list:
    """Routes whose signature lies inside the medians' intervals in both classes.
    intervals = {cls: ((lo_E, hi_E), (lo_L, hi_L))}."""
    out = []
    for route, sig in signatures.items():
        ok = True
        for cls, (ie, il) in intervals.items():
            ge, gl = sig[cls]
            ok &= (ie[0] <= ge <= ie[1]) and (il[0] <= gl <= il[1])
        if ok:
            out.append(route)
    return out


# ---------------------------------------------------------------- AV2 readers (runner only)

def _log_dir(log):
    root = os.environ["AV2_SENSOR_ROOT"]
    for split in ("val", "train", "test"):
        p = os.path.join(root, split, log)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(log)


def log_boxes(L):
    """All annotations of a log with ego-frame geometry and the study's city-frame central-difference velocity,
    rotated into the ego frame at each sweep."""
    from careport.xfc7_t2_av2_labels import PoseTrack, VEHICLE, _read, yaw_from_quat
    ann = _read(os.path.join(L, "annotations.feather"))
    pz = _read(os.path.join(L, "city_SE3_egovehicle.feather"))
    poses = PoseTrack(pz["timestamp_ns"].to_numpy(), pz[["qw", "qx", "qy", "qz"]].to_numpy(), pz[["tx_m", "ty_m", "tz_m"]].to_numpy())
    Ts = {int(t): poses.at(int(t)) for t in ann["timestamp_ns"].unique()}
    ann["yaw_ego"] = [yaw_from_quat(*q) for q in ann[["qw", "qx", "qy", "qz"]].to_numpy()]
    cen = np.array([Ts[int(t)][:3, :3] @ c + Ts[int(t)][:3, 3] for t, c in zip(ann["timestamp_ns"], ann[["tx_m", "ty_m", "tz_m"]].to_numpy())])
    ann[["cx_c", "cy_c", "cz_c"]] = cen
    ann = ann.sort_values(["track_uuid", "timestamp_ns"]).reset_index(drop=True)
    vel = np.full((len(ann), 3), np.nan)
    for _, idx in ann.groupby("track_uuid").indices.items():
        if len(idx) >= 2:
            g = ann.iloc[idx]
            vel[idx] = np.gradient(g[["cx_c", "cy_c", "cz_c"]].to_numpy(), g["timestamp_ns"].to_numpy() / 1e9, axis=0)
    ann[["vx_c", "vy_c", "vz_c"]] = vel
    ve = np.array([Ts[int(t)][:3, :3].T @ np.nan_to_num(v) for t, v in zip(ann["timestamp_ns"], vel)])
    ann[["vx_e", "vy_e"]] = ve[:, :2]
    ann["speed"] = np.hypot(vel[:, 0], vel[:, 1])
    ann["is_vehicle"] = ann["category"].isin(VEHICLE)
    return ann


def _box_of(r):
    return {"cx": r.tx_m, "cy": r.ty_m, "cz": r.tz_m, "l": r.length_m, "w": r.width_m, "h": r.height_m, "yaw": r.yaw_ego}


def _neighbour_in_window(box, others, margin):
    if not len(others):
        return False
    c, s = np.cos(box["yaw"]), np.sin(box["yaw"])
    dx, dy = others["tx_m"].to_numpy() - box["cx"], others["ty_m"].to_numpy() - box["cy"]
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return bool(np.any((np.abs(lx) <= box["l"] / 2 + margin + others["length_m"].to_numpy() / 2)
                       & (np.abs(ly) <= box["w"] / 2 + others["width_m"].to_numpy() / 2)))


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run_v7r1(csv_path, out):
    df = pd.read_csv(csv_path)
    n, n5, n10 = worst_camera_counts(df)
    res = {"input": os.path.basename(csv_path), "input_sha256": _sha(csv_path),
           "gate": {"segments": n, "worst_gt5": n5, "worst_gt10": n10, "expected": [168, 112, 85]}}
    if (n, n5, n10) != (168, 112, 85):
        res["gate"]["passed"] = False
    else:
        res["gate"]["passed"] = True
        res["summary"] = deviation_summary(df)
        res["share_below_comment_error"] = share_below_comparator(df, COMMENT_ERROR_MS)
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    return res


def run_v7r2(t3_json, out):
    d = json.load(open(t3_json, encoding="utf-8"))
    tb = d["primary"]["t2_block"]
    rows = {}
    for c, tf in tb["two_frame"].items():
        est = tb["verdicts"]["estimates"][c]
        tbar = tb["theta_bar_by_camera"][c]["est"]
        ta, tbb = tbar - est["A"]["est"], tbar - est["B"]["est"]
        r = two_frame_bound(tf["lo"], tf["hi"], tbar, ta, tbb)
        rows[c] = {"diff_ms": tf["diff_ms"], "lo": tf["lo"], "hi": tf["hi"], "G": tf["G"], "theta_bar": tbar,
                   "theta_A": ta, "theta_B": tbb, **r}
    clear = {"ring_front_left": 27.3, "ring_side_left": 17.1, "ring_rear_left": 16.1, "ring_rear_right": 23.3}
    res = {"input": os.path.basename(t3_json), "input_sha256": _sha(t3_json), "run": "primary", "cameras": rows,
           "rejecting": {c: {"shift_ms": rows[c]["shift_ms"], "clearance_ms": v, "below": rows[c]["shift_ms"] < v}
                         for c, v in clear.items()}}
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    return res


def run_v7r3(label_side, out):
    logs = sorted(pd.read_parquet(label_side, columns=["segment"])["segment"].unique())
    frames = []
    for log in logs:
        a = log_boxes(_log_dir(log))
        a = a[a["is_vehicle"]].copy()
        a["log"] = log
        frames.append(a[["log", "track_uuid", "category", "timestamp_ns", "length_m", "speed"]])
    ann = pd.concat(frames, ignore_index=True)
    rng = track_length_ranges(ann)
    varies = rng["length_range"] > 0.01
    res = {"logs": len(logs), "tracks": int(len(rng)), "tracks_length_varies": int(varies.sum()),
           "share_varies": float(varies.mean()),
           "regular_vehicle": {"tracks": int((rng["category"] == "REGULAR_VEHICLE").sum()),
                               "share_varies": float(varies[rng["category"] == "REGULAR_VEHICLE"].mean())}}
    # step 2 eligibility
    st = ann.merge(rng[varies][["log", "track_uuid"]], on=["log", "track_uuid"])
    elig = []
    for (log, tr), g in st.groupby(["log", "track_uuid"]):
        mv, pk = g[g["speed"] >= 3.0], g[g["speed"] <= 0.5]
        if len(mv) and len(pk):
            elig.append({"log": log, "track": tr, "excess": float(mv["length_m"].median() - pk["length_m"].median()),
                         "expected": float((mv["speed"] * 0.05).median())})
    res["step2_eligible_tracks"] = len(elig)
    if len(elig) >= 30:
        e = pd.DataFrame(elig)
        e["ratio"] = e["excess"] / e["expected"]
        res["step2_ratio"] = cluster_bootstrap(e["ratio"], e["log"], np.median)
    mt = moving_tracks(ann)
    res["fallback"] = {"tracks": int(len(mt)), "logs": int(mt["log"].nunique()),
                       "slope_m_per_mps": cluster_bootstrap_frame(mt, "log", lambda f: ols_slope(f["speed"], f["length_m"])),
                       "median_speed": float(mt["speed"].median())}
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    return res


def run_v7r4(label_side, per_box_dir, out):
    from careport.xfc7_t2_av2_labels import _read
    ls = pd.read_parquet(label_side, columns=["segment", "object_id", "sweep_ns", "incl_moving", "incl_static"])
    mov = ls[ls["incl_moving"]].drop_duplicates(["segment", "object_id", "sweep_ns"])
    par = ls[ls["incl_static"]].drop_duplicates(["segment", "object_id", "sweep_ns"])
    os.makedirs(per_box_dir, exist_ok=True)
    rows, prow = [], []
    for log in sorted(set(mov["segment"]) | set(par["segment"])):
        L = _log_dir(log)
        ann = log_boxes(L)
        by_t = {int(t): g for t, g in ann.groupby("timestamp_ns")}
        want = pd.concat([mov[mov["segment"] == log].assign(kind="m"), par[par["segment"] == log].assign(kind="p")])
        for t, g in want.groupby("sweep_ns"):
            boxes = by_t.get(int(t))
            if boxes is None:
                continue
            pts = _read(os.path.join(L, "sensors", "lidar", f"{int(t)}.feather"), ["x", "y", "z", "laser_number", "offset_ns"])
            xyz = pts[["x", "y", "z"]].to_numpy(float)
            las, off = pts["laser_number"].to_numpy(), pts["offset_ns"].to_numpy(float) / 1e6
            for r in g.itertuples():
                b = boxes[boxes["track_uuid"] == r.object_id]
                if not len(b):
                    continue
                br = b.iloc[0]
                box = _box_of(br)
                others = boxes[boxes["track_uuid"] != r.object_id]
                if r.kind == "m":
                    margin = float(br.speed) * 0.07 + 0.5
                    if _neighbour_in_window(box, others, margin):
                        continue
                    res = copy_face_gaps(xyz, las, off, box, (br.vx_e, br.vy_e))
                    if res is not None:
                        rows.append({"segment": log, "object_id": r.object_id, "sweep_ns": int(t), **res})
                else:
                    if _neighbour_in_window(box, others, 0.5):
                        continue
                    gp = parked_near_face_gap(xyz, box)
                    if gp is not None:
                        prow.append({"segment": log, "object_id": r.object_id, "sweep_ns": int(t), "gap": gp})
        print(log, len(rows), len(prow), flush=True)
    m, p = pd.DataFrame(rows), pd.DataFrame(prow)
    m.to_parquet(os.path.join(per_box_dir, "v7r4_moving.parquet"), index=False)
    p.to_parquet(os.path.join(per_box_dir, "v7r4_parked.parquet"), index=False)
    b = float(p["gap"].median())
    m["G_earlier"] = (m["g_earlier"] - b) / m["D"]
    m["G_later"] = (m["g_later"] - b) / m["D"]
    res = {"moving_boxes": int(len(m)), "parked_boxes": int(len(p)), "margin_b_m": b,
           "parked_gap": cluster_bootstrap(p["gap"], p["segment"], np.median), "classes": {}}
    intervals = {}
    for cls, g in m.groupby("cls"):
        ge = cluster_bootstrap(g["G_earlier"], g["segment"], np.median)
        gl = cluster_bootstrap(g["G_later"], g["segment"], np.median)
        sep = cluster_bootstrap(g["separation"], g["segment"], np.median)
        targets = (0.0, 0.5, 1.0) if cls == "approaching" else (0.0, -0.5, -1.0)
        shares = {str(x): float(np.mean(np.abs(g["G_later"] - x) <= 0.25)) for x in targets}
        res["classes"][cls] = {"boxes": int(len(g)), "logs": int(g["segment"].nunique()), "G_earlier": ge, "G_later": gl,
                               "separation": sep, "shares_G_later": shares, "median_D_m": float(g["D"].median()),
                               "median_dt_ms": float(g["dt_ms"].median())}
        intervals[cls] = ((ge[1], ge[2]), (gl[1], gl[2]))
    sep_ok = all(not (c["separation"][2] < 0.5 or c["separation"][1] > 1.5) for c in res["classes"].values())
    res["separation_check_passed"] = sep_ok
    res["consistent_routes"] = consistent_routes(intervals) if sep_ok and len(intervals) == 2 else None
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    return res


def main(argv):
    if len(argv) < 4:
        print(__doc__)
        return 64
    cmd = argv[1]
    if cmd == "v7r1":
        r = run_v7r1(argv[2], argv[3])
    elif cmd == "v7r2":
        r = run_v7r2(argv[2], argv[3])
    elif cmd == "v7r3":
        r = run_v7r3(argv[2], argv[3])
    elif cmd == "v7r4" and len(argv) >= 5:
        r = run_v7r4(argv[2], argv[3], argv[4])
    else:
        print(__doc__)
        return 64
    print(json.dumps(r, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
