"""XF-C7 post hoc set V6R, part 1: the image av2-api pairs with each LiDAR sweep by default (v6 review R19, R2).

The plan is `t3_replication_2026-11-04/posthoc_2026-11-08/V6R_PLAN_2026-11-08.md`, pinned before anything was computed
(SHA-256 PLAN_SHA). Post hoc and descriptive: no frozen rule, tolerance, test or interval changes.

Rule replicated from av2-api commit b7321d1f71f6ce0ecdd151f4f2b648338c191edd,
`src/av2/datasets/sensor/sensor_dataloader.py`, `_build_synchronization_cache`: each LiDAR sweep timestamp is joined to
each ring camera's image timestamps with `pandas.merge_asof(direction="nearest")`, tolerance
`CAM_SHUTTER_INTERVAL_MS / 2` = 25 ms for a LiDAR source.

Only timestamps from file names are read (`sensors/lidar/<ns>.feather`, `sensors/cameras/ring_*/<ns>.jpg`) from the
logs under $AV2_SENSOR_ROOT/{train,val}. The replication's camera means (sweep start, CR1, primary run) come from the
pinned T3 result file.

Usage (repository root):
  AV2_SENSOR_ROOT=... PYTHONPATH=src python -m careport.xfc7_t3_posthoc_v6r <plan.md> <T3_RESULTS.json> <out.json>
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

PLAN_SHA = "df688a71bdc4cc459c2da262e83afe6c0954b8d4abe5548f392335264319933b"
T3_SHA = "e28f472dca12f9438bc9d7fe7bc4cd95d3cbe0ff488a1dcfd388b8ae94d6f3dc"
AV2_API_COMMIT = "b7321d1f71f6ce0ecdd151f4f2b648338c191edd"
MS = 1_000_000
TOLERANCE_NS = 25 * MS
SWEEP_NS = 100 * MS
RING = ("ring_front_center", "ring_front_left", "ring_front_right", "ring_side_left", "ring_side_right",
        "ring_rear_left", "ring_rear_right")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_timestamps(names, suffix):
    """Integer nanosecond timestamps from file names '<ns><suffix>', sorted; other names are ignored."""
    out = [int(n[: -len(suffix)]) for n in names if n.endswith(suffix) and n[: -len(suffix)].isdigit()]
    return np.array(sorted(out), dtype=np.int64)


def nearest_within(sweeps_ns, images_ns, tol_ns):
    """For each sweep, the nearest image timestamp if |difference| <= tol_ns (merge_asof 'nearest' semantics).

    Returns (matched_ns, ok); matched_ns is meaningful only where ok is True.
    """
    sweeps_ns = np.asarray(sweeps_ns, dtype=np.int64)
    images_ns = np.asarray(images_ns, dtype=np.int64)
    matched = np.zeros(len(sweeps_ns), dtype=np.int64)
    ok = np.zeros(len(sweeps_ns), dtype=bool)
    if len(images_ns) == 0 or len(sweeps_ns) == 0:
        return matched, ok
    idx = np.searchsorted(images_ns, sweeps_ns, side="left")
    lo = np.clip(idx - 1, 0, len(images_ns) - 1)
    hi = np.clip(idx, 0, len(images_ns) - 1)
    d_lo = np.abs(sweeps_ns - images_ns[lo])
    d_hi = np.abs(images_ns[hi] - sweeps_ns)
    # pandas' 'nearest' prefers the backward (earlier or equal) match on a tie
    take_lo = d_lo <= d_hi
    best = np.where(take_lo, images_ns[lo], images_ns[hi])
    dist = np.where(take_lo, d_lo, d_hi)
    ok = dist <= tol_ns
    matched = np.where(ok, best, 0)
    return matched, ok


def summarize(d_ms):
    d = np.asarray(d_ms, dtype=float)
    if len(d) == 0:
        return {"n": 0}
    return {"n": int(len(d)), "median": float(np.median(d)), "p025": float(np.percentile(d, 2.5)),
            "p975": float(np.percentile(d, 97.5)), "min": float(d.min()), "max": float(d.max()),
            "frac_negative": float(np.mean(d < 0))}


def compare(theta_ms, d_ms):
    """Leaving a box unchanged in a frame at offset d errs by |theta - d|; re-timing from the sweep start, by |theta|."""
    leave = abs(theta_ms - d_ms)
    retime = abs(theta_ms)
    return {"leave_unchanged_ms": round(leave, 6), "retime_from_sweep_start_ms": round(retime, 6),
            "leave_errs_more": bool(leave > retime)}


def own_frames(sweeps_ns, images_ns):
    """Offsets (ms) of the first and second images in [t, t + 100 ms) after each sweep timestamp t."""
    first, second = [], []
    for t in sweeps_ns:
        a = np.searchsorted(images_ns, t, side="left")
        b = np.searchsorted(images_ns, t + SWEEP_NS, side="left")
        offs = (images_ns[a:b] - t) / MS
        if len(offs) >= 1:
            first.append(float(offs[0]))
        if len(offs) >= 2:
            second.append(float(offs[1]))
    return first, second


def list_logs(root):
    logs = []
    for split in ("train", "val"):
        base = os.path.join(root, split)
        if not os.path.isdir(base):
            continue
        for log_id in sorted(os.listdir(base)):
            if os.path.isdir(os.path.join(base, log_id, "sensors", "lidar")):
                logs.append((split, log_id))
    return logs


def main(argv):
    if len(argv) != 4:
        raise SystemExit(__doc__)
    plan, t3_path, out = argv[1:]
    for path, pinned in ((plan, PLAN_SHA), (t3_path, T3_SHA)):
        got = sha256(path)
        if got != pinned:
            raise SystemExit(f"{path}: SHA-256 {got} is not the pinned {pinned}; stop")
    root = os.environ["AV2_SENSOR_ROOT"]
    logs = list_logs(root)
    d_all = {c: [] for c in RING}
    per_log_median = {c: [] for c in RING}
    sweeps_total = {c: 0 for c in RING}
    matched_total = {c: 0 for c in RING}
    first_all = {c: [] for c in RING}
    second_all = {c: [] for c in RING}
    for split, log_id in logs:
        log_dir = os.path.join(root, split, log_id)
        sweeps = parse_timestamps(os.listdir(os.path.join(log_dir, "sensors", "lidar")), ".feather")
        for c in RING:
            cam_dir = os.path.join(log_dir, "sensors", "cameras", c)
            images = parse_timestamps(os.listdir(cam_dir), ".jpg") if os.path.isdir(cam_dir) else np.array([], np.int64)
            matched, ok = nearest_within(sweeps, images, TOLERANCE_NS)
            d = (matched[ok] - sweeps[ok]) / MS
            sweeps_total[c] += int(len(sweeps))
            matched_total[c] += int(ok.sum())
            d_all[c].extend(d.tolist())
            if len(d):
                per_log_median[c].append(float(np.median(d)))
            f, s = own_frames(sweeps, images)
            first_all[c].extend(f)
            second_all[c].extend(s)
    with open(t3_path, encoding="utf-8") as fh:
        b = json.load(fh)["primary"]
    tm = b["three_method"]
    rejecting = sorted(tm["three"]["runs"]["primary"]["start"]["rejected_in"])
    cams = {}
    for c in RING:
        s = summarize(d_all[c])
        entry = {"sweeps": sweeps_total[c], "matched": matched_total[c],
                 "matched_fraction": matched_total[c] / sweeps_total[c] if sweeps_total[c] else None,
                 "default_pair_offset_ms": s,
                 "per_log_median_range_ms": [min(per_log_median[c]), max(per_log_median[c])] if per_log_median[c] else None,
                 "own_frames_median_ms": [float(np.median(first_all[c])) if first_all[c] else None,
                                          float(np.median(second_all[c])) if second_all[c] else None],
                 "rejects_sweep_start_primary": c in rejecting}
        if c in tm["intervals"]["primary"] and s.get("n"):
            ci = tm["intervals"]["primary"][c]["start"]["cr1"]
            entry["camera_mean_ms_cr1"] = ci
            entry["comparison_at"] = {k: compare(ci[k], s["median"]) for k in ("est", "lo", "hi")}
        cams[c] = entry
    res = {"part": "V6R-1", "plan": os.path.basename(plan), "plan_sha256": PLAN_SHA, "t3_results_sha256": T3_SHA,
           "av2_api_commit": AV2_API_COMMIT,
           "rule": "nearest image timestamp to each sweep timestamp within +-25 ms (merge_asof 'nearest')",
           "data_root": "$AV2_SENSOR_ROOT", "logs": {"train": sum(1 for s, _ in logs if s == "train"),
                                                     "val": sum(1 for s, _ in logs if s == "val")},
           "log_ids": [f"{s}/{l}" for s, l in logs], "rejecting_cameras_primary": rejecting, "cameras": cams,
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
           "computed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(out, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    for c in RING:
        e = cams[c]
        s = e["default_pair_offset_ms"]
        print(f"{c:18s} matched {e['matched']}/{e['sweeps']}  d median {s.get('median', float('nan')):7.2f} "
              f"[{s.get('p025', float('nan')):6.2f}, {s.get('p975', float('nan')):6.2f}]  own {e['own_frames_median_ms']}"
              f"  reject={e['rejects_sweep_start_primary']}  cmp={e.get('comparison_at', {}).get('est')}")
    return 0


# ---------------------------------------------------------------------------------------------------------------
# V6R-2 to V6R-7 (plan Addendum 2): arithmetic on stored values, plus the speed summary of V6R-4.
# Usage: PYTHONPATH=src python -m careport.xfc7_t3_posthoc_v6r arith <cycle_dir> <out.json>
# ---------------------------------------------------------------------------------------------------------------

WOD_CAMS = ("FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT")
WOD_CODE = {"FRONT": 1, "FRONT_LEFT": 2, "FRONT_RIGHT": 3, "SIDE_LEFT": 4}
SPEED_FILES = {  # stored per-observation tables and their pinned SHA-256 (FACTS_V7.md item 6)
    # (root environment variable or None for the repository root, relative path, pinned SHA-256)
    "wod_human": ("WAYMO_V2_ROOT", ("derived", "g5", "moving_analysis_set.parquet"),
                  "54016253c11a29c367dc0c2590fcbdf793d380e6a6529cafcb98e2ef6bbff32f"),
    "wod_detector": ("WAYMO_V2_ROOT", ("derived", "ext_e0", "s050", "moving_analysis_set_det.parquet"),
                     "2ca109db8a63d3ad7a43b9bb05dbae3babc44cfe259cbf6ff75425b5e3c1ff54"),
    "av2_replication": (None, ("outputs", "xfc7_v5r", "t3_static_s050", "moving_analysis_set_t2.parquet"),
                        "b6df2c96d5142ac83f33c88ffacce4f97ba8353056d5bd730404920a6d48ec40"),
}


def ci(x):
    """(est, lo, hi) from a stored interval, either {'est', 'lo', 'hi', ...} or [est, lo, hi]."""
    if isinstance(x, dict):
        return (x["est"], x["lo"], x["hi"])
    return (x[0], x[1], x[2])


def clearance(lo, hi, band):
    """Distance by which an interval clears +-band on its rejecting side (negative if it reaches inside)."""
    return max(lo - band, -hi - band)


def second_largest(xs):
    return sorted(xs)[-2]


def second_smallest(xs):
    return sorted(xs)[1]


def speed_summary(values):
    v = np.asarray(values, dtype=float)
    return {"n": int(len(v)), "median": float(np.median(v)), "p90": float(np.percentile(v, 90))}


def _load(cycle, rel):
    with open(os.path.join(cycle, rel), encoding="utf-8") as fh:
        return json.load(fh)


def arith_main(argv):
    cycle, out = argv[2], argv[3]
    res = {"part": "V6R-2 to V6R-7", "plan_addenda": "1 and 2", "computed_utc": None}
    ext = _load(cycle, "extension_2026-10-31/tier1_results_2026-11-01.json")["primary"]["E1"]
    r4 = _load(cycle, "tier2_2026-11-01/posthoc_2026-11-03/r4_results.json")["R4_1_WOD_detector"]
    hum = _load(cycle, "pilot/primary_results_v2.json")
    g5 = _load(cycle, "pilot/g5_summary.json")["cameras"]
    e0 = _load(cycle, "extension_2026-10-31/E0_summary_s050.json")["detector"]
    t3 = _load(cycle, "t3_replication_2026-11-04/cluster_results/results/T3_RESULTS_2026-11-05.json")["primary"]
    v5r = _load(cycle, "t3_replication_2026-11-04/posthoc_2026-11-05/v5r_results.json")["V5R_2"]["analyses"]["W-up"]

    # V6R-2: second-largest frame-start clearance among rejecting cameras, per run; minimum over runs
    det_rej = list(ext["verdicts_primary_run"]["frame"]["rejected_in"])
    if sorted(det_rej) != sorted(ext["verdicts_margin_flag_excluded"]["frame"]["rejected_in"]):
        raise SystemExit("detector rejection sets differ between runs; the margin definition needs both")
    det = {"rejecting": det_rej, "runs": {}}
    det_runs = {"primary": {c: ci(ext["estimates"][c]["delta"]["frame"]) for c in det_rej},
                "margin_flag_excluded": {c: ci(r4["cells"]["margin_flag_excluded"][c]["frame"]["cr1"]) for c in det_rej}}
    for run, cells in det_runs.items():
        cl = {c: clearance(lo, hi, 5 + ext["budgets_ms"][c]) for c, (_, lo, hi) in cells.items()}
        det["runs"][run] = {"clearances_ms": cl, "second_largest_ms": second_largest(list(cl.values()))}
    det["margin_ms"] = min(r["second_largest_ms"] for r in det["runs"].values())
    hum_cells = {"primary": {c: ci(hum["estimates"][c]["delta"]["frame"]) for c in WOD_CAMS},
                 "margin_flag_excluded": {c: ci(hum["robustness"]["margin_flag_excluded"]["delta"][c]["frame"])
                                          for c in WOD_CAMS}}
    hb = {"rejecting": list(WOD_CAMS), "runs": {}}
    for run, cells in hum_cells.items():
        cl = {c: clearance(lo, hi, 5 + hum["budgets_ms"][c]) for c, (_, lo, hi) in cells.items()}
        hb["runs"][run] = {"clearances_ms": cl, "second_largest_ms": second_largest(list(cl.values()))}
    hb["margin_ms"] = min(r["second_largest_ms"] for r in hb["runs"].values())
    res["V6R_2"] = {"wod_detector": det, "wod_human": hb}

    # V6R-3: 5 + B_c + h_c (predicted) per tested camera; dataset-level = second-smallest
    def resolution(budget, hw, cams):
        per = {c: 5 + budget[c] + hw[c] for c in cams}
        return {"per_camera_ms": per, "two_camera_rule_ms": second_smallest(list(per.values()))}
    res["V6R_3"] = {
        "wod_human": resolution(hum["budgets_ms"], {c: g5[c]["predicted_halfwidth_ms"] for c in WOD_CAMS}, WOD_CAMS),
        "wod_detector": resolution({c: e0["budget"][c]["B_c_ms"] for c in WOD_CAMS},
                                   {c: e0["cameras"][c]["predicted_halfwidth_ms"] for c in WOD_CAMS}, WOD_CAMS),
        "av2_replication": resolution(t3["budgets_ms"], t3["gate_bm"]["primary"]["hw_bm_ms"],
                                      [c for c in RING if c in t3["budgets_ms"]]),
    }

    # V6R-5: sweep start at the within-camera slope's upper end (W-up), per composition
    res["V6R_5"] = {comp: {"final": v["final"], "rejected_in": v["rejected_in"]} for comp, v in v5r["verdicts"].items()}

    # V6R-6: front left and side left, smallest three-method clearance of the sweep start, primary run
    iv = t3["three_method"]["intervals"]["primary"]
    v6 = {}
    for c in ("ring_front_left", "ring_side_left"):
        row = {}
        for comp in ("rss", "sum"):
            band = 5 + t3["compositions"][comp]["budgets_ms"][c]
            row[comp] = {
                "unadjusted_ms": min(clearance(ci(iv[c]["start"][m])[1], ci(iv[c]["start"][m])[2], band)
                                     for m in ("cr1", "cr2_bm", "bootstrap")),
                "w_up_ms": min(clearance(ci(v5r["intervals"]["primary"][c]["start"][m])[1],
                                         ci(v5r["intervals"]["primary"][c]["start"][m])[2], band)
                               for m in ("cr1", "cr2_bm", "bootstrap")),
            }
        v6[c] = row
    res["V6R_6"] = v6

    # V6R-7: the deciding intervals of the four rejecting sweep-start cells, primary run
    rej = t3["three_method"]["three"]["runs"]["primary"]["start"]["rejected_in"]
    res["V6R_7"] = {c: {m: ci(iv[c]["start"][m]) for m in ("cr1", "cr2_bm", "bootstrap")} for c in rej}

    # V6R-4: speeds of the moving observations (primary run rows), overall and per rejecting camera
    av2_codes = {i: n for i, n in enumerate(RING, 1)}
    v4 = {}
    for name, (env, rel, pinned) in SPEED_FILES.items():
        path = os.path.join(os.environ[env], *rel) if env else os.path.join(*rel)
        got = sha256(path)
        if got != pinned:
            raise SystemExit(f"{path}: SHA-256 {got} is not the pinned {pinned}; stop")
        df = pd.read_parquet(path, columns=["camera", "speed"])
        entry = {"sha256": got, "overall": speed_summary(df["speed"])}
        if name == "av2_replication":
            entry["per_camera"] = {av2_codes[k]: speed_summary(g["speed"]) for k, g in df.groupby("camera")
                                   if av2_codes[k] in rej}
        else:
            cams = WOD_CAMS if name == "wod_human" else det_rej
            entry["per_camera"] = {c: speed_summary(df.loc[df["camera"] == WOD_CODE[c], "speed"]) for c in cams}
        v4[name] = entry
    res["V6R_4"] = v4
    res["environment"] = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}
    res["computed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(out, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k in ("V6R_2",)}, indent=1)[:1500])
    return 0


# ---------------------------------------------------------------------------------------------------------------
# V6R-8 to V6R-11 (plan Addendum 3). Usage: PYTHONPATH=src python -m careport.xfc7_t3_posthoc_v6r posthoc3 <cycle> <out>
# ---------------------------------------------------------------------------------------------------------------

def lin_intermediate(c1, c2, c3):
    return c1 + max(c2, c3)


def quad_intermediate(c1, c2, c3):
    return float(np.hypot(c1, max(c2, c3)))


def rejected(lo, hi, band):
    return bool(lo > band or hi < -band)


def tipping(clearances, gaps):
    return second_largest([c / g for c, g in zip(clearances, gaps)])


def _comps(d):
    return d["neighbour_bins_ms"], d["ego_class_spread_ms"], d["ego_speed_slope_ms"]


def posthoc3_main(argv):
    cycle, out = argv[2], argv[3]
    ext = _load(cycle, "extension_2026-10-31/tier1_results_2026-11-01.json")["primary"]["E1"]
    r4 = _load(cycle, "tier2_2026-11-01/posthoc_2026-11-03/r4_results.json")
    e0 = _load(cycle, "extension_2026-10-31/E0_summary_s050.json")
    b1 = _load(cycle, "pilot/revision_v2.json")["B1_budget_composition"]["components_ms"]
    k5 = _load(cycle, "pilot/revision2_v2.json")["K5_compositions_all_runs"]["max"]
    g5 = _load(cycle, "pilot/g5_summary.json")["cameras"]
    hum = _load(cycle, "pilot/primary_results_v2.json")
    t3 = _load(cycle, "t3_replication_2026-11-04/cluster_results/results/T3_RESULTS_2026-11-05.json")["primary"]
    s3 = _load(cycle, "t3_replication_2026-11-04/cluster_results/derived/t3_static/S3_summary_s050.json")["budget"]
    v5r = _load(cycle, "t3_replication_2026-11-04/posthoc_2026-11-05/v5r_results.json")["V5R_2"]["analyses"]["W-up"]
    comps_fn = {"lin": lin_intermediate, "quad": quad_intermediate, "sum": lambda a, b, c: a + b + c,
                "max": lambda a, b, c: max(a, b, c)}
    res = {"part": "V6R-8 to V6R-11", "plan_addendum": 3}

    # V6R-8, WOD detector, both runs
    det_iv = {"primary": {c: ci(ext["estimates"][c]["delta"]["frame"]) for c in WOD_CAMS},
              "margin_flag_excluded": {c: ci(r4["R4_1_WOD_detector"]["cells"]["margin_flag_excluded"][c]["frame"]["cr1"])
                                       for c in WOD_CAMS}}
    det = {}
    for name in ("lin", "quad", "sum", "max"):
        B = {c: comps_fn[name](*_comps(e0["detector"]["budget"][c])) for c in WOD_CAMS}
        runs = {}
        for run, cells in det_iv.items():
            rej = [c for c, (_, lo, hi) in cells.items() if rejected(lo, hi, 5 + B[c])]
            cl = {c: clearance(lo, hi, 5 + B[c]) for c, (_, lo, hi) in cells.items()}
            runs[run] = {"rejected_in": rej, "clearances_ms": cl}
        det[name] = {"budgets_ms": B, "runs": runs,
                     "ruled_out": all(len(r["rejected_in"]) >= 2 for r in runs.values())}
    # V6R-8, WOD human boxes, six runs, from the stored planned clearances
    hb = {}
    six = [r for r in k5 if r != "budget_ms"]
    for name in ("lin", "quad", "sum", "max"):
        B = {c: comps_fn[name](*_comps(b1[c])) for c in WOD_CAMS}
        runs = {}
        for run in six:
            cl = {c: k5[run]["frame_clearance_ms"][c] - (B[c] - k5["budget_ms"][c]) for c in WOD_CAMS}
            runs[run] = {"clearances_ms": cl, "rejected_in": [c for c in WOD_CAMS if cl[c] > 0]}
        hb[name] = {"budgets_ms": B, "runs": runs,
                    "runs_ruled_out": sum(len(r["rejected_in"]) >= 2 for r in runs.values()), "of_runs": len(six)}
    # V6R-8, copying tipping points (R4_2 method; gaps implied by the stored planned shares)
    cop = {}
    for seed in ("lidar_seed", "static_point_seed"):
        gaps = {c: k5["primary"]["frame_clearance_ms"][c] / r4["R4_2_copying_threshold"]["cameras"][c]["max"][seed]
                for c in WOD_CAMS}
        row = {"check_max_reproduces_binding": tipping([k5["primary"]["frame_clearance_ms"][c] for c in WOD_CAMS],
                                                       [gaps[c] for c in WOD_CAMS]),
               "stored_binding_max": r4["R4_2_copying_threshold"]["binding"]["max"][seed]}
        for name in ("lin", "quad", "sum"):
            row[name] = tipping([hb[name]["runs"]["primary"]["clearances_ms"][c] for c in WOD_CAMS],
                                [gaps[c] for c in WOD_CAMS])
        cop[seed] = row
    # V6R-8, AV2 three-method, both runs
    iv = t3["three_method"]["intervals"]
    av = {}
    for name in ("lin", "quad", "sum", "max"):
        B = {c: comps_fn[name](*_comps(s3[c])) for c in RING if c in s3}
        runs = {}
        for run in ("primary", "margin_flag_excluded"):
            rej = [c for c in B if all(rejected(ci(iv[run][c]["start"][m])[1], ci(iv[run][c]["start"][m])[2], 5 + B[c])
                                       for m in ("cr1", "cr2_bm", "bootstrap"))]
            runs[run] = rej
        av[name] = {"rejected_in": runs, "ruled_out": all(len(r) >= 2 for r in runs.values())}
    res["V6R_8"] = {"wod_detector": det, "wod_human": hb, "copying_tipping": cop, "av2_replication": av}

    # V6R-9: summed clearances, both runs, front left and side left
    v9 = {}
    for c in ("ring_front_left", "ring_side_left"):
        band = 5 + t3["compositions"]["sum"]["budgets_ms"][c]
        v9[c] = {run: {"unadjusted_ms": min(clearance(ci(iv[run][c]["start"][m])[1], ci(iv[run][c]["start"][m])[2], band)
                                            for m in ("cr1", "cr2_bm", "bootstrap")),
                       "w_up_ms": min(clearance(ci(v5r["intervals"][run][c]["start"][m])[1],
                                                ci(v5r["intervals"][run][c]["start"][m])[2], band)
                                      for m in ("cr1", "cr2_bm", "bootstrap"))}
                 for run in ("primary", "margin_flag_excluded")}
    res["V6R_9"] = v9
    # V6R-10: detector under the sum, margin run
    res["V6R_10"] = det["sum"]["runs"]["margin_flag_excluded"]
    # V6R-11: realized against predicted half-widths
    v11 = {"wod_human": {}, "wod_detector": {}, "av2_replication": {}}
    for c in WOD_CAMS:
        _, lo, hi = ci(hum["estimates"][c]["delta"]["frame"])
        v11["wod_human"][c] = {"realized_ms": (hi - lo) / 2, "predicted_ms": g5[c]["predicted_halfwidth_ms"]}
        _, lo, hi = ci(ext["estimates"][c]["delta"]["frame"])
        v11["wod_detector"][c] = {"realized_ms": (hi - lo) / 2, "predicted_ms": e0["detector"]["cameras"][c]["predicted_halfwidth_ms"]}
    for c in t3["gate_bm"]["primary"]["hw_bm_ms"]:
        _, lo, hi = ci(iv["primary"][c]["start"]["cr2_bm"])
        v11["av2_replication"][c] = {"realized_ms": (hi - lo) / 2, "predicted_ms": t3["gate_bm"]["primary"]["hw_bm_ms"][c]}
    for blk in v11.values():
        for d in blk.values():
            d["ratio_larger_to_smaller"] = max(d["realized_ms"], d["predicted_ms"]) / min(d["realized_ms"], d["predicted_ms"])
    res["V6R_11"] = v11
    res["environment"] = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}
    res["computed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(out, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "arith":
        sys.exit(arith_main(sys.argv))
    if len(sys.argv) > 1 and sys.argv[1] == "posthoc3":
        sys.exit(posthoc3_main(sys.argv))
    sys.exit(main(sys.argv))
