"""XF-C7 revision-2 POST HOC analyses K1-K3, K5, K6 (pilot record, 2026-10-25; pinned there before running).

Usage: PYTHONPATH=src python -m careport.xfc7_revision2_run <vehicle_label_only.parquet> <g5_summary.json> <pilot_dir> <out_dir>
<pilot_dir> holds primary_results_v2.json, posthoc_v2.json, robustness_v2.json, revision_v2.json, s1_masks.json and
display_v2.json; the D3 example key is read from $WAYMO_V2_ROOT/derived/display/example.json. K4 (injected frame-start
placement) is in xfc7_revision2_shift_run. None of these changes a verdict. Writes <out_dir>/revision2_v2.json.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from careport.xfc7_budget import ego_spread, neighbour_spread
from careport.xfc7_decision_v2 import decide_rejection
from careport.xfc7_estimators import weighted_mean_cr1, wls_slope_cr1
from careport.xfc7_g5_core import K4, build_sets
from careport.xfc7_g5_run import load_boxes, sha
from careport.xfc7_observations import frustum_margin_deg
from careport.xfc7_posthoc import clip_boxes, compose_budget, n_in_gate, paired_iou
from careport.xfc7_primary import CAM, HYP, RO, add_tau_hat, analyse
from careport.xfc7_primary_v2_run import verdicts
from careport.xfc7_robustness_run import CB, CC
from careport.xfc7_revision2 import (cells_mean_cov_cr1, clearance, direction_z, percentile_rank, retention_lower_bound,
                                     split_camera, stratified_camera, wald_chi2)
from careport.xfc7_revision_run import human_boxes
from careport.xfc7_zero_line import LEVELS

COMP = ("neighbour_bins_ms", "ego_class_spread_ms", "ego_speed_slope_ms")
SPEED_BINS, SPEED_LAB = [3.0, 8.0, 15.0, np.inf], ["3-8", "8-15", ">=15"]
Z_BONF = 2.4977          # two-sided 0.05 over four cameras
MOVES = ("frame_start_comment", "lidar_released_law", "no_retiming_capture")


def ci(est, se, crit=1.96):
    return {"est": float(est), "se": float(se), "lo": float(est - crit * se), "hi": float(est + crit * se)}


def reweight(mo):
    mo = mo.copy()
    mo["n_track"] = mo.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
    mo["w"] = mo["sdot"] ** 2 / (mo["sigma_b"] ** 2 * mo["n_track"])
    return mo


def margin_run(mo):
    return reweight(mo[(mo["margin"] >= 0.2) | np.isinf(mo["margin"])])


def cell_estimates(d):
    out = {}
    for (c, r), g in d.groupby(["cam0", "readout"]):
        out[f"{CAM[int(c)]}|{RO[int(r)]}"] = {
            "obs": int(len(g)), "tracks": int(g[["segment", "object_id"]].drop_duplicates().shape[0]), "segments": int(g["segment"].nunique()),
            "delta": {h: ci(*weighted_mean_cr1(g["tau_hat"] - g[col], g["w"], g["segment"])) for h, col in HYP.items()}}
    return out


def strat_budget(st, mo, tables):
    """G5 budget definitions (xfc7_g5_run step 6) applied to the stratified tables; descriptive only."""
    t0, out = tables[0], {}
    for c, g in mo.groupby("camera"):
        cam, ro = split_camera([c])
        sdot_med = float(g["sdot"].median())
        nb = np.nanmean(neighbour_spread(t0, g[LEVELS[0]]))
        es = np.nanmean(ego_spread(t0, g[LEVELS[0]]))
        sc = st[st["camera"] == c]
        X = np.column_stack([np.ones(len(sc)), sc["ego_speed"].to_numpy()])
        W = sc["w"].to_numpy()
        beta = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (W * sc["r"].to_numpy()))[1]
        dspeed = float(g["ego_speed"].median() - sc["ego_speed"].median())
        comp = [float(nb / sdot_med), float(es / sdot_med), float(abs(beta * dspeed) / sdot_med)]
        out[f"{CAM[int(cam[0])]}|{RO[int(ro[0])]}"] = dict(zip(COMP, comp)) | {"max_ms": float(np.nanmax(comp))}
    return out


def k1(lab, prim, ref, bud, cams):
    ro = lab[K4 + ["readout"]]
    lab2 = lab.assign(camera=stratified_camera(lab["camera"], lab["readout"]))
    prim2 = prim.merge(ro, on=K4, how="inner")
    prim2 = prim2.assign(camera=stratified_camera(prim2["camera"], prim2["readout"])).drop(columns=["readout"])
    st, mo, tables = build_sets(lab2, prim2)
    mo = mo.assign(cam0=split_camera(mo["camera"])[0])
    d = add_tau_hat(mo)
    ref_cells = ref.groupby(["camera", "readout"]).size()
    lost = {f"{CAM[int(c)]}|{RO[int(r)]}": int(n - ((d["cam0"] == c) & (d["readout"] == r)).sum()) for (c, r), n in ref_cells.items()}
    runs = {"primary": cell_estimates(d), "margin_excluded": cell_estimates(add_tau_hat(margin_run(mo)))}
    out = {"rows": int(len(d)), "rows_pooled_zero_line": int(len(ref)), "lost_zero_line_by_cell": lost, "cells": runs["primary"],
           "cells_margin_excluded": runs["margin_excluded"], "budget_frozen_ms": bud, "within_direction": {}}
    for rname in ("L2R", "R2L"):
        res = {}
        for run, cells in runs.items():
            tested = [c for c in cams if f"{c}|{rname}" in cells and cells[f"{c}|{rname}"]["tracks"] >= 30]
            delta = {h: {c: (cells[f"{c}|{rname}"]["delta"][h]["est"], cells[f"{c}|{rname}"]["delta"][h]["lo"],
                             cells[f"{c}|{rname}"]["delta"][h]["hi"]) for c in tested} for h in HYP}
            v = decide_rejection(delta, bud, tested)
            res[run] = {"tested": tested, "verdicts": {h: {"rejected_in": v[h]["rejected_in"], "ruled_out": v[h]["ruled_out"]} for h in v},
                        "frame_clearance_ms": {c: clearance(delta["frame"][c][1], delta["frame"][c][2], 5.0 + bud[c]) for c in tested}}
        res["ruled_out_both_runs"] = {h: bool(res["primary"]["verdicts"][h]["ruled_out"] and res["margin_excluded"]["verdicts"][h]["ruled_out"])
                                      for h in HYP}
        out["within_direction"][rname] = res
    # heterogeneity between directions
    het = {}
    both = [c for c in cams if all(f"{c}|{r}" in runs["primary"] and runs["primary"][f"{c}|{r}"]["tracks"] >= 30 for r in ("L2R", "R2L"))]
    for h, col in HYP.items():
        per = {}
        for c in both:
            a, b = runs["primary"][f"{c}|L2R"]["delta"][h], runs["primary"][f"{c}|R2L"]["delta"][h]
            z = direction_z(a["est"], a["se"], b["est"], b["se"])
            per[c] = {"diff_ms": a["est"] - b["est"], "se_ms": float(np.hypot(a["se"], b["se"])), "z": z,
                      "detectable_ms_2_8se": float(2.8 * np.hypot(a["se"], b["se"]))}
        mus, covs = {}, {}
        for r, rname in RO.items():
            g = d[(d["readout"] == r) & d["cam0"].map(lambda k: CAM[int(k)] in both)]
            cells, mu, V = cells_mean_cov_cr1(g["tau_hat"] - g[col], g["w"], g["cam0"].map(lambda k: CAM[int(k)]), g["segment"])
            mus[rname], covs[rname] = dict(zip(cells, mu)), (list(cells), V)
        order = covs["L2R"][0]
        assert order == covs["R2L"][0]
        diff = np.array([mus["L2R"][c] - mus["R2L"][c] for c in order])
        stat, df, p = wald_chi2(diff, covs["L2R"][1] + covs["R2L"][1])
        het[h] = {"cameras": per, "max_abs_z": float(max(abs(x["z"]) for x in per.values())), "bonferroni_z": Z_BONF,
                  "heterogeneity_found": bool(max(abs(x["z"]) for x in per.values()) > Z_BONF),
                  "joint_wald": {"cameras": order, "chi2": stat, "df": df, "p": p}}
    out["heterogeneity"] = het
    # descriptive: stratified budget components and pooled column slopes per direction
    out["budget_stratified_descriptive"] = strat_budget(st, mo, tables)
    slopes = {}
    for r, g in d.groupby("readout"):
        slopes[RO[int(r)]] = {h: ci(*wls_slope_cr1(g["tau_hat"] - g[HYP[h]], g["xi"], g["w"], g["cam0"], g["segment"])) for h in ("lidar", "cam")}
        slopes[RO[int(r)]]["obs"] = int(len(g))
    out["column_slopes_stratified_descriptive"] = slopes
    return out


def iou_frame(lab, prim, hb):
    m = lab[(lab["speed"] >= 3.0) & (lab["top_points"] >= 20) & (~lab["overlap"])]
    m = m.merge(prim[prim["camera_object_id"].notna()][K4 + ["camera_object_id"]], on=K4, how="inner")
    m = m.merge(hb, on=["segment", "frame", "camera", "camera_object_id"], how="inner")
    h0 = m[["h0_u0", "h0_v0", "h0_u1", "h0_v1"]].to_numpy(float)
    human = m[["b_u0", "b_v0", "b_u1", "b_v1"]].to_numpy(float)
    W, H = m["width"].to_numpy(float), m["height"].to_numpy(float)
    for name, (cu, cv) in zip(MOVES, (("p0_u", "p0_v"), ("pl_u", "pl_v"), ("pc_u", "pc_v"))):
        sh = np.stack([(m[cu] - m["p0_u"]).to_numpy(float), (m[cv] - m["p0_v"]).to_numpy(float)], axis=1)
        m[f"iou_{name}"] = paired_iou(clip_boxes(h0 + np.concatenate([sh, sh], axis=1), W, H), human)
    m["speed_class"] = pd.cut(m["speed"], SPEED_BINS, labels=SPEED_LAB, right=False)
    return m


def iou_summary(g):
    e = {"obs": int(len(g)), "segments": int(g["segment"].nunique())}
    if len(g) == 0:
        return e
    e["median_iou"] = {k: float(g[f"iou_{k}"].median()) for k in MOVES}
    for lab_, other in (("comment_minus_released", "lidar_released_law"), ("comment_minus_no_retiming", "no_retiming_capture")):
        x = g["iou_frame_start_comment"] - g[f"iou_{other}"]
        e[lab_] = {"median": float(x.median()), "mean": ci(*weighted_mean_cr1(x, np.ones(len(x)), g["segment"]))} if g["segment"].nunique() > 1 \
            else {"median": float(x.median())}
    return e


def by_speed(g):
    return {"all": iou_summary(g)} | {str(s): iou_summary(gg) for s, gg in g.groupby("speed_class", observed=True)}


def k2(lab, prim, vdir):
    m = iou_frame(lab, prim, human_boxes(vdir))
    boxes = load_boxes(vdir)
    bg = {k: g for k, g in boxes.groupby(["segment", "frame", "camera"])}
    cnt = pd.Series(0, index=m.index)
    for k, g in m.groupby(["segment", "frame", "camera"]):
        b = bg.get(k)
        bb = np.zeros((0, 2)) if b is None else b[["u", "v"]].to_numpy(float)
        cnt.loc[g.index] = n_in_gate(g[["anchor_u", "anchor_v"]].to_numpy(float), g["gate"].to_numpy(float), bb)
    m["n_in_gate"] = cnt
    out = {}
    for grp, g in (("most_visible", m[m["most_visible"]]), ("other_cameras", m[~m["most_visible"]])):
        out[grp] = {"all_rows": by_speed(g),
                    "inside_all": by_speed(g[g["inside_all"]]), "not_inside_all": by_speed(g[~g["inside_all"]]),
                    "one_box_in_gate": by_speed(g[g["n_in_gate"] == 1]), "two_plus_in_gate": by_speed(g[g["n_in_gate"] >= 2]),
                    "share_inside_all": float(g["inside_all"].mean()), "share_one_box_in_gate": float((g["n_in_gate"] == 1).mean())}
    return out


def k3(g5, pr, ph, s1, bud, cams):
    out = {"human": {}, "mask": {}}
    for c in cams:
        lo = pr["estimates"][c]["delta"]["frame"]["lo"]
        pc = ph["PC1_projected_reference"]["cameras"][c]["delta"]["lidar"]["est"]
        tl, tc = g5["cameras"][c]["predicted_tau_bar"]["lidar"], g5["cameras"][c]["predicted_tau_bar"]["cam"]
        tol = 5.0 + bud[c]
        out["human"][c] = {"lo_frame_ms": lo, "pc_bias_ms": pc, "tolerance_ms": tol, "tau_bar_lidar_ms": tl, "tau_bar_cap_ms": tc,
                           "lidar_seed": {"zero_bias": retention_lower_bound(lo, 0.0, tl), "pc_bias": retention_lower_bound(lo, pc, tl),
                                          "tolerance": retention_lower_bound(lo, tol, tl)},
                           "static_point_seed": {"zero_bias": retention_lower_bound(lo, 0.0, tc), "pc_bias": retention_lower_bound(lo, pc, tc),
                                                 "tolerance": retention_lower_bound(lo, tol, tc)}}
    for c, x in (s1.get("cameras") or {}).items():
        if not x or c not in bud:
            continue
        df, tol = x["delta"]["frame"], 5.0 + bud[c]
        tl, tc = df["est"] - x["delta"]["lidar"]["est"], df["est"] - x["delta"]["cam"]["est"]
        out["mask"][c] = {"tracks": x["tracks"], "lo_frame_ms": df["lo"], "tau_bar_lidar_ms": tl, "tau_bar_cap_ms": tc,
                          "lidar_seed": {"zero_bias": retention_lower_bound(df["lo"], 0.0, tl), "tolerance": retention_lower_bound(df["lo"], tol, tl)},
                          "static_point_seed": {"zero_bias": retention_lower_bound(df["lo"], 0.0, tc),
                                                "tolerance": retention_lower_bound(df["lo"], tol, tc)}}
    return out


def subset_deltas(ref, prim, vdir, cams, bud, rb):
    """Level-2 and frustum-edge runs, rebuilt with xfc7_robustness_run's definitions (only their verdicts were saved)."""
    diff = []
    for f in sorted(glob.glob(os.path.join(vdir, "camera_box", "*.parquet"))):
        t = pq.read_table(f, columns=["key.segment_context_name", "key.frame_timestamp_micros", "key.camera_name", "key.camera_object_id",
                                      f"{CB}.difficulty_level.detection", f"{CB}.type"]).to_pandas()
        t = t[t[f"{CB}.type"] == 1]
        diff.append(pd.DataFrame({"segment": t["key.segment_context_name"], "frame": t["key.frame_timestamp_micros"],
                                  "camera": t["key.camera_name"].astype(int), "camera_object_id": t["key.camera_object_id"],
                                  "difficulty": t[f"{CB}.difficulty_level.detection"]}))
    diff = pd.concat(diff, ignore_index=True)
    mo_d = ref.merge(prim[K4 + ["camera_object_id"]], on=K4, how="left").merge(diff, on=["segment", "frame", "camera", "camera_object_id"], how="left")
    cal = []
    for f in sorted(glob.glob(os.path.join(vdir, "camera_calibration", "*.parquet"))):
        t = pq.read_table(f).to_pandas()
        cal.append(pd.DataFrame({"segment": t["key.segment_context_name"], "camera": t["key.camera_name"].astype(int),
                                 "f_u": t[f"{CC}.intrinsic.f_u"], "f_v": t[f"{CC}.intrinsic.f_v"], "c_u": t[f"{CC}.intrinsic.c_u"],
                                 "c_v": t[f"{CC}.intrinsic.c_v"], "W": t[f"{CC}.width"], "H": t[f"{CC}.height"]}))
    mf = ref.merge(pd.concat(cal, ignore_index=True), on=["segment", "camera"], how="left")
    mf["margin_deg"] = frustum_margin_deg(mf["p0_u"], mf["p0_v"], mf["f_u"], mf["f_v"], mf["c_u"], mf["c_v"], mf["W"], mf["H"])
    out = {}
    for name, sub in (("level2_excluded", mo_d[mo_d["difficulty"] != 2]), ("frustum_edge_5deg_excluded", mf[mf["margin_deg"] >= 5.0])):
        r = analyse(reweight(sub), [], [])
        v = verdicts(r, cams, bud)
        assert all(v[h]["rejected_in"] == rb["runs"][name]["verdicts"][h]["rejected_in"] for h in v), name   # reproduces the saved run
        out[name] = {c: r["cameras"][c]["delta"] for c in cams}
    return out


def k5(g5, pr, rb, cams, extra):
    runs = {"primary": {c: pr["estimates"][c]["delta"] for c in cams},
            "margin_excluded": {c: pr["robustness"]["margin_flag_excluded"]["delta"][c] for c in cams}}
    runs |= {k: {c: {"frame": v["delta_frame"][c]} for c in cams} for k, v in rb["runs"].items() if "delta_frame" in v}
    runs |= extra
    out = {}
    for mode in ("max", "quadrature", "sum"):
        bud = {c: compose_budget([g5["budget"][c][k] for k in COMP], mode) for c in cams}
        res = {"budget_ms": bud}
        for run, dd in runs.items():
            hs = [h for h in ("frame", "pose") if all(h in dd[c] for c in dd)]
            delta = {h: {c: (dd[c][h]["est"], dd[c][h]["lo"], dd[c][h]["hi"]) for c in dd} for h in hs}
            v = decide_rejection(delta, bud, list(dd))
            res[run] = {h: {"rejected_in": v[h]["rejected_in"], "ruled_out": v[h]["ruled_out"]} for h in hs}
            res[run]["frame_clearance_ms"] = {c: clearance(dd[c]["frame"]["lo"], dd[c]["frame"]["hi"], 5.0 + bud[c]) for c in dd}
            res[run]["pose_available"] = "pose" in hs
        out[mode] = res
    return out


def k6(ref, ex):
    d = add_tau_hat(ref)
    f = d[d["camera"] == 1]
    seg, frame, cam, oid = ex["key"].split("|")
    row = f[(f["segment"] == seg) & (f["frame"] == int(frame)) & (f["object_id"] == oid)]
    assert len(row) == 1
    r = row.iloc[0]
    D_f, D_all = f["sdot"] * f["tau_lidar"], d["sdot"] * d["tau_lidar"]
    th = f["tau_hat"]
    return {"sdot_px_per_ms": float(r["sdot"]), "displacement_px": float(r["sdot"] * r["tau_lidar"]),
            "pct_sdot_front": percentile_rank(f["sdot"], r["sdot"]), "pct_sdot_all": percentile_rank(d["sdot"], r["sdot"]),
            "pct_displacement_front": percentile_rank(D_f, r["sdot"] * r["tau_lidar"]),
            "pct_displacement_all": percentile_rank(D_all, r["sdot"] * r["tau_lidar"]),
            "tau_hat_front_unweighted": {"median": float(th.median()), "scaled_mad": float(1.4826 * (th - th.median()).abs().median()),
                                         "p05": float(th.quantile(0.05)), "p95": float(th.quantile(0.95))},
            "tau_lidar_ms": float(r["tau_lidar"]), "tau_cap_ms": float(r["tau_cap"]),
            "lidar_to_capture_offset_px": float(r["sdot"] * (r["tau_cap"] - r["tau_lidar"]))}


def main(argv):
    lab_path, g5_path, pilot, out_dir = argv[1:5]
    g5 = json.load(open(g5_path, encoding="utf-8"))
    P = {k: os.path.join(pilot, f"{k}.json") for k in ("primary_results_v2", "posthoc_v2", "robustness_v2", "s1_masks")}
    pr, ph, rb, s1 = (json.load(open(P[k], encoding="utf-8")) for k in ("primary_results_v2", "posthoc_v2", "robustness_v2", "s1_masks"))
    assert sha(lab_path) == g5["inputs"]["label_only_sha256"]
    assert sha(g5["moving_analysis_set"]["file"]) == g5["moving_analysis_set"]["sha256"]
    root = os.environ["WAYMO_V2_ROOT"]
    vdir = os.path.join(root, "validation")
    lab = pd.read_parquet(lab_path)
    prim = pd.read_parquet(g5["association"]["table"])
    ref = pd.read_parquet(g5["moving_analysis_set"]["file"])
    cams = [c for c, v in g5["cameras"].items() if v["tracks"] >= 30]
    bud = {c: max(g5["budget"][c][k] for k in COMP) for c in g5["budget"]}
    ex = json.load(open(os.path.join(root, "derived", "display", "example.json"), encoding="utf-8"))
    out = {"inputs": {"label_only_sha256": g5["inputs"]["label_only_sha256"], "moving_analysis_set_sha256": g5["moving_analysis_set"]["sha256"],
                      "association_table_sha256": sha(g5["association"]["table"])} | {f"{k}_sha256": sha(p) for k, p in P.items()},
           "pinned_record_sha256_at_pin": "87facba22d8df50a354ff105ae41bef646aeb8ae4a4d511648b6925d590d1cc6"}
    out["K1_readout_direction"] = k1(lab, prim, ref, bud, cams)
    out["K2_other_camera_iou"] = k2(lab, prim, vdir)
    out["K3_retention_bounds"] = k3(g5, pr, ph, s1, bud, cams)
    out["K5_compositions_all_runs"] = k5(g5, pr, rb, cams, subset_deltas(ref, prim, vdir, cams, bud, rb))
    out["K6_example_context"] = k6(ref, ex)
    path = os.path.join(out_dir, "revision2_v2.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print("written", path, sha(path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
