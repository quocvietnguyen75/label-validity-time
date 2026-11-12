"""XF-C7 revision-round POST HOC analyses B1-B3, B5, B7-B9 (pilot record, 2026-10-23; pinned there before running).

Usage: PYTHONPATH=src python -m careport.xfc7_revision_run <vehicle_label_only.parquet> <g5_summary.json> <primary_results_v2.json> <s1_masks.json> <out_dir>
B4 (injected shifts) is in xfc7_revision_shift_run. None of these changes a verdict. Writes <out_dir>/revision_v2.json.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr, t as student_t

from careport.xfc7_decision_v2 import decide_rejection, rejected
from careport.xfc7_estimators import weighted_mean_cr1, wls_intercept_slope_cr1
from careport.xfc7_g5_core import K4, build_sets
from careport.xfc7_g5_run import sha
from careport.xfc7_posthoc import clip_boxes, compose_budget, paired_iou, ruled_out_up_to, tipping_factor
from careport.xfc7_primary import CAM, HYP, RO, add_tau_hat

CB = "[CameraBoxComponent]"
SPEED_BINS, SPEED_LAB = [3.0, 8.0, 15.0, np.inf], ["3-8", "8-15", ">=15"]
COMP = ("neighbour_bins_ms", "ego_class_spread_ms", "ego_speed_slope_ms")


def ci(est, se, crit=1.96):
    return {"est": float(est), "se": float(se), "lo": float(est - crit * se), "hi": float(est + crit * se)}


def tri(d):
    return (d["est"], d["lo"], d["hi"])


def b1(g5, pr, cams):
    out = {"components_ms": {c: {k: g5["budget"][c][k] for k in COMP} for c in g5["budget"]}, "compositions": {}}
    runs = {"primary": {h: {c: tri(pr["estimates"][c]["delta"][h]) for c in cams} for h in HYP},
            "margin_excluded": {h: {c: tri(pr["robustness"]["margin_flag_excluded"]["delta"][c][h]) for c in cams} for h in HYP}}
    for mode in ("max", "quadrature", "sum"):
        bud = {c: compose_budget([g5["budget"][c][k] for k in COMP], mode) for c in g5["budget"]}
        res = {"budget_ms": bud}
        for run, delta in runs.items():
            v = decide_rejection(delta, bud, cams)
            res[run] = {h: {"rejected_in": v[h]["rejected_in"], "ruled_out": v[h]["ruled_out"]} for h in v}
            f = {c: tipping_factor(delta["frame"][c][1], delta["frame"][c][2], bud[c]) for c in cams}
            res[run]["frame_tipping_factors"] = f
            res[run]["frame_ruled_out_up_to"] = ruled_out_up_to(f)
        res["frame_ruled_out_both_runs"] = bool(res["primary"]["frame"]["ruled_out"] and res["margin_excluded"]["frame"]["ruled_out"])
        out["compositions"][mode] = res
    return out


def b2(d, bud, cams):
    out = {}
    for (c, r), g in d.groupby(["camera", "readout"]):
        name, ro = CAM[int(c)], RO[int(r)]
        cell = {"obs": int(len(g)), "tracks": int(g[["segment", "object_id"]].drop_duplicates().shape[0]),
                "segments": int(g["segment"].nunique()),
                "column_third_share": {str(k): float(v) for k, v in g["column_third"].value_counts(normalize=True).sort_index().items()},
                "mean": {}, "intercept": {}, "slope": {}}
        for h, col in HYP.items():
            y = g["tau_hat"] - g[col]
            cell["mean"][h] = ci(*weighted_mean_cr1(y, g["w"], g["segment"]))
            if g["segment"].nunique() > 2 and g["xi"].std() > 0:
                (a, sa), (b, sb) = wls_intercept_slope_cr1(y, g["xi"], g["w"], g["segment"])
                cell["intercept"][h], cell["slope"][h] = ci(a, sa), ci(b, sb)
        if name in bud and "frame" in cell["intercept"]:
            cell["frame_intercept_rejected"] = rejected(tri(cell["intercept"]["frame"]), bud[name])
        cell["tested"] = bool(cell["tracks"] >= 30 and name in cams)
        out[f"{name}|{ro}"] = cell
    return out


def b3(d, pr, s1, cams):
    out = {"retention_human": {}, "retention_mask": {}, "displacement_quartiles": {}}
    for c in cams:
        g = d[d["camera"].map(lambda k: CAM[int(k)]) == c]
        tl = float((g["w"] * g["tau_lidar"]).sum() / g["w"].sum())
        df = pr["estimates"][c]["delta"]["frame"]
        out["retention_human"][c] = {"tau_bar_lidar_ms": tl, "rho_est": df["est"] / tl, "rho_lo": df["lo"] / tl, "rho_hi": df["hi"] / tl}
    for c, x in (s1.get("cameras") or {}).items():
        if not x:
            continue
        df, dl = x["delta"]["frame"], x["delta"]["lidar"]
        tl = df["est"] - dl["est"]
        out["retention_mask"][c] = {"tracks": x["tracks"], "tau_bar_lidar_ms": tl, "rho_est": df["est"] / tl, "rho_lo": df["lo"] / tl,
                                    "rho_hi": df["hi"] / tl}
    D = d["sdot"] * d["tau_lidar"]
    q = np.quantile(D, [0.25, 0.5, 0.75])
    d = d.assign(D=D, Dq=np.digitize(D, q))
    for k, g in d.groupby("Dq"):
        e = ci(*weighted_mean_cr1(g["tau_hat"] - g["tau_lidar"], g["w"], g["segment"]))
        e.update({"obs": int(len(g)), "D_px_median": float(g["D"].median()), "D_px_range": [float(g["D"].min()), float(g["D"].max())]})
        out["displacement_quartiles"][f"Q{int(k) + 1}"] = e
    out["quartile_edges_px"] = [float(v) for v in q]
    return out


def human_boxes(vdir):
    out = []
    for f in sorted(glob.glob(os.path.join(vdir, "camera_box", "*.parquet"))):
        b = pq.read_table(f, columns=["key.segment_context_name", "key.frame_timestamp_micros", "key.camera_name", "key.camera_object_id",
                                      f"{CB}.box.center.x", f"{CB}.box.center.y", f"{CB}.box.size.x", f"{CB}.box.size.y", f"{CB}.type"]).to_pandas()
        b = b[b[f"{CB}.type"] == 1]
        cu, cv, w, h = b[f"{CB}.box.center.x"], b[f"{CB}.box.center.y"], b[f"{CB}.box.size.x"], b[f"{CB}.box.size.y"]
        out.append(pd.DataFrame({"segment": b["key.segment_context_name"], "frame": b["key.frame_timestamp_micros"],
                                 "camera": b["key.camera_name"].astype(int), "camera_object_id": b["key.camera_object_id"],
                                 "b_u0": cu - w / 2, "b_v0": cv - h / 2, "b_u1": cu + w / 2, "b_v1": cv + h / 2}))
    return pd.concat(out, ignore_index=True)


def b5(lab, prim, hb, ref):
    m = lab[(lab["speed"] >= 3.0) & (lab["top_points"] >= 20) & (~lab["overlap"])]
    m = m.merge(prim[prim["camera_object_id"].notna()][K4 + ["camera_object_id"]], on=K4, how="inner")
    m = m.merge(hb, on=["segment", "frame", "camera", "camera_object_id"], how="inner")
    h0 = m[["h0_u0", "h0_v0", "h0_u1", "h0_v1"]].to_numpy(float)
    human = m[["b_u0", "b_v0", "b_u1", "b_v1"]].to_numpy(float)
    W, H = m["width"].to_numpy(float), m["height"].to_numpy(float)
    res, ious = {}, {}
    for name, (cu, cv) in (("frame_start_comment", ("p0_u", "p0_v")), ("lidar_released_law", ("pl_u", "pl_v")),
                           ("no_retiming_capture", ("pc_u", "pc_v"))):
        sh = np.stack([(m[cu] - m["p0_u"]).to_numpy(float), (m[cv] - m["p0_v"]).to_numpy(float)], axis=1)
        hull = h0 + np.concatenate([sh, sh], axis=1)
        ious[name] = paired_iou(clip_boxes(hull, W, H), human)
    m = m.assign(**{f"iou_{k}": v for k, v in ious.items()})
    m["speed_class"] = pd.cut(m["speed"], SPEED_BINS, labels=SPEED_LAB, right=False)
    for grp, g in (("most_visible", m[m["most_visible"]]), ("other_cameras", m[~m["most_visible"]])):
        e = {"obs": int(len(g)), "tracks": int(g[["segment", "object_id"]].drop_duplicates().shape[0])}
        for k in ious:
            x = g[f"iou_{k}"]
            e[k] = {"median": float(x.median()), "share_ge_0_5": float((x >= 0.5).mean()), "share_ge_0_7": float((x >= 0.7).mean())}
        e["by_speed"] = {str(s): {k: float(gg[f"iou_{k}"].median()) for k in ious} | {"obs": int(len(gg))}
                         for s, gg in g.groupby("speed_class", observed=True)}
        res[grp] = e
    # (b) label-only, primary moving set
    r = ref.copy()
    r["speed_class"] = pd.cut(r["speed"], SPEED_BINS, labels=SPEED_LAB, right=False)
    lab_only = {}
    for s, g in r.groupby("speed_class", observed=True):
        nc = g["speed"] * (g["tau_cap"] - g["tau_lidar"]).abs() / 1000.0
        doc = g["speed"] * g["tau_lidar"] / 1000.0
        lab_only[str(s)] = {"obs": int(len(g)), "no_correction_m_median": float(nc.median()), "no_correction_m_p95": float(nc.quantile(0.95)),
                            "documented_m_median": float(doc.median()), "documented_m_p95": float(doc.quantile(0.95)),
                            "no_correction_px_median": float((g["sdot"].abs() * (g["tau_cap"] - g["tau_lidar"]).abs()).median())}
    vmed = float(r["speed"].median())
    lab_only["at_median_speed"] = {"speed_mps": vmed, "m_per_ms": vmed / 1000.0,
                                   "lidar_vs_capture_means_2_3_to_3_2_ms_m": [2.3 * vmed / 1000.0, 3.2 * vmed / 1000.0],
                                   "edge_15_ms_m": 15 * vmed / 1000.0, "edge_29_ms_m": 29 * vmed / 1000.0,
                                   "segment_dev_5_ms_m": 5 * vmed / 1000.0, "segment_dev_10_ms_m": 10 * vmed / 1000.0}
    res["label_only"] = lab_only
    return res


def b7(lab, prim, vdir):
    rows = []
    for f in sorted(glob.glob(os.path.join(vdir, "stats", "*.parquet"))):
        t = pq.read_table(f, columns=["key.segment_context_name", "[StatsComponent].time_of_day", "[StatsComponent].location"]).to_pandas()
        if len(t):
            rows.append((t.iloc[0, 0], t.iloc[:, 1].mode().iloc[0], t.iloc[:, 2].mode().iloc[0]))
    meta = pd.DataFrame(rows, columns=["segment", "time_of_day", "location"])
    ro = lab.groupby("segment")["readout"].agg(lambda x: int(x.mode().iloc[0])).rename("readout").reset_index()
    meta = meta.merge(ro, on="segment", how="inner")
    meta["readout"] = meta["readout"].map(RO)
    tab = {f"{a}|{b}": {k: int(v) for k, v in g["readout"].value_counts().items()} for (a, b), g in meta.groupby(["location", "time_of_day"])}
    st, _, _ = build_sets(lab, prim)
    g = st.groupby(["segment", "camera"])
    seg = pd.DataFrame({"med": g["r"].median(), "tracks": g["object_id"].nunique()}).reset_index()
    seg = seg[seg["tracks"] >= 3]
    pv = seg.pivot(index="segment", columns="camera", values="med")
    pairs = {}
    for a, b in ((1, 2), (1, 3), (2, 4), (1, 4), (3, 5)):
        if a in pv and b in pv:
            x = pv[[a, b]].dropna()
            if len(x) >= 10:
                rho, p = spearmanr(x[a], x[b])
                pairs[f"{CAM[a]}~{CAM[b]}"] = {"segments": int(len(x)), "spearman": float(rho), "p": float(p)}
    return {"readout_by_location_time": tab, "segments_with_meta": int(len(meta)), "segment_offset_correlation": pairs}


def b8(lab, prim, ref):
    m = lab[lab["most_visible"]]
    steps = [("moving_base", (m["speed"] >= 3.0) & (m["top_points"] >= 20) & m["synced_present"])]
    cur = steps[0][1]
    for name, cond in (("inside_image_all_hypotheses", m["inside_all"]), ("no_overlap", ~m["overlap"]),
                       ("sdot_ge_0_25", m["sdot"] >= 0.25), ("not_along_axis", m["along_axis_deg"] >= 20.0)):
        cur = cur & cond
        steps.append((name, cur))
    out = [{"step": n, "rows": int(c.sum())} for n, c in steps]
    out.append({"step": "incl_moving_all_rules", "rows": int(m["incl_moving"].sum())})
    a = m[m["incl_moving"]].merge(prim[K4 + ["camera_object_id"]], on=K4, how="left")
    out.append({"step": "assigned_human_box", "rows": int(a["camera_object_id"].notna().sum())})
    out.append({"step": "zero_line_available_analysis_set", "rows": int(len(ref))})
    by_cam = {}
    for c, g in m.groupby("camera"):
        base = (g["speed"] >= 3.0) & (g["top_points"] >= 20) & g["synced_present"]
        by_cam[CAM[int(c)]] = {"moving_base": int(base.sum()), "incl_moving": int(g["incl_moving"].sum()),
                               "analysis_set": int((ref["camera"] == c).sum())}
    return {"flow": out, "by_camera": by_cam}


def b9(pr, bud, cams):
    out = {}
    runs = {"primary": {c: pr["estimates"][c]["delta"] for c in cams},
            "margin_excluded": {c: pr["robustness"]["margin_flag_excluded"]["delta"][c] for c in cams}}
    crit = {c: float(student_t.ppf(0.975, pr["estimates"][c]["segments"] - 1)) for c in cams}
    for run, dd in runs.items():
        delta = {h: {c: (dd[c][h]["est"], dd[c][h]["est"] - crit[c] * dd[c][h]["se"], dd[c][h]["est"] + crit[c] * dd[c][h]["se"])
                     for c in cams} for h in HYP}
        v = decide_rejection(delta, bud, cams)
        out[run] = {h: {"rejected_in": v[h]["rejected_in"], "ruled_out": v[h]["ruled_out"]} for h in v}
    out["t_critical"] = crit
    return out


def main(argv):
    lab_path, g5_path, pr_path, s1_path, out_dir = argv[1:6]
    g5 = json.load(open(g5_path, encoding="utf-8"))
    pr = json.load(open(pr_path, encoding="utf-8"))
    s1 = json.load(open(s1_path, encoding="utf-8"))
    assert sha(lab_path) == g5["inputs"]["label_only_sha256"]
    assert sha(g5["moving_analysis_set"]["file"]) == g5["moving_analysis_set"]["sha256"]
    vdir = os.path.join(os.environ["WAYMO_V2_ROOT"], "validation")
    lab = pd.read_parquet(lab_path)
    prim = pd.read_parquet(g5["association"]["table"])
    ref = pd.read_parquet(g5["moving_analysis_set"]["file"])
    cams = [c for c, v in g5["cameras"].items() if v["tracks"] >= 30]
    bud = {c: max(g5["budget"][c][k] for k in COMP) for c in g5["budget"]}
    d = add_tau_hat(ref)
    out = {"inputs": {"label_only_sha256": g5["inputs"]["label_only_sha256"], "moving_analysis_set_sha256": g5["moving_analysis_set"]["sha256"],
                      "primary_results_sha256": sha(pr_path), "s1_masks_sha256": sha(s1_path)}}
    out["B1_budget_composition"] = b1(g5, pr, cams)
    out["B2_readout_cells"] = b2(d, bud, cams)
    out["B3_seeding_bound"] = b3(d, pr, s1, cams)
    out["B5_camera_side"] = b5(lab, prim, human_boxes(vdir), ref)
    out["B7_readout_meta_and_offsets"] = b7(lab, prim, vdir)
    out["B8_attrition"] = b8(lab, prim, ref)
    out["B9_t_critical"] = b9(pr, bud, cams)
    path = os.path.join(out_dir, "revision_v2.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=1, default=float)[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
