"""XF-C7 POST HOC analyses PC1, PC2, T1, U1-U3 and C1 (pilot record, 2026-10-22; pinned there before running).

Usage: PYTHONPATH=src python -m careport.xfc7_posthoc_run <vehicle_label_only.parquet> <g5_summary.json> <primary_results_v2.json> <out_dir>
None of these changes a verdict. Writes <out_dir>/posthoc_v2.json (aggregates only).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from careport.xfc7_estimators import weighted_mean_cr1
from careport.xfc7_g5_core import K4, build_sets
from careport.xfc7_g5_run import load_boxes, sha
from careport.xfc7_posthoc import n_in_gate, ruled_out_up_to, tipping_factor
from careport.xfc7_primary import CAM, HYP, RO, add_tau_hat, analyse
from careport.xfc7_primary_v2_run import verdicts

PB = "[ProjectedLiDARBoxComponent]"
SPEED_BINS = [3.0, 8.0, 15.0, np.inf]
SPEED_LAB = ["3-8", "8-15", ">=15"]


def reweight(mo: pd.DataFrame) -> pd.DataFrame:
    mo = mo.copy()
    mo["n_track"] = mo.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
    mo["w"] = mo["sdot"] ** 2 / (mo["sigma_b"] ** 2 * mo["n_track"])
    return mo


def summary(r, cams, budgets):
    out = {"verdicts": verdicts(r, cams, budgets), "cameras": {}, "slopes": {}}
    for c, x in r["cameras"].items():
        out["cameras"][c] = {"obs": x["obs"], "tracks": x["tracks"], "segments": x["segments"],
                             "delta": {h: {k: x["delta"][h][k] for k in ("est", "lo", "hi")} for h in x["delta"]}}
    for ro, x in r["slopes"].items():
        out["slopes"][ro] = {h: {k: x[h][k] for k in ("est", "lo", "hi")} for h in HYP} | {"obs": x["obs"]}
    return out


def projected_boxes(vdir):
    out = []
    for f in sorted(glob.glob(os.path.join(vdir, "projected_lidar_box", "*.parquet"))):
        t = pq.read_table(f, columns=["key.segment_context_name", "key.frame_timestamp_micros", "key.camera_name", "key.laser_object_id",
                                      f"{PB}.box.center.x", f"{PB}.box.center.y", f"{PB}.box.size.x", f"{PB}.type"]).to_pandas()
        t = t[t[f"{PB}.type"] == 1]
        out.append(pd.DataFrame({"segment": t["key.segment_context_name"], "frame": t["key.frame_timestamp_micros"],
                                 "camera": t["key.camera_name"].astype(int), "object_id": t["key.laser_object_id"],
                                 "pu": t[f"{PB}.box.center.x"], "pv": t[f"{PB}.box.center.y"], "pw": t[f"{PB}.box.size.x"]}))
    return pd.concat(out, ignore_index=True)


def time_of_day(vdir):
    rows = []
    for f in sorted(glob.glob(os.path.join(vdir, "stats", "*.parquet"))):
        t = pq.read_table(f, columns=["key.segment_context_name", "[StatsComponent].time_of_day"]).to_pandas()
        if len(t):
            rows.append((t.iloc[0, 0], t.iloc[:, 1].mode().iloc[0]))
    return pd.DataFrame(rows, columns=["segment", "time_of_day"])


def pooled(d, col):
    est, se = weighted_mean_cr1(d["tau_hat"] - d[col], d["w"], d["segment"])
    return {"est": est, "lo": est - 1.96 * se, "hi": est + 1.96 * se, "obs": int(len(d)),
            "tracks": int(d[["segment", "object_id"]].drop_duplicates().shape[0])}


def main(argv):
    lab_path, g5_path, prim_path, out_dir = argv[1:5]
    g5 = json.load(open(g5_path, encoding="utf-8"))
    pr = json.load(open(prim_path, encoding="utf-8"))
    assert sha(lab_path) == g5["inputs"]["label_only_sha256"]
    assert sha(g5["moving_analysis_set"]["file"]) == g5["moving_analysis_set"]["sha256"]
    vdir = os.path.join(os.environ["WAYMO_V2_ROOT"], "validation")
    lab = pd.read_parquet(lab_path)
    prim = pd.read_parquet(g5["association"]["table"])
    ref = pd.read_parquet(g5["moving_analysis_set"]["file"])
    cams = [c for c, v in g5["cameras"].items() if v["tracks"] >= 30]
    budgets = {c: max(v["neighbour_bins_ms"], v["ego_class_spread_ms"], v["ego_speed_slope_ms"]) for c, v in g5["budget"].items()}
    out = {"inputs": {"label_only_sha256": g5["inputs"]["label_only_sha256"], "moving_analysis_set_sha256": g5["moving_analysis_set"]["sha256"],
                      "primary_results_sha256": sha(prim_path)}}
    # PC1: Waymo's projected boxes as the reference
    pb = projected_boxes(vdir)
    a = prim[prim["camera_object_id"].notna()].merge(pb, on=K4, how="inner")
    a["u_ref"], a["v_ref"], a["w_ref"] = a["pu"], a["pv"], a["pw"]
    _, mo_p, _ = build_sets(lab, a)
    mo_p = reweight(mo_p.merge(ref[K4], on=K4, how="inner"))
    r_p = analyse(mo_p, [], [])
    out["PC1_projected_reference"] = summary(r_p, cams, budgets) | {"rows": int(len(mo_p))}
    # display: human minus projected-box tau-hat on common rows, primary weights
    h = add_tau_hat(ref)[K4 + ["tau_hat", "w", "speed", "sdot"]].rename(columns={"tau_hat": "th"})
    p = add_tau_hat(mo_p)[K4 + ["tau_hat"]].rename(columns={"tau_hat": "tp"})
    hp = h.merge(p, on=K4, how="inner")
    hp["diff"] = hp["th"] - hp["tp"]
    disp = {}
    for c, g in hp.groupby("camera"):
        est, se = weighted_mean_cr1(g["diff"], g["w"], g["segment"])
        disp[CAM[int(c)]] = {"rows": int(len(g)), "mean_ms": est, "lo": est - 1.96 * se, "hi": est + 1.96 * se,
                             "median_abs_px": float((g["diff"].abs() * g["sdot"]).median()),
                             "median_abs_m": float((g["diff"].abs() * g["speed"] / 1000.0).median())}
    out["PC1_display_human_minus_projected"] = disp
    # PC2: synthetic references (label-only projection centres), z = 0, primary weights
    m2 = ref.drop(columns=["u_ref", "v_ref"]).merge(lab[K4 + ["pc_u", "pc_v", "pl_u", "pl_v"]], on=K4, how="left")
    pc2 = {}
    for name, (cu, cv), col in (("cam", ("pc_u", "pc_v"), "tau_cap"), ("lidar", ("pl_u", "pl_v"), "tau_lidar")):
        d = m2.assign(u_ref=m2[cu], v_ref=m2[cv], z_b=0.0)
        d = add_tau_hat(d)
        pc2[name] = {CAM[int(c)]: {k: v for k, v in pooled(g, col).items() if k in ("est", "lo", "hi")} for c, g in d.groupby("camera")}
    out["PC2_synthetic"] = pc2
    # T1: tolerance tipping point for H_frame
    t1 = {}
    for run, delta in (("primary", pr["estimates"]), ("margin_excluded", pr["robustness"]["margin_flag_excluded"]["delta"])):
        f = {}
        for c in cams:
            x = delta[c]["delta"]["frame"] if "delta" in delta[c] else delta[c]["frame"]
            f[c] = tipping_factor(x["lo"], x["hi"], budgets[c])
        t1[run] = {"factors": f, "ruled_out_up_to": ruled_out_up_to(f)}
    t1["ruled_out_up_to"] = min(t1["primary"]["ruled_out_up_to"], t1["margin_excluded"]["ruled_out_up_to"])
    out["T1_tipping_point_frame"] = t1
    # U1: uncrowded subset (exactly one human vehicle box inside the gate)
    boxes = load_boxes(vdir)
    bg = {k: g for k, g in boxes.groupby(["segment", "frame", "camera"])}
    m3 = ref.merge(lab[K4 + ["anchor_u", "anchor_v", "gate"]], on=K4, how="left")
    cnt = pd.Series(0, index=m3.index)
    for k, g in m3.groupby(["segment", "frame", "camera"]):
        b = bg.get(k)
        bb = np.zeros((0, 2)) if b is None else b[["u", "v"]].to_numpy(float)
        cnt.loc[g.index] = n_in_gate(g[["anchor_u", "anchor_v"]].to_numpy(float), g["gate"].to_numpy(float), bb)
    u1 = reweight(m3[cnt == 1])
    out["U1_uncrowded"] = summary(analyse(u1, [], []), cams, budgets) | {"rows": int(len(u1)), "share_of_primary": float(len(u1) / len(ref))}
    # U2: day versus not-day
    tod = time_of_day(vdir)
    m4 = ref.merge(tod, on="segment", how="left")
    u2 = {}
    for lab_, sub in (("day", m4[m4["time_of_day"] == "Day"]), ("not_day", m4[m4["time_of_day"] != "Day"])):
        sub = reweight(sub)
        r = analyse(sub, [], [])
        u2[lab_] = {"rows": int(len(sub)), "segments": int(sub["segment"].nunique()),
                    "cameras": {c: {h: {k: x["delta"][h][k] for k in ("est", "lo", "hi")} for h in ("frame", "lidar", "cam")}
                                | {"tracks": x["tracks"]} for c, x in r["cameras"].items()}}
    out["U2_time_of_day"] = u2
    # U3: speed classes, pooled over cameras
    d = add_tau_hat(ref)
    d["speed_class"] = pd.cut(d["speed"], SPEED_BINS, labels=SPEED_LAB, right=False)
    out["U3_speed_classes"] = {str(k): {"lidar": pooled(g, "tau_lidar"), "cam": pooled(g, "tau_cap")}
                               for k, g in d.groupby("speed_class", observed=True)}
    # C1: cost of the documented formula (label-only)
    c1 = {}
    for k, g in d.groupby("speed_class", observed=True):
        m = g["speed"] * g["tau_lidar"] / 1000.0
        px = g["sdot"] * g["tau_lidar"]
        c1[str(k)] = {"obs": int(len(g)), "median_m": float(m.median()), "p95_m": float(m.quantile(0.95)),
                      "median_px": float(px.median()), "p95_px": float(px.quantile(0.95))}
    out["C1_documented_formula_cost"] = c1
    path = os.path.join(out_dir, "posthoc_v2.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print("PC1 rows", out["PC1_projected_reference"]["rows"])
    for c, x in out["PC1_projected_reference"]["cameras"].items():
        print(" PC1", c, x["tracks"], {h: round(v["est"], 1) for h, v in x["delta"].items()})
    print(" PC1 slopes", {ro: {h: (round(x[h]["est"], 1), round(x[h]["lo"], 1), round(x[h]["hi"], 1)) for h in ("lidar", "cam")} for ro, x in out["PC1_projected_reference"]["slopes"].items()})
    print(" PC1 verdicts", {h: (v["ruled_out"], v["rejected_in"]) for h, v in out["PC1_projected_reference"]["verdicts"].items()})
    print("display", {c: (round(v["mean_ms"], 1), round(v["median_abs_px"], 1), round(v["median_abs_m"], 2)) for c, v in disp.items()})
    print("PC2", {h: {c: round(v["est"], 2) for c, v in x.items()} for h, x in pc2.items()})
    print("T1", json.dumps(t1))
    print("U1", out["U1_uncrowded"]["rows"], round(out["U1_uncrowded"]["share_of_primary"], 3), {h: (v["ruled_out"], v["rejected_in"]) for h, v in out["U1_uncrowded"]["verdicts"].items()})
    for c, x in out["U1_uncrowded"]["cameras"].items():
        print("  U1", c, x["tracks"], {h: round(v["est"], 1) for h, v in x["delta"].items()})
    for k, v in u2.items():
        print("U2", k, v["rows"], v["segments"], {c: {h: round(x[h]["est"], 1) for h in ("frame", "lidar", "cam")} for c, x in v["cameras"].items()})
    print("U3", {k: {h: (round(v[h]["est"], 1), round(v[h]["lo"], 1), round(v[h]["hi"], 1), v[h]["obs"]) for h in ("lidar", "cam")} for k, v in out["U3_speed_classes"].items()})
    print("C1", json.dumps(c1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
