"""XF-C7 tier 2, steps S1, S3 and S4 for AV2 (T2 protocol v0.1 sections 8 and 9). Static and label side only.

Mirrors xfc7_ext_e0.run_reference (the frozen G5 procedure: association at gate scale 1.0, static zero line, weights,
design effects, the three budget components), with AV2 ring-camera names, the theta hypotheses of section 5 and
Student-t critical values with G - 1 degrees of freedom (section 7). Like G5 and E0 it NEVER computes theta-hat, a
residual or a displacement of a moving vehicle: moving rows get association, zero line and weights only.

S1: median 2D centre offset |p0 - u_ref| of assigned parked vehicles per camera, pass below 15 px; rule 'written'
uses all of them, rule 'far40' those beyond 40 m. A camera failing S1 under the chosen rule is not a decision camera.
S4: predicted theta-bar per hypothesis and camera (final weights), and the capacity table per contrast.

Usage: PYTHONPATH=src python -m careport.xfc7_t2_static <label_side.parquet> <raw_det_dir> <score_min> <nms_iou>
       <summary.json> <out_dir> [av2|ts] [written|far40]"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from careport.xfc7_association import associate_images
from careport.xfc7_budget import ego_spread, neighbour_spread
from careport.xfc7_estimators import design_effect_mean
from careport.xfc7_ext_detect import reference_table
from careport.xfc7_t2_detect import RING
from careport.xfc7_zero_line import LEVELS, lookup_zero_line, zero_line_tables

K4 = ["segment", "frame", "camera", "object_id"]
CAM_T2 = {i: n for i, n in enumerate(RING, 1)}
HYP_T2 = {"start": "theta_start", "A": "theta_A", "B": "theta_B", "mid": "theta_mid"}
CONTRASTS = [("start", "A"), ("start", "B"), ("start", "mid"), ("A", "B"), ("A", "mid"), ("B", "mid")]
S1_LIMIT_PX = 15.0


def t_crit(g):
    return float(stats.t.ppf(0.975, g - 1)) if g >= 2 else float("nan")


def predicted_halfwidth(de, v, g):
    return float(t_crit(g) * np.sqrt(de * v))


def capacity(theta, thr, contrasts=None):
    """theta: camera -> {hypothesis: predicted theta-bar}; thr: camera -> 5 + B_c + hw_c."""
    contrasts = CONTRASTS if contrasts is None else contrasts
    cams = {}
    for c, th in theta.items():
        cams[c] = {}
        for a, b in contrasts:
            gap = abs(th[a] - th[b])
            cams[c][f"{a}-{b}"] = {"gap": float(gap), "threshold": float(thr[c]), "decidable": bool(gap > thr[c])}
    ds = {f"{a}-{b}": sum(cams[c][f"{a}-{b}"]["decidable"] for c in cams) >= 2 for a, b in contrasts}
    return {"cameras": cams, "dataset": ds}


def projection_check(st, names, limit=S1_LIMIT_PX):
    out = {}
    for c, g in st.groupby("camera"):
        off = np.hypot(g["p0_u"] - g["u_ref"], g["p0_v"] - g["v_ref"])
        med = float(np.median(off))
        out[names.get(int(c), str(c))] = {"n": int(len(g)), "median_px": med, "p90_px": float(np.quantile(off, 0.9)),
                                          "pass": bool(med < limit)}
    return out


S1_RULES = {"written": None, "far40": 40.0}            # rule -> minimum range (m) of the parked rows used
MIN_DECISION_TRACKS = 30


def s1_table(st_all, names, rule="written"):
    """S1 under a named rule: 'written' uses all assigned parked rows (T2 section 8 as drafted); 'far40' only those
    beyond 40 m, where the box-geometry offset is small (a user choice at the freeze, section 12)."""
    if rule not in S1_RULES:
        raise ValueError(f"unknown S1 rule {rule!r}")
    r = S1_RULES[rule]
    return projection_check(st_all if r is None else st_all[st_all["range_m"] > r], names)


def decision_cameras(cams, s1, min_tracks=MIN_DECISION_TRACKS):
    """Cameras with >= min_tracks moving tracks that pass S1 (section 8: a camera failing S1 is excluded)."""
    return sorted(c for c, v in cams.items() if v["tracks"] >= min_tracks and s1.get(c, {}).get("pass", False))


def run_reference_t2(lab, boxes, hyp=None, names=None):
    hyp = HYP_T2 if hyp is None else hyp
    names = CAM_T2 if names is None else names
    assoc = associate_images(lab, boxes, 1.0)
    d = lab.merge(assoc[K4 + ["camera_object_id", "u_ref", "v_ref", "w_ref", "cost", "margin"]], on=K4, how="left")
    d["assigned"] = d["camera_object_id"].notna()
    st = d[d["incl_static"] & d["assigned"]].copy()
    st["e"] = (st["p0_u"] - st["u_ref"]) * st["e_u"] + (st["p0_v"] - st["v_ref"]) * st["e_v"]
    st_all = st.copy()                                   # all assigned parked vehicles (S1 uses these)
    tables = zero_line_tables(st)
    st = st.join(lookup_zero_line(st[LEVELS[0]], tables))
    st = st[~st["no_zero_line"]].copy()
    st["r"] = st["e"] - st["z_b"]
    st["n_track"] = st.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
    st["w"] = 1.0 / (st["sigma_b"] ** 2 * st["n_track"])
    mo = d[d["incl_moving"] & d["assigned"]].copy()
    mo_assigned = mo.copy()                              # before the zero-line filter (TruckScenes camera-level route)
    zm = lookup_zero_line(mo[LEVELS[0]], tables)
    mo = mo.join(zm)
    n_no_zl = int(mo["no_zero_line"].sum())
    mo = mo[~mo["no_zero_line"] & (mo["sigma_b"] > 0)].copy()
    mo["n_track"] = mo.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
    mo["w"] = mo["sdot"] ** 2 / (mo["sigma_b"] ** 2 * mo["n_track"])
    mo["var_theta"] = mo["sigma_b"] ** 2 / mo["sdot"] ** 2
    cams, budget = {}, {}
    t0 = tables[0]
    for c, g in mo.groupby("camera"):
        name = names[int(c)]
        sc = st[st["camera"] == c]
        if len(sc) < 2:
            continue
        de = design_effect_mean(sc["r"], sc["w"], sc["sigma_b"] ** 2, sc["segment"])
        v = float((g["w"] ** 2 * g["var_theta"]).sum() / g["w"].sum() ** 2)
        G = int(g["segment"].nunique())
        two = {}
        for ft, gf in g.groupby("frame_type"):
            two[ft] = float((gf["w"] ** 2 * gf["var_theta"]).sum() / gf["w"].sum() ** 2)
        cams[name] = {"obs": int(len(g)), "tracks": int(g[["segment", "object_id"]].drop_duplicates().shape[0]),
                      "segments": G, "t_crit": t_crit(G), "DE": de, "V_ind": v,
                      "predicted_halfwidth_ms": predicted_halfwidth(de, v, G),
                      "predicted_halfwidth_two_frame_diff_ms": (predicted_halfwidth(de, two.get("A", np.nan) + two.get("B", np.nan), G)
                                                                if len(two) == 2 else float("nan")),
                      "predicted_theta_bar": {h: float((g["w"] * g[col]).sum() / g["w"].sum()) if g[col].notna().all()
                                              else float(np.average(g[col].dropna(), weights=g.loc[g[col].notna(), "w"]))
                                              if g[col].notna().any() else float("nan") for h, col in hyp.items()},
                      "obs_by_frame_type": {k: int(v2) for k, v2 in g["frame_type"].value_counts().items()}}
        sdot_med = float(g["sdot"].median())
        nb = np.nanmean(neighbour_spread(t0, g[LEVELS[0]]))
        es = np.nanmean(ego_spread(t0, g[LEVELS[0]]))
        x = np.column_stack([np.ones(len(sc)), sc["ego_speed"].to_numpy()])
        w = sc["w"].to_numpy()
        beta = np.linalg.lstsq(x.T @ (w[:, None] * x), x.T @ (w * sc["r"].to_numpy()), rcond=None)[0][1]
        dspeed = float(g["ego_speed"].median() - sc["ego_speed"].median())
        comp = {"neighbour_bins_ms": float(nb / sdot_med), "ego_class_spread_ms": float(es / sdot_med),
                "ego_speed_slope_ms": float(abs(beta * dspeed) / sdot_med)}
        budget[name] = dict(comp, B_c_ms=float(np.nanmax(list(comp.values()))))
    counts = {"moving_included": int(d["incl_moving"].sum()), "moving_assigned": int((d["incl_moving"] & d["assigned"]).sum()),
              "static_included": int(d["incl_static"].sum()), "static_assigned": int((d["incl_static"] & d["assigned"]).sum()),
              "static_used": int(len(st)), "moving_no_zero_line_excluded": n_no_zl, "moving_analysis_rows": int(len(mo)),
              "moving_levels": {int(k): int(v) for k, v in zm["zl_level"].value_counts().items()},
              "sigma_b_median_px": float(mo["sigma_b"].median()) if len(mo) else float("nan")}
    return {"assoc": assoc, "tables": tables, "st": st, "st_all": st_all, "mo": mo, "mo_assigned": mo_assigned,
            "cams": cams, "budget": budget, "counts": counts}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def save_descriptive_tables(res, odir):
    """Write the assigned parked rows (with e), the assigned moving rows before the zero-line filter (the inputs of
    the T2 section 11 camera-level zero line), and the zero-lined parked rows with r = e - z_b (the input of the
    per-log intercept secondary, section 10). Returns file name -> SHA-256."""
    out = {}
    for name, key in (("static_all.parquet", "st_all"), ("moving_assigned.parquet", "mo_assigned"),
                      ("static_zero_lined.parquet", "st")):
        p = os.path.join(odir, name)
        res[key].to_parquet(p, index=False)
        out[name] = sha(p)
    return out


def main(argv):
    if len(argv) not in (7, 8, 9):
        print(__doc__)
        return 64
    lab_path, raw_dir, score_min, nms_iou, summ_path, odir = argv[1], argv[2], float(argv[3]), float(argv[4]), argv[5], argv[6]
    dataset = argv[7] if len(argv) >= 8 else "av2"
    s1_rule = argv[8] if len(argv) == 9 else "written"
    if s1_rule not in S1_RULES:
        raise SystemExit(f"unknown S1 rule {s1_rule!r}")
    if dataset == "ts":
        from careport.xfc7_t2_ts_labels import TS_CAMERAS, TS_LIDARS
        hyp = {"sample": "theta_sample", **{l: f"theta_{l}" for l in TS_LIDARS}}
        names = {i: n for i, n in enumerate(TS_CAMERAS, 1)}
        contrasts = [("sample", l) for l in TS_LIDARS]
    else:
        hyp, names, contrasts = HYP_T2, CAM_T2, CONTRASTS
    os.makedirs(odir, exist_ok=True)
    lab = pd.read_parquet(lab_path)
    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.parquet")))
    raw = pd.concat([pd.read_parquet(f) for f in raw_files], ignore_index=True)
    raw = raw[raw["segment"].isin(set(lab["segment"]))]
    ref = reference_table(raw, score_min, nms_iou)
    ref_path = os.path.join(odir, "reference_table.parquet")
    ref.to_parquet(ref_path, index=False)
    res = run_reference_t2(lab, ref, hyp, names)
    res["assoc"].to_parquet(os.path.join(odir, "association.parquet"), index=False)
    m_path = os.path.join(odir, "moving_analysis_set_t2.parquet")
    res["mo"].to_parquet(m_path, index=False)
    desc = save_descriptive_tables(res, odir)
    s1_all = {r: s1_table(res["st_all"], names, r) for r in S1_RULES if r == "written" or "range_m" in res["st_all"]}
    s1 = s1_all[s1_rule]
    dec = {c: res["cams"][c] for c in decision_cameras(res["cams"], s1)}
    thr = {c: 5.0 + res["budget"][c]["B_c_ms"] + dec[c]["predicted_halfwidth_ms"] for c in dec}
    cap = capacity({c: dec[c]["predicted_theta_bar"] for c in dec}, thr, contrasts) if dec else {}
    out = {"protocol": "PILOT_PROTOCOL_XF-C7_T2_v0_1.md (draft), steps S1, S3, S4", "dataset": dataset,
           "score_min": score_min, "nms_iou": nms_iou,
           "inputs": {"label_side_sha256": sha(lab_path), "logs": int(lab["segment"].nunique()), "raw_files": len(raw_files),
                      "raw_detections_used": int(len(raw)), "reference_rows": int(len(ref)), "reference_table_sha256": sha(ref_path)},
           "S1_rule": s1_rule, "S1_projection_check": s1, "S1_by_rule": s1_all, "counts": res["counts"],
           "cameras": res["cams"], "budget": res["budget"],
           "cameras_with_30_tracks": sorted(c for c, v in res["cams"].items() if v["tracks"] >= MIN_DECISION_TRACKS),
           "decision_cameras": sorted(dec), "thresholds_ms": thr, "capacity": cap,
           "moving_analysis_set_sha256": sha(m_path), "descriptive_tables_sha256": desc}
    os.makedirs(os.path.dirname(os.path.abspath(summ_path)), exist_ok=True)
    json.dump(out, open(summ_path, "w", encoding="utf-8"), indent=2, default=float)
    print(json.dumps({k: out[k] for k in ("S1_projection_check", "decision_cameras", "thresholds_ms")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
