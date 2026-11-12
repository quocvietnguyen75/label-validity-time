"""XF-C7 extension, tier 1, step E0 (EXT protocol v1.0 section 4): the static step for the detector reference.

Runs the frozen G5 procedure (association at gate scale 1.0, static zero line, weights, design effects, the three budget
components) with the detector reference table in place of Waymo's human boxes. It then sizes the paired comparison E2
from static data only. Like G5, it NEVER computes tau-hat, a residual or a displacement of a moving vehicle against
either reference: moving rows get association, zero line and weights only.

The same procedure is first rerun with the human boxes. Its moving analysis set must reproduce the frozen v2.0 set
(SHA-256 54016253...), which shows the procedure here is G5's.

Usage: PYTHONPATH=src python -m careport.xfc7_ext_e0 <label_only.parquet> <expected_sha256> <raw_det_dir> <score_min>
       <nms_iou> <summary.json>
Per-object tables go to $WAYMO_V2_ROOT/derived/ext_e0/s<score>/ (outside Git); the summary holds aggregates only."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

from careport.xfc7_association import associate_images
from careport.xfc7_budget import ego_spread, neighbour_spread
from careport.xfc7_estimators import design_effect_mean
from careport.xfc7_ext_detect import reference_table
from careport.xfc7_g5_run import CAM, HYP, load_boxes
from careport.xfc7_zero_line import LEVELS, lookup_zero_line, zero_line_tables

K4 = ["segment", "frame", "camera", "object_id"]
FROZEN_MOVING_SHA = "54016253c11a29c367dc0c2590fcbdf793d380e6a6529cafcb98e2ef6bbff32f"


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def paired_static(st_h: pd.DataFrame, st_d: pd.DataFrame) -> pd.DataFrame:
    """Static observations matched by both references: r_pair = r_human - r_det (px), its variance and weight."""
    m = st_h[K4 + ["r", "sigma_b"]].merge(st_d[K4 + ["r", "sigma_b"]], on=K4, suffixes=("_h", "_d"))
    m["r_pair"] = m["r_h"] - m["r_d"]
    m["var_pair"] = m["sigma_b_h"] ** 2 + m["sigma_b_d"] ** 2
    n_track = m.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
    m["w_pair"] = 1.0 / (m["var_pair"] * n_track)
    return m[K4 + ["r_pair", "var_pair", "w_pair"]]


def common_moving(mo_h: pd.DataFrame, mo_d: pd.DataFrame) -> pd.DataFrame:
    """Moving observations assigned by both references, with E2's weights; no reference position is carried."""
    m = mo_h[K4 + ["sdot", "sigma_b"]].merge(mo_d[K4 + ["sigma_b"]], on=K4, suffixes=("_h", "_d"))
    var = m["sigma_b_h"] ** 2 + m["sigma_b_d"] ** 2
    n_track = m.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
    m["w_pair"] = m["sdot"] ** 2 / (var * n_track)
    m["var_pair_ms2"] = var / m["sdot"] ** 2
    return m[K4 + ["sdot", "w_pair", "var_pair_ms2"]]


def predicted_pair_halfwidth(st_pair: pd.DataFrame, mv_pair: pd.DataFrame):
    """1.96 sqrt(DE x V_ind): DE from the static paired residuals by segment, V_ind from the common moving weights."""
    de = design_effect_mean(st_pair["r_pair"], st_pair["w_pair"], st_pair["var_pair"], st_pair["segment"])
    v = float((mv_pair["w_pair"] ** 2 * mv_pair["var_pair_ms2"]).sum() / mv_pair["w_pair"].sum() ** 2)
    return float(1.96 * np.sqrt(de * v)), float(de)


def run_reference(lab: pd.DataFrame, boxes: pd.DataFrame):
    """G5 steps 1, 3, 4 and 6 at gate scale 1.0 (xfc7_g5_run.main), for any reference table."""
    assoc = associate_images(lab, boxes, 1.0)
    d = lab.merge(assoc[K4 + ["camera_object_id", "u_ref", "v_ref", "w_ref", "cost", "margin"]], on=K4, how="left")
    d["assigned"] = d["camera_object_id"].notna()
    st = d[d["incl_static"] & d["assigned"]].copy()
    st["e"] = (st["p0_u"] - st["u_ref"]) * st["e_u"] + (st["p0_v"] - st["v_ref"]) * st["e_v"]
    tables = zero_line_tables(st)
    st = st.join(lookup_zero_line(st[LEVELS[0]], tables))
    st = st[~st["no_zero_line"]].copy()
    st["r"] = st["e"] - st["z_b"]
    st["n_track"] = st.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
    st["w"] = 1.0 / (st["sigma_b"] ** 2 * st["n_track"])
    mo = d[d["incl_moving"] & d["assigned"]].copy()
    zm = lookup_zero_line(mo[LEVELS[0]], tables)
    mo = mo.join(zm)
    n_no_zl = int(mo["no_zero_line"].sum())
    mo = mo[~mo["no_zero_line"] & (mo["sigma_b"] > 0)].copy()
    mo["n_track"] = mo.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
    mo["w"] = mo["sdot"] ** 2 / (mo["sigma_b"] ** 2 * mo["n_track"])
    mo["var_tau"] = mo["sigma_b"] ** 2 / mo["sdot"] ** 2
    cams, budget = {}, {}
    t0 = tables[0]
    for c, g in mo.groupby("camera"):
        sc = st[st["camera"] == c]
        de = design_effect_mean(sc["r"], sc["w"], sc["sigma_b"] ** 2, sc["segment"])
        v = float((g["w"] ** 2 * g["var_tau"]).sum() / g["w"].sum() ** 2)
        cams[CAM[c]] = {"obs": int(len(g)), "tracks": int(g[["segment", "object_id"]].drop_duplicates().shape[0]),
                        "segments": int(g["segment"].nunique()), "DE": de, "V_ind": v,
                        "predicted_halfwidth_ms": float(1.96 * np.sqrt(de * v)),
                        "predicted_tau_bar": {h: float((g["w"] * g[col]).sum() / g["w"].sum()) for h, col in HYP.items()}}
        sdot_med = float(g["sdot"].median())
        nb = np.nanmean(neighbour_spread(t0, g[LEVELS[0]]))
        es = np.nanmean(ego_spread(t0, g[LEVELS[0]]))
        x = np.column_stack([np.ones(len(sc)), sc["ego_speed"].to_numpy()])
        w = sc["w"].to_numpy()
        beta = np.linalg.solve(x.T @ (w[:, None] * x), x.T @ (w * sc["r"].to_numpy()))[1]
        dspeed = float(g["ego_speed"].median() - sc["ego_speed"].median())
        comp = {"neighbour_bins_ms": float(nb / sdot_med), "ego_class_spread_ms": float(es / sdot_med),
                "ego_speed_slope_ms": float(abs(beta * dspeed) / sdot_med)}
        budget[CAM[c]] = dict(comp, B_c_ms=max(comp.values()))
    counts = {"moving_included": int(d["incl_moving"].sum()), "moving_assigned": int((d["incl_moving"] & d["assigned"]).sum()),
              "static_included": int(d["incl_static"].sum()), "static_assigned": int((d["incl_static"] & d["assigned"]).sum()),
              "static_used": int(len(st)), "moving_no_zero_line_excluded": n_no_zl, "moving_analysis_rows": int(len(mo)),
              "moving_levels": {int(k): int(v) for k, v in zm["zl_level"].value_counts().items()},
              "sigma_b_median_px": float(mo["sigma_b"].median())}
    return {"assoc": assoc, "tables": tables, "st": st, "mo": mo, "cams": cams, "budget": budget, "counts": counts}


def main(argv):
    if len(argv) != 7:
        print(__doc__)
        return 64
    lab_path, expected, raw_dir, score_min, nms_iou, summ_path = argv[1], argv[2], argv[3], float(argv[4]), float(argv[5]), argv[6]
    got = sha(lab_path)
    if got != expected:
        raise SystemExit(f"label-only file hash mismatch: {got} != {expected}")
    root = os.environ["WAYMO_V2_ROOT"]
    odir = os.path.join(root, "derived", "ext_e0", f"s{int(round(score_min * 100)):03d}")
    os.makedirs(odir, exist_ok=True)
    lab = pd.read_parquet(lab_path)
    # 1. the human reference, rerun through this procedure, must reproduce the frozen v2.0 moving analysis set
    hum = run_reference(lab, load_boxes(os.path.join(root, "validation")))
    frozen = pd.read_parquet(os.path.join(root, "derived", "g5", "moving_analysis_set.parquet"))
    if sha(os.path.join(root, "derived", "g5", "moving_analysis_set.parquet")) != FROZEN_MOVING_SHA:
        raise SystemExit("frozen moving analysis set hash mismatch")
    a = hum["mo"][frozen.columns].sort_values(K4).reset_index(drop=True)
    b = frozen.sort_values(K4).reset_index(drop=True)
    same_keys = len(a) == len(b) and (a[K4].to_numpy() == b[K4].to_numpy()).all()
    same_vals = same_keys and all(np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), equal_nan=True)
                                  for c in ("z_b", "sigma_b", "w", "u_ref", "v_ref"))
    if not same_vals:
        raise SystemExit("human rerun does not reproduce the frozen moving analysis set; stop")
    # 2. the detector reference
    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.parquet")))
    raw = pd.concat([pd.read_parquet(f) for f in raw_files], ignore_index=True)
    ref = reference_table(raw, score_min, nms_iou)
    ref_path = os.path.join(odir, "reference_table.parquet")
    ref.to_parquet(ref_path, index=False)
    det = run_reference(lab, ref)
    det["assoc"].to_parquet(os.path.join(odir, "association.parquet"), index=False)
    pd.concat([t.assign(level=lev) for lev, t in det["tables"].items()], ignore_index=True).to_parquet(
        os.path.join(odir, "zero_line_tables.parquet"), index=False)
    m_path = os.path.join(odir, "moving_analysis_set_det.parquet")
    det["mo"][frozen.columns].to_parquet(m_path, index=False)
    # 3. E2 capacity from static data only
    st_pair = paired_static(hum["st"], det["st"])
    mv_pair = common_moving(hum["mo"], det["mo"])
    cap = {}
    for c, g in mv_pair.groupby("camera"):
        s = st_pair[st_pair["camera"] == c]
        hw, de = predicted_pair_halfwidth(s, g)
        cap[CAM[c]] = {"common_obs": int(len(g)), "common_tracks": int(g[["segment", "object_id"]].drop_duplicates().shape[0]),
                       "static_pairs": int(len(s)), "DE_pair": de, "predicted_halfwidth_ms": hw,
                       "agree_possible": bool(hw < 5.0)}
    res = {"protocol": "PILOT_PROTOCOL_XF-C7_EXT_v1_0.md (draft), step E0", "score_min": score_min, "nms_iou": nms_iou,
           "inputs": {"label_only_sha256": got, "raw_files": len(raw_files), "raw_detections": int(len(raw)),
                      "raw_manifest_sha256": sha(os.path.join(raw_dir, "run_manifest.json")),
                      "reference_rows": int(len(ref)), "reference_table_sha256": sha(ref_path)},
           "human_rerun_reproduces_frozen_set": True,
           "detector": {"counts": det["counts"], "cameras": det["cams"], "budget": det["budget"],
                        "moving_analysis_set_sha256": sha(m_path)},
           "human": {"counts": hum["counts"], "budget": hum["budget"]},
           "e2_capacity": cap}
    os.makedirs(os.path.dirname(summ_path), exist_ok=True)
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps({k: res[k] for k in ("inputs", "e2_capacity")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
