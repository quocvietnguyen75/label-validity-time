"""XF-C7 remaining pre-specified robustness runs (protocol v1.0 section 7, verdicts by the v2.0 rule).

Usage: PYTHONPATH=src python -m careport.xfc7_robustness_run <vehicle_label_only.parquet> <g5_summary.json> <out_dir>
Runs: association gate x0.5 and x2 (each re-associates and rebuilds its zero line and moving set); LEVEL_2 human boxes
excluded; box centre within 5 deg of the frustum edge excluded. Budgets B_c and the camera set are those of v2.0.
Before the runs, checks that build_sets on the G5 primary association reproduces G5's moving analysis set.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from careport.xfc7_association import associate_images
from careport.xfc7_g5_core import K4, build_sets
from careport.xfc7_g5_run import load_boxes, sha
from careport.xfc7_observations import frustum_margin_deg
from careport.xfc7_primary import analyse
from careport.xfc7_primary_v2_run import verdicts

CB, CC = "[CameraBoxComponent]", "[CameraCalibrationComponent]"


def main(argv):
    lab_path, g5_path, out_dir = argv[1:4]
    g5 = json.load(open(g5_path, encoding="utf-8"))
    assert sha(lab_path) == g5["inputs"]["label_only_sha256"]
    root = os.environ["WAYMO_V2_ROOT"]
    vdir = os.path.join(root, "validation")
    lab = pd.read_parquet(lab_path)
    boxes = load_boxes(vdir)
    cams = [c for c, v in g5["cameras"].items() if v["tracks"] >= 30]
    budgets = {c: max(v["neighbour_bins_ms"], v["ego_class_spread_ms"], v["ego_speed_slope_ms"]) for c, v in g5["budget"].items()}
    # reproduction check
    prim = pd.read_parquet(g5["association"]["table"])
    _, mo_p, _ = build_sets(lab, prim)
    ref = pd.read_parquet(g5["moving_analysis_set"]["file"])
    kk = K4
    a = mo_p.sort_values(kk)[kk + ["w", "z_b"]].reset_index(drop=True)
    b = ref.sort_values(kk)[kk + ["w", "z_b"]].reset_index(drop=True)
    reproduced = bool(len(a) == len(b) and np.allclose(a["w"], b["w"]) and np.allclose(a["z_b"], b["z_b"]) and (a[kk] == b[kk]).all().all())
    if not reproduced:
        raise SystemExit("build_sets does not reproduce the G5 moving analysis set")
    out = {"reproduction_check": reproduced, "runs": {}}
    for scale in (0.5, 2.0):
        assoc = associate_images(lab, boxes, scale)
        _, mo, _ = build_sets(lab, assoc)
        r = analyse(mo, [], [])
        out["runs"][f"gate_x{scale}"] = {"rows": int(len(mo)), "verdicts": verdicts(r, cams, budgets),
                                          "delta_lidar": {c: r["cameras"][c]["delta"]["lidar"] for c in r["cameras"]},
                                          "delta_frame": {c: r["cameras"][c]["delta"]["frame"] for c in r["cameras"]}}
    # LEVEL_2 exclusion (detection difficulty of the matched human box)
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
    sub = mo_d[mo_d["difficulty"] != 2].copy()
    sub["n_track"] = sub.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
    sub["w"] = sub["sdot"] ** 2 / (sub["sigma_b"] ** 2 * sub["n_track"])
    r = analyse(sub, [], [])
    out["runs"]["level2_excluded"] = {"rows": int(len(sub)), "level2_share": float((mo_d["difficulty"] == 2).mean()),
                                      "verdicts": verdicts(r, cams, budgets)}
    # frustum edge 5 deg
    cal = []
    for f in sorted(glob.glob(os.path.join(vdir, "camera_calibration", "*.parquet"))):
        t = pq.read_table(f).to_pandas()
        cal.append(pd.DataFrame({"segment": t["key.segment_context_name"], "camera": t["key.camera_name"].astype(int),
                                 "f_u": t[f"{CC}.intrinsic.f_u"], "f_v": t[f"{CC}.intrinsic.f_v"], "c_u": t[f"{CC}.intrinsic.c_u"],
                                 "c_v": t[f"{CC}.intrinsic.c_v"], "W": t[f"{CC}.width"], "H": t[f"{CC}.height"]}))
    cal = pd.concat(cal, ignore_index=True)
    mf = ref.merge(cal, on=["segment", "camera"], how="left")
    mf["margin_deg"] = frustum_margin_deg(mf["p0_u"], mf["p0_v"], mf["f_u"], mf["f_v"], mf["c_u"], mf["c_v"], mf["W"], mf["H"])
    sub = mf[mf["margin_deg"] >= 5.0].copy()
    sub["n_track"] = sub.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
    sub["w"] = sub["sdot"] ** 2 / (sub["sigma_b"] ** 2 * sub["n_track"])
    r = analyse(sub, [], [])
    out["runs"]["frustum_edge_5deg_excluded"] = {"rows": int(len(sub)), "verdicts": verdicts(r, cams, budgets)}
    with open(os.path.join(out_dir, "robustness_v2.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    for name, v in out["runs"].items():
        print(f"{name:28s} rows={v['rows']:5d} " + " ".join(f"{h}:{'OUT' if x['ruled_out'] else 'in '}{x['rejected_in']}" for h, x in v["verdicts"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
