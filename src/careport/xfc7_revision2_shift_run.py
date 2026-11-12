"""XF-C7 revision-2 POST HOC K4: injected frame-start placement through the whole pipeline (pilot record, 2026-10-25).

Usage: PYTHONPATH=src python -m careport.xfc7_revision2_shift_run <vehicle_label_only.parquet> <g5_summary.json> <out_dir>
As B4 (xfc7_revision_shift_run), written to a new file. Every human box matched to an included moving observation is moved
by -delta_i * sdot_i along d_hat_i; runs: delta_i = -t_lidar,i (the reference placed where frame start would put it if the
boxes are now LiDAR-time-valid), and uniform delta = -50 and +50 ms. The association (gate x1), zero line and moving set
are rebuilt; parked boxes are not shifted. Reports assignment changes, per-camera delta_frame and delta_lidar, the v2.0
verdicts in the primary and margin runs, and the recovered shift. Writes <out_dir>/revision2_shift_v2.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from careport.xfc7_association import associate_images
from careport.xfc7_decision_v2 import combine_runs
from careport.xfc7_g5_core import K4, build_sets
from careport.xfc7_g5_run import load_boxes, sha
from careport.xfc7_primary import HYP, analyse
from careport.xfc7_primary_v2_run import verdicts
from careport.xfc7_revision2 import shift_each

IMG = ["segment", "frame", "camera", "camera_object_id"]


def reweight(mo):
    mo = mo.copy()
    mo["n_track"] = mo.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
    mo["w"] = mo["sdot"] ** 2 / (mo["sigma_b"] ** 2 * mo["n_track"])
    return mo


def main(argv):
    lab_path, g5_path, out_dir = argv[1:4]
    g5 = json.load(open(g5_path, encoding="utf-8"))
    assert sha(lab_path) == g5["inputs"]["label_only_sha256"]
    vdir = os.path.join(os.environ["WAYMO_V2_ROOT"], "validation")
    lab = pd.read_parquet(lab_path)
    prim = pd.read_parquet(g5["association"]["table"])
    ref = pd.read_parquet(g5["moving_analysis_set"]["file"])
    cams = [c for c, v in g5["cameras"].items() if v["tracks"] >= 30]
    bud = {c: max(v["neighbour_bins_ms"], v["ego_class_spread_ms"], v["ego_speed_slope_ms"]) for c, v in g5["budget"].items()}
    boxes = load_boxes(vdir)
    base = analyse(ref, [], [])
    matched = ref[K4 + ["sdot", "d_u", "d_v", "tau_lidar"]].merge(prim[K4 + ["camera_object_id"]], on=K4, how="left")
    matched = matched[matched["camera_object_id"].notna()]
    out = {"inputs": {"label_only_sha256": g5["inputs"]["label_only_sha256"], "moving_analysis_set_sha256": g5["moving_analysis_set"]["sha256"],
                      "association_table_sha256": sha(g5["association"]["table"])},
           "pinned_record_sha256_at_pin": "87facba22d8df50a354ff105ae41bef646aeb8ae4a4d511648b6925d590d1cc6",
           "shifted_boxes": int(len(matched)), "runs": {}}
    runs = {"frame_start_placement": -matched["tau_lidar"].to_numpy(float), "-50ms": np.full(len(matched), -50.0),
            "+50ms": np.full(len(matched), 50.0)}
    for name, dlt in runs.items():
        uv = shift_each(np.zeros((len(matched), 2)), matched[["d_u", "d_v"]].to_numpy(float), matched["sdot"].to_numpy(float), dlt)
        sh = matched[IMG].assign(du=uv[:, 0], dv=uv[:, 1]).drop_duplicates(IMG)
        b = boxes.merge(sh, on=IMG, how="left")
        b["u"] = b["u"] + b["du"].fillna(0.0)
        b["v"] = b["v"] + b["dv"].fillna(0.0)
        assoc = associate_images(lab, b.drop(columns=["du", "dv"]), 1.0)
        _, mo, _ = build_sets(lab, assoc)
        r = analyse(mo, [], [])
        rm = analyse(reweight(mo[(mo["margin"] >= 0.2) | np.isinf(mo["margin"])]), [], [])
        chk = ref[K4].merge(prim[K4 + ["camera_object_id"]], on=K4, how="left").merge(
            assoc[K4 + ["camera_object_id"]].rename(columns={"camera_object_id": "cid_new"}), on=K4, how="left")
        changed = float((chk["camera_object_id"].fillna("") != chk["cid_new"].fillna("")).mean())
        v1, v2 = verdicts(r, cams, bud), verdicts(rm, cams, bud)
        run = {"rows": int(len(mo)), "assignment_changed_share": changed, "verdicts_primary": v1, "verdicts_margin": v2,
               "final": combine_runs(v1, v2), "cameras": {}}
        for c, x in r["cameras"].items():
            if c not in base["cameras"]:
                continue
            d0, d1 = base["cameras"][c]["delta"]["lidar"]["est"], x["delta"]["lidar"]["est"]
            run["cameras"][c] = {"tracks": x["tracks"], "delta": {h: x["delta"][h] for h in HYP}, "recovered_shift_ms": d1 - d0,
                                 "recovery_ratio": (d1 - d0) / float(dlt[0]) if name != "frame_start_placement" else None}
        out["runs"][name] = run
        print(name, "rows", len(mo), "changed", round(changed, 4), run["final"], flush=True)
    path = os.path.join(out_dir, "revision2_shift_v2.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print("written", path, sha(path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
