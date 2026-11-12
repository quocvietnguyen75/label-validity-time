"""XF-C7 revision-round POST HOC B4: known time shifts injected into the human reference (pilot record, 2026-10-23).

Usage: PYTHONPATH=src python -m careport.xfc7_revision_shift_run <vehicle_label_only.parquet> <g5_summary.json> <out_dir> [deltas...]
For each delta (ms), every human box matched to an included moving observation in the primary association is moved by
-delta * sdot along d_hat (so tau-hat should rise by delta), the association is re-run (gate x1), the zero line and the
moving set are rebuilt, and per-camera delta_lidar is re-estimated. Reports the recovered shift and the share of included
observations whose assignment changed. Parked boxes are not shifted. Writes <out_dir>/revision_shift_v2.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from careport.xfc7_association import associate_images
from careport.xfc7_g5_core import K4, build_sets
from careport.xfc7_g5_run import load_boxes, sha
from careport.xfc7_posthoc import shift_along
from careport.xfc7_primary import analyse

IMG = ["segment", "frame", "camera", "camera_object_id"]


def main(argv):
    lab_path, g5_path, out_dir = argv[1:4]
    deltas = [float(x) for x in argv[4:]] or [-20.0, -10.0, 10.0, 20.0]
    g5 = json.load(open(g5_path, encoding="utf-8"))
    assert sha(lab_path) == g5["inputs"]["label_only_sha256"]
    vdir = os.path.join(os.environ["WAYMO_V2_ROOT"], "validation")
    lab = pd.read_parquet(lab_path)
    prim = pd.read_parquet(g5["association"]["table"])
    ref = pd.read_parquet(g5["moving_analysis_set"]["file"])
    boxes = load_boxes(vdir)
    base = analyse(ref, [], [])
    matched = ref[K4 + ["sdot", "d_u", "d_v"]].merge(prim[K4 + ["camera_object_id"]], on=K4, how="left")
    matched = matched[matched["camera_object_id"].notna()]
    out = {"inputs": {"label_only_sha256": g5["inputs"]["label_only_sha256"], "moving_analysis_set_sha256": g5["moving_analysis_set"]["sha256"]},
           "shifted_boxes": int(len(matched)), "runs": {}}
    for dlt in deltas:
        uv = shift_along(np.zeros((len(matched), 2)), matched[["d_u", "d_v"]].to_numpy(float), matched["sdot"].to_numpy(float), dlt)
        sh = matched[IMG].assign(du=uv[:, 0], dv=uv[:, 1]).drop_duplicates(IMG)
        b = boxes.merge(sh, on=IMG, how="left")
        b["u"] = b["u"] + b["du"].fillna(0.0)
        b["v"] = b["v"] + b["dv"].fillna(0.0)
        b = b.drop(columns=["du", "dv"])
        assoc = associate_images(lab, b, 1.0)
        _, mo, _ = build_sets(lab, assoc)
        r = analyse(mo, [], [])
        chk = ref[K4].merge(prim[K4 + ["camera_object_id"]], on=K4, how="left").merge(
            assoc[K4 + ["camera_object_id"]].rename(columns={"camera_object_id": "cid_new"}), on=K4, how="left")
        changed = float((chk["camera_object_id"].fillna("") != chk["cid_new"].fillna("")).mean())
        run = {"rows": int(len(mo)), "assignment_changed_share": changed, "cameras": {}}
        for c, x in r["cameras"].items():
            if c not in base["cameras"]:
                continue
            d0, d1 = base["cameras"][c]["delta"]["lidar"]["est"], x["delta"]["lidar"]["est"]
            run["cameras"][c] = {"tracks": x["tracks"], "delta_lidar": x["delta"]["lidar"], "recovered_shift_ms": d1 - d0,
                                 "recovery_ratio": (d1 - d0) / dlt}
        out["runs"][f"{dlt:+.0f}ms"] = run
        print(f"delta {dlt:+.0f} ms: rows {len(mo)}, assignment changed {changed:.4f}, recovered",
              {c: round(v["recovered_shift_ms"], 2) for c, v in run["cameras"].items()})
    with open(os.path.join(out_dir, "revision_shift_v2.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
