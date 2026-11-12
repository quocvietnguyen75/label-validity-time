"""XF-C7 G8(e): validate the association rule on pedestrians against Waymo's camera_to_lidar_box_association.

Usage: PYTHONPATH=src python -m careport.xfc7_g8e_run <ped_label_only.parquet> <output_dir>
Reads WAYMO_V2_ROOT. Reads camera_box rows of type 2 (pedestrian) only, and camera_to_lidar_box_association.
Metric (pre-stated in the pilot record, 2026-10-21): among Waymo pairs whose 3D box is in the image's pedestrian roster
and whose human box is in the camera_box pedestrian set, the fraction the rule assigns identically (gate >= 95 %);
also the share of the rule's assignments that Waymo's association does not contain.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

from careport.xfc7_association import associate_images

CB = "[CameraBoxComponent]"


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 64
    root = os.environ["WAYMO_V2_ROOT"]
    ped = pd.read_parquet(argv[1], columns=["segment", "frame", "camera", "object_id", "anchor_u", "anchor_v", "gate",
                                            "hull_w0", "hull_wmax"])
    vdir = os.path.join(root, "validation")
    boxes, assoc = [], []
    for f in sorted(glob.glob(os.path.join(vdir, "camera_box", "*.parquet"))):
        b = pq.read_table(f, columns=["key.segment_context_name", "key.frame_timestamp_micros", "key.camera_name",
                                      "key.camera_object_id", f"{CB}.box.center.x", f"{CB}.box.center.y", f"{CB}.box.size.x",
                                      f"{CB}.type"]).to_pandas()
        b = b[b[f"{CB}.type"] == 2]
        boxes.append(pd.DataFrame({"segment": b["key.segment_context_name"], "frame": b["key.frame_timestamp_micros"],
                                   "camera": b["key.camera_name"].astype(int), "camera_object_id": b["key.camera_object_id"],
                                   "u": b[f"{CB}.box.center.x"], "v": b[f"{CB}.box.center.y"], "w": b[f"{CB}.box.size.x"]}))
    for f in sorted(glob.glob(os.path.join(vdir, "camera_to_lidar_box_association", "*.parquet"))):
        a = pq.read_table(f).to_pandas()
        assoc.append(pd.DataFrame({"segment": a["key.segment_context_name"], "frame": a["key.frame_timestamp_micros"],
                                   "camera": a["key.camera_name"].astype(int), "camera_object_id": a["key.camera_object_id"],
                                   "object_id": a["key.laser_object_id"]}))
    boxes = pd.concat(boxes, ignore_index=True)
    assoc = pd.concat(assoc, ignore_index=True)
    ours = associate_images(ped, boxes, gate_scale=1.0)
    k = ["segment", "frame", "camera"]
    w = assoc.merge(ped[k + ["object_id"]], on=k + ["object_id"]).merge(boxes[k + ["camera_object_id"]], on=k + ["camera_object_id"])
    w = w.merge(ours[k + ["object_id", "camera_object_id"]].rename(columns={"camera_object_id": "ours"}), on=k + ["object_id"], how="left")
    agree = float((w["ours"] == w["camera_object_id"]).mean())
    assigned = ours.dropna(subset=["camera_object_id"])
    rev = assigned.merge(assoc.rename(columns={"camera_object_id": "waymo"}), on=k + ["object_id"], how="left")
    has_w = rev["waymo"].notna()
    res = {"waymo_pairs_eligible": int(len(w)), "agreement": agree, "gate_95pct": bool(agree >= 0.95),
           "our_assignments": int(len(assigned)), "our_assignments_with_waymo_pair": int(has_w.sum()),
           "our_assignments_disagreeing_with_waymo": int((rev.loc[has_w, "waymo"] != rev.loc[has_w, "camera_object_id"]).sum()),
           "our_assignments_without_waymo_pair": int((~has_w).sum()),
           "unassigned_among_eligible_waymo_pairs": int(w["ours"].isna().sum()),
           "roster_rows": int(len(ped)), "human_ped_boxes": int(len(boxes))}
    os.makedirs(argv[2], exist_ok=True)
    with open(os.path.join(argv[2], "g8e_pedestrian_association.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
