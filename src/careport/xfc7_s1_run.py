"""XF-C7 S1 mask instrument run (protocol v1.0 section 7; pilot-record specification of 2026-10-22).

Usage: PYTHONPATH=src python -m careport.xfc7_s1_run <vehicle_label_only.parquet> <g5_summary.json> <out_dir>
Links PVPS vehicle masks to human vehicle boxes (IoU >= 0.7, one-to-one), replaces u_ref with the mask centre on the
primary association, rebuilds a mask-specific static zero line and moving set, and reports per-camera delta with the
v2.0 rule as a consistency check. Writes <out_dir>/s1_masks.json (aggregates) and derived/s1/mask_links.parquet.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from careport.xfc7_g5_core import K4, build_sets
from careport.xfc7_g5_run import sha
from careport.xfc7_masks import decode_panoptic, instance_boxes, link_masks
from careport.xfc7_primary import analyse
from careport.xfc7_primary_v2_run import verdicts

CB, CS = "[CameraBoxComponent]", "[CameraSegmentationLabelComponent]"
IMG = ["segment", "frame", "camera"]


def human_boxes(vdir, images):
    out = []
    for f in sorted(glob.glob(os.path.join(vdir, "camera_box", "*.parquet"))):
        b = pq.read_table(f, columns=["key.segment_context_name", "key.frame_timestamp_micros", "key.camera_name", "key.camera_object_id",
                                      f"{CB}.box.center.x", f"{CB}.box.center.y", f"{CB}.box.size.x", f"{CB}.box.size.y",
                                      f"{CB}.type"]).to_pandas()
        b = b[b[f"{CB}.type"] == 1]
        cu, cv, w, h = (b[f"{CB}.box.center.x"], b[f"{CB}.box.center.y"], b[f"{CB}.box.size.x"], b[f"{CB}.box.size.y"])
        out.append(pd.DataFrame({"segment": b["key.segment_context_name"], "frame": b["key.frame_timestamp_micros"],
                                 "camera": b["key.camera_name"].astype(int), "camera_object_id": b["key.camera_object_id"],
                                 "u0": cu - w / 2, "v0": cv - h / 2, "u1": cu + w / 2, "v1": cv + h / 2}))
    hb = pd.concat(out, ignore_index=True)
    return hb.merge(images, on=IMG, how="inner")


def mask_links(vdir, hb):
    groups = {k: g for k, g in hb.groupby(IMG)}
    rows, n_img, n_masks = [], 0, 0
    for f in sorted(glob.glob(os.path.join(vdir, "camera_segmentation", "*.parquet"))):
        t = pq.read_table(f, columns=["key.segment_context_name", "key.frame_timestamp_micros", "key.camera_name",
                                      f"{CS}.panoptic_label_divisor", f"{CS}.panoptic_label"]).to_pandas()
        for seg, fr, cam, div, png in t.itertuples(index=False):
            n_img += 1
            sem, inst = decode_panoptic(png, int(div))
            mb = instance_boxes(sem, inst)
            n_masks += len(mb)
            g = groups.get((seg, int(fr), int(cam)))
            if g is None or len(mb) == 0:
                continue
            for i, j in link_masks(mb[["u0", "v0", "u1", "v1"]].to_numpy(float), g[["u0", "v0", "u1", "v1"]].to_numpy(float)):
                m = mb.iloc[i]
                rows.append((seg, int(fr), int(cam), g["camera_object_id"].iloc[j], float(m["mu"]), float(m["mv"]),
                             float(m["u1"] - m["u0"]), int(m["semantic"]), int(m["n_px"])))
    links = pd.DataFrame(rows, columns=IMG + ["camera_object_id", "mu", "mv", "mw", "semantic", "n_px"])
    return links, n_img, n_masks


def main(argv):
    lab_path, g5_path, out_dir = argv[1:4]
    g5 = json.load(open(g5_path, encoding="utf-8"))
    assert sha(lab_path) == g5["inputs"]["label_only_sha256"]
    root = os.environ["WAYMO_V2_ROOT"]
    vdir = os.path.join(root, "validation")
    lab = pd.read_parquet(lab_path)
    prim = pd.read_parquet(g5["association"]["table"])
    cams = [c for c, v in g5["cameras"].items() if v["tracks"] >= 30]
    budgets = {c: max(v["neighbour_bins_ms"], v["ego_class_spread_ms"], v["ego_speed_slope_ms"]) for c, v in g5["budget"].items()}
    images = []
    for f in sorted(glob.glob(os.path.join(vdir, "camera_segmentation", "*.parquet"))):
        t = pq.read_table(f, columns=["key.segment_context_name", "key.frame_timestamp_micros", "key.camera_name"]).to_pandas()
        images.append(pd.DataFrame({"segment": t.iloc[:, 0], "frame": t.iloc[:, 1], "camera": t.iloc[:, 2].astype(int)}))
    images = pd.concat(images, ignore_index=True).drop_duplicates()
    hb = human_boxes(vdir, images)
    links, n_img, n_masks = mask_links(vdir, hb)
    ddir = os.path.join(root, "derived", "s1")
    os.makedirs(ddir, exist_ok=True)
    links_path = os.path.join(ddir, "mask_links.parquet")
    links.to_parquet(links_path, index=False)
    am = prim.merge(links, on=IMG + ["camera_object_id"], how="inner")
    am["u_ref"], am["v_ref"], am["w_ref"] = am["mu"], am["mv"], am["mw"]
    st, mo, _ = build_sets(lab, am)
    r = analyse(mo, [], [])
    cam_rows = {}
    for c in cams:
        x = r["cameras"].get(c)
        cam_rows[c] = None if x is None else {
            "obs": x["obs"], "tracks": x["tracks"], "segments": x["segments"],
            "informative": bool(x["tracks"] >= 30 and x["segments"] >= 8),
            "delta": {h: {k: x["delta"][h][k] for k in ("est", "lo", "hi")} for h in x["delta"]}}
    out = {
        "inputs": {"label_only_sha256": sha(lab_path), "association_sha256": sha(g5["association"]["table"]),
                   "mask_links": links_path, "mask_links_sha256": sha(links_path)},
        "coverage": {"images": n_img, "vehicle_masks": n_masks, "human_vehicle_boxes_on_mask_images": int(len(hb)),
                     "linked": int(len(links)), "linked_share_of_human_boxes": float(len(links) / max(len(hb), 1)),
                     "static_rows": int(len(st)), "moving_rows": int(len(mo)),
                     "moving_tracks": int(mo[["segment", "object_id"]].drop_duplicates().shape[0]),
                     "moving_segments": int(mo["segment"].nunique())},
        "cameras": cam_rows,
        "verdicts": verdicts(r, cams, budgets),
    }
    with open(os.path.join(out_dir, "s1_masks.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out["coverage"], indent=1))
    for c, x in cam_rows.items():
        if x:
            print(c, x["obs"], x["tracks"], x["segments"], "inf" if x["informative"] else "UNINF",
                  " ".join(f"{h}:{x['delta'][h]['est']:.1f}[{x['delta'][h]['lo']:.1f},{x['delta'][h]['hi']:.1f}]" for h in x["delta"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
