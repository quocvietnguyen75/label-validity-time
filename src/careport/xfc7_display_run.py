"""XF-C7 POST HOC paper displays D2 and D3 (pilot record, 2026-10-24; pinned there before running). Displays only.

Usage: PYTHONPATH=src python -m careport.xfc7_display_run <moving_analysis_set.parquet> <association_primary.parquet>
       <vehicle_label_only.parquet> <primary_results_v2.json> <validation_dir> <out_json> <example_dir>
D2: weighted mean of tau-hat - tau^(H) for H_lidar and H_cam by camera x column third (the v1.0 §5 display) and by
camera x readout x column third (post hoc split), 95 % CR1 intervals by segment, no interval below 5 segments.
D3: the FRONT example chosen by the pinned rule. Its image crop and box coordinates go to <example_dir>, which must lie
outside Git with the Waymo data; <out_json> keeps aggregates, pool sizes and a hash of the example key only.
"""
from __future__ import annotations

import glob
import hashlib
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from careport.xfc7_g5_core import K4
from careport.xfc7_posthoc import cell_mean_ci, closest_row, crop_window, n_in_gate
from careport.xfc7_posthoc_run import time_of_day
from careport.xfc7_g5_run import load_boxes
from careport.xfc7_primary import CAM, RO, add_tau_hat

TESTED = [1, 2, 3, 4]
THIRD = {0: "left", 1: "middle", 2: "right"}
CB = "[CameraBoxComponent]"
CI = "[CameraImageComponent]"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def cell(g):
    r = {h: cell_mean_ci(g["tau_hat"] - g[col], g["w"], g["segment"]) for h, col in (("lidar", "tau_lidar"), ("cam", "tau_cap"))}
    w = g["w"].to_numpy(float)
    r["pred_cam_minus_lidar"] = float((w * (g["tau_cap"] - g["tau_lidar"])).sum() / w.sum())
    r["tracks"] = int(g[["segment", "object_id"]].drop_duplicates().shape[0])
    return r


def d2(d):
    pooled, split = {}, {}
    for c in TESTED:
        g = d[d["camera"] == c]
        pooled[CAM[c]] = {THIRD[t]: cell(x) for t, x in g.groupby("column_third")}
        split[CAM[c]] = {RO[int(r)]: {THIRD[t]: cell(y) for t, y in x.groupby("column_third")} for r, x in g.groupby("readout")}
    return {"pooled_prespecified": pooled, "by_readout_posthoc": split}


def human_box(vdir, seg, frame, cam, cid):
    t = pq.read_table(os.path.join(vdir, "camera_box", f"{seg}.parquet"),
                      columns=["key.frame_timestamp_micros", "key.camera_name", "key.camera_object_id", f"{CB}.box.center.x",
                               f"{CB}.box.center.y", f"{CB}.box.size.x", f"{CB}.box.size.y"]).to_pandas()
    r = t[(t["key.frame_timestamp_micros"] == frame) & (t["key.camera_name"] == cam) & (t["key.camera_object_id"] == cid)].iloc[0]
    cu, cv, sw, sh = (float(r[f"{CB}.box.{k}"]) for k in ("center.x", "center.y", "size.x", "size.y"))
    return [cu - sw / 2, cv - sh / 2, cu + sw / 2, cv + sh / 2]


def image(vdir, seg, frame, cam):
    from PIL import Image
    f = os.path.join(vdir, "camera_image", f"{seg}.parquet")
    t = pq.read_table(f, columns=["key.frame_timestamp_micros", "key.camera_name", f"{CI}.image"],
                      filters=[("key.frame_timestamp_micros", "=", frame), ("key.camera_name", "=", cam)]).to_pandas()
    return Image.open(io.BytesIO(t[f"{CI}.image"].iloc[0])).convert("RGB")


def d3(d, assoc, lab, vdir, target, example_dir):
    m = d[d["camera"] == 1].merge(lab[K4 + ["anchor_u", "anchor_v", "gate", "h0_u0", "h0_v0", "h0_u1", "h0_v1", "pl_u", "pl_v",
                                           "width", "height"]], on=K4, how="left")
    m = m.merge(assoc[K4 + ["camera_object_id"]], on=K4, how="left")
    boxes = load_boxes(vdir)
    boxes = boxes[boxes["camera"] == 1]
    bg = {k: g for k, g in boxes.groupby(["segment", "frame"])}
    cnt = np.zeros(len(m), dtype=int)
    for k, g in m.groupby(["segment", "frame"]):
        b = bg.get(k)
        bb = np.zeros((0, 2)) if b is None else b[["u", "v"]].to_numpy(float)
        cnt[m.index.get_indexer(g.index)] = n_in_gate(g[["anchor_u", "anchor_v"]].to_numpy(float), g["gate"].to_numpy(float), bb)
    m = m.merge(time_of_day(vdir), on="segment", how="left")
    pool = m[(cnt == 1) & (m["time_of_day"] == "Day") & (m["sdot"] >= 1.0) & (m["range_m"] >= 8.0) & (m["range_m"] <= 40.0)]
    pool = pool.reset_index(drop=True)
    tried = []
    while True:
        rest = pool.drop(index=tried)
        i = rest.index[closest_row(rest["tau_hat"], target, rest["segment"].to_numpy(), rest["frame"].to_numpy())]
        r = pool.loc[i]
        hb = human_box(vdir, r["segment"], int(r["frame"]), 1, r["camera_object_id"])
        if hb[3] - hb[1] >= 40.0:
            break
        tried.append(i)
    h0 = [r["h0_u0"], r["h0_v0"], r["h0_u1"], r["h0_v1"]]
    du, dv = r["pl_u"] - r["p0_u"], r["pl_v"] - r["p0_v"]
    hl = [h0[0] + du, h0[1] + dv, h0[2] + du, h0[3] + dv]
    win = crop_window(np.array([hb, h0, hl]), pad=0.25, min_width=160, width=int(r["width"]), height=int(r["height"]))
    img = image(vdir, r["segment"], int(r["frame"]), 1).crop(win)
    os.makedirs(example_dir, exist_ok=True)
    img.save(os.path.join(example_dir, "example_crop.png"))
    rel = lambda b: [b[0] - win[0], b[1] - win[1], b[2] - win[0], b[3] - win[1]]
    geo = {"crop_px": [win[2] - win[0], win[3] - win[1]], "human": rel(hb), "frame_start": rel(h0), "lidar_time": rel(hl),
           "tau_hat_ms": float(r["tau_hat"]), "tau_lidar_ms": float(r["tau_lidar"]), "speed_mps": float(r["speed"]),
           "sdot_px_per_ms": float(r["sdot"]), "range_m": float(r["range_m"]), "readout": RO[int(r["readout"])]}
    json.dump(geo | {"key": f"{r['segment']}|{int(r['frame'])}|1|{r['object_id']}"}, open(os.path.join(example_dir, "example.json"), "w"), indent=1)
    key_sha = hashlib.sha256(f"{r['segment']}|{int(r['frame'])}|1|{r['object_id']}".encode()).hexdigest()
    return {"pool_rows": int(len(pool)), "fallbacks": len(tried), "target_ms": target, "example_key_sha256": key_sha} | geo


def main(argv):
    if len(argv) != 8:
        print(__doc__)
        return 64
    mo_path, assoc_path, lab_path, prim_path, vdir, out_json, example_dir = argv[1:8]
    try:
        inside = os.path.commonpath([os.path.abspath(example_dir), os.path.abspath(os.getcwd())]) == os.path.abspath(os.getcwd())
    except ValueError:
        inside = False
    if inside:
        raise SystemExit("example_dir must lie outside the workspace (Waymo pixels never enter Git)")
    d = add_tau_hat(pd.read_parquet(mo_path))
    assoc = pd.read_parquet(assoc_path, columns=K4 + ["camera_object_id"])
    lab = pd.read_parquet(lab_path, columns=K4 + ["anchor_u", "anchor_v", "gate", "h0_u0", "h0_v0", "h0_u1", "h0_v1", "pl_u", "pl_v",
                                              "width", "height"])
    prim = json.load(open(prim_path, encoding="utf-8"))
    target = float(prim["estimates"]["FRONT"]["delta"]["frame"]["est"])
    out = {"note": "POST HOC displays D2-D3 (pilot record 2026-10-24); displays only, no decision",
           "inputs_sha256": {"moving_analysis_set": sha(mo_path), "association_primary": sha(assoc_path),
                             "label_only": sha(lab_path), "primary_results_v2": sha(prim_path)},
           "D2": d2(d), "D3": d3(d, assoc, lab, vdir, target, example_dir)}
    json.dump(out, open(out_json, "w", encoding="utf-8"), indent=1)
    for c, x in out["D2"]["by_readout_posthoc"].items():
        for ro, y in x.items():
            print(c, ro, {t: (round(z["lidar"]["est"], 1), round(z["pred_cam_minus_lidar"], 1), z["lidar"]["clusters"]) for t, z in y.items()})
    print("D3", {k: out["D3"][k] for k in ("pool_rows", "fallbacks", "tau_hat_ms", "speed_mps", "sdot_px_per_ms", "range_m", "readout", "crop_px")})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
