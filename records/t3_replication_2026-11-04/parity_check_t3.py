"""XF-C7 T3: instrument parity, H100 cluster against the T2 RTX 5070 run, on the 50 T2 logs (SAMPLE_PIN addendum 4).

Reads detector boxes, parked-vehicle and weight-only static summaries, and label-side tables. Never forms a moving
theta-hat. Criteria (pinned in addendum 4 before any T3 log is processed):
  (a) reference boxes (score >= 0.5, class-agnostic NMS 0.7), greedy IoU >= 0.5 matching per (log, camera, frame):
      >= 99 % of each run's boxes matched, and >= 99 % of matched boxes with |du|, |dv|, |dw| <= 1 px;
  (b) the static step (careport.xfc7_t2_static, s050, far40) on T2's label side with the H100 raw tables: per T2
      decision camera, B_c and the predicted half-width within 1 ms, S1 (far40) median within 0.5 px; assigned parked
      and moving counts within 1 %;
  (c) label side: per-log tables identical, or every numeric column within 1e-6 relative and all else equal.
Usage: PYTHONPATH=src python parity_check_t3.py <t2_raw_dir> <h100_raw_dir> <t2_label_dir> <h100_label_dir>
       <t2_static_summary.json> <h100_static_summary.json> <out.json>"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

from careport.xfc7_ext_detect import class_agnostic_nms

SCORE, NMS, IOU_MATCH, PX, SHARE = 0.5, 0.7, 0.5, 1.0, 0.99


def ref_boxes(raw):
    out = []
    for key, g in raw[raw["score"] >= SCORE].groupby(["frame", "camera"], sort=True):
        b = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
        k = class_agnostic_nms(b, g["score"].to_numpy(float), NMS)
        out.append((key, b[k]))
    return dict(out)


def iou_matrix(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def greedy_match(a, b):
    if len(a) == 0 or len(b) == 0:
        return []
    m = iou_matrix(a, b)
    pairs, used_a, used_b = [], set(), set()
    for i, j in sorted(zip(*np.nonzero(m >= IOU_MATCH)), key=lambda ij: -m[ij]):
        if i not in used_a and j not in used_b:
            used_a.add(i); used_b.add(j); pairs.append((i, j))
    return pairs


def part_a(t2_dir, h_dir):
    n_t2 = n_h = n_m = within = 0
    worst = 0.0
    logs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(t2_dir, "*.parquet")))
    missing = [L for L in logs if not os.path.exists(os.path.join(h_dir, L))]
    for L in logs:
        if L in missing:
            continue
        A, B = ref_boxes(pd.read_parquet(os.path.join(t2_dir, L))), ref_boxes(pd.read_parquet(os.path.join(h_dir, L)))
        for key in set(A) | set(B):
            a, b = A.get(key, np.zeros((0, 4))), B.get(key, np.zeros((0, 4)))
            n_t2 += len(a); n_h += len(b)
            for i, j in greedy_match(a, b):
                n_m += 1
                d = np.abs(np.array([(a[i, 0] + a[i, 2] - b[j, 0] - b[j, 2]) / 2, (a[i, 1] + a[i, 3] - b[j, 1] - b[j, 3]) / 2,
                                     (a[i, 2] - a[i, 0]) - (b[j, 2] - b[j, 0])]))
                worst = max(worst, float(d.max()))
                within += int(d.max() <= PX)
    res = {"logs": len(logs), "logs_missing_on_h100": missing, "ref_boxes_t2": n_t2, "ref_boxes_h100": n_h, "matched": n_m,
           "matched_share_t2": n_m / n_t2 if n_t2 else None, "matched_share_h100": n_m / n_h if n_h else None,
           "within_1px_share": within / n_m if n_m else None, "max_abs_diff_px": worst}
    res["pass"] = bool(logs and not missing and n_m and res["matched_share_t2"] >= SHARE
                       and res["matched_share_h100"] >= SHARE and res["within_1px_share"] >= SHARE)
    return res


def part_b(t2_sum, h_sum):
    a, b = json.load(open(t2_sum, encoding="utf-8")), json.load(open(h_sum, encoding="utf-8"))
    rows, ok = {}, True
    for c in a["decision_cameras"]:
        r = {"B_c_t2": a["budget"][c]["B_c_ms"], "B_c_h100": b["budget"][c]["B_c_ms"],
             "hw_t2": a["cameras"][c]["predicted_halfwidth_ms"], "hw_h100": b["cameras"][c]["predicted_halfwidth_ms"],
             "S1_t2": a["S1_by_rule"]["far40"][c]["median_px"], "S1_h100": b["S1_by_rule"]["far40"][c]["median_px"]}
        r["pass"] = bool(abs(r["B_c_t2"] - r["B_c_h100"]) <= 1 and abs(r["hw_t2"] - r["hw_h100"]) <= 1
                         and abs(r["S1_t2"] - r["S1_h100"]) <= 0.5)
        ok &= r["pass"]
        rows[c] = r
    counts = {k: (a["counts"][k], b["counts"][k]) for k in ("moving_assigned", "static_assigned")}
    counts_ok = all(abs(x - y) <= 0.01 * x for x, y in counts.values())
    return {"cameras": rows, "counts": counts, "decision_cameras_h100": b["decision_cameras"],
            "thresholds_ms_h100": b["thresholds_ms"], "pass": bool(ok and counts_ok)}


def part_c(t2_dir, h_dir):
    logs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(t2_dir, "*.parquet")))
    identical = close = differ = 0
    bad, missing = [], []
    for L in logs:
        p, q = os.path.join(t2_dir, L), os.path.join(h_dir, L)
        if not os.path.exists(q):
            missing.append(L); continue
        if hashlib.sha256(open(p, "rb").read()).digest() == hashlib.sha256(open(q, "rb").read()).digest():
            identical += 1; continue
        x, y = pd.read_parquet(p), pd.read_parquet(q)
        same = list(x.columns) == list(y.columns) and len(x) == len(y)
        if same:
            for col in x.columns:
                if pd.api.types.is_numeric_dtype(x[col]) and not pd.api.types.is_bool_dtype(x[col]):
                    same &= bool(np.allclose(x[col].to_numpy(float), y[col].to_numpy(float), rtol=1e-6, atol=0, equal_nan=True))
                else:
                    same &= bool(x[col].equals(y[col]))
        if same:
            close += 1
        else:
            differ += 1; bad.append(L)
    return {"logs": len(logs), "identical_bytes": identical, "within_1e-6": close, "differ": differ,
            "differing_logs": bad, "missing_on_h100": missing, "pass": bool(differ == 0 and not missing)}


if __name__ == "__main__":
    t2_raw, h_raw, t2_lab, h_lab, t2_sum, h_sum, out = sys.argv[1:8]
    res = {"a_reference_boxes": part_a(t2_raw, h_raw), "b_static_step": part_b(t2_sum, h_sum),
           "c_label_side": part_c(t2_lab, h_lab)}
    res["pass"] = all(v["pass"] for v in res.values())
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))
