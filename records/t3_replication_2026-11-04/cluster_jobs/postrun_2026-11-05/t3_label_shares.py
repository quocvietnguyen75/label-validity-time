"""Seam-rule and fallback shares of a label side (T3 protocol section 16; T2 section 5). Written after the run, test-first.

label_shares is copied verbatim from tier2_2026-11-01/diagnostics/t2_label_and_s1_tables.py, which gave T2's
figures; test_t3_label_shares checks that both give the same output. Label-side columns only: per camera, the
moving rows removed by the seam rule (among rows passing every other moving condition), and the share of included
moving rows whose theta_A or theta_B uses the azimuth fallback (< 5 points).

Usage (cluster, env311): python -X utf8 t3_label_shares.py LABEL_SIDE.parquet EXPECTED_SHA256 OUT.json
"""
import hashlib
import json
import sys

import pandas as pd

RING = ["ring_front_center", "ring_front_left", "ring_front_right", "ring_side_left", "ring_side_right",
        "ring_rear_left", "ring_rear_right"]
COLS = ["camera", "most_visible", "num_pts", "inside_all", "overlap", "static_proj_ok", "speed", "sdot",
        "along_axis_deg", "seam", "theta_fallback_A", "theta_fallback_B"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def label_shares(lab):
    out = {}
    pre = (lab["most_visible"] & (lab["num_pts"] >= 20) & lab["inside_all"] & ~lab["overlap"] & lab["static_proj_ok"]
           & lab["speed"].notna() & (lab["speed"] >= 3.0) & (lab["sdot"] >= 0.25) & (lab["along_axis_deg"] >= 20.0))
    for c, g in lab[pre].groupby("camera"):
        inc = g[~g["seam"]]
        out[RING[int(c) - 1]] = {"moving_before_seam_rule": int(len(g)), "removed_by_seam_rule": int(g["seam"].sum()),
                                 "share_removed": round(float(g["seam"].mean()), 3) if len(g) else None,
                                 "included": int(len(inc)),
                                 "share_theta_A_fallback": round(float(inc["theta_fallback_A"].mean()), 3) if len(inc) else None,
                                 "share_theta_B_fallback": round(float(inc["theta_fallback_B"].mean()), 3) if len(inc) else None}
    tot = lab[pre]
    out["_all"] = {"moving_before_seam_rule": int(len(tot)), "removed_by_seam_rule": int(tot["seam"].sum()),
                   "share_removed": round(float(tot["seam"].mean()), 3)}
    return out


def main(argv):
    if len(argv) != 4:
        raise SystemExit(__doc__)
    src, expected, out = argv[1:]
    got = sha(src)
    if got != expected:
        raise SystemExit(f"{src}: SHA-256 {got} is not the expected {expected}; stop")
    lab = pd.read_parquet(src, columns=COLS)
    res = {"input": src, "input_sha256": got, "rows": int(len(lab)), "label_side_shares": label_shares(lab)}
    with open(out, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    for k, v in res["label_side_shares"].items():
        print(k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
