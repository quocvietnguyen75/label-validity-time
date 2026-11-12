"""XF-C7 T3, SAMPLE_PIN addendum 7, cross-platform check 1: the frozen static step run on the cluster with T2's PC inputs
against T2's frozen PC outputs. Criteria (pinned before the run):
  - decision cameras and all counts identical;
  - every B_c, hw, S1 median (far40) and predicted theta-bar within 1e-6 relative;
  - moving_analysis_set_t2.parquet and static_zero_lined.parquet: same rows and columns, numeric columns within
    rtol 1e-9 plus atol 1e-9, all else equal (the other four tables are compared and reported, not judged).
Usage: python xplat_check.py <frozen_summary.json> <cluster_summary.json> <frozen_dir> <cluster_dir> <out.json>"""
import json
import os
import sys

import numpy as np
import pandas as pd

REL, RTOL, ATOL = 1e-6, 1e-9, 1e-9
JUDGED = ("moving_analysis_set_t2.parquet", "static_zero_lined.parquet")
TABLES = JUDGED + ("reference_table.parquet", "association.parquet", "static_all.parquet", "moving_assigned.parquet")


def rel_ok(a, b):
    a, b = float(a), float(b)
    if np.isnan(a) and np.isnan(b):
        return True
    return abs(a - b) <= REL * max(abs(a), abs(b)) or a == b


def summary_check(a, b):
    bad = []
    if a["decision_cameras"] != b["decision_cameras"]:
        bad.append("decision_cameras")
    for k, v in a["counts"].items():
        w = b["counts"].get(k)
        ok = rel_ok(v, w) if isinstance(v, float) else (v == w)
        if not ok:
            bad.append(f"counts.{k}")
    for c, x in a["cameras"].items():
        y = b["cameras"].get(c, {})
        for k in ("obs", "tracks", "segments"):
            if x[k] != y.get(k):
                bad.append(f"{c}.{k}")
        if not rel_ok(x["predicted_halfwidth_ms"], y.get("predicted_halfwidth_ms", np.nan)):
            bad.append(f"{c}.hw")
        for h, t in x["predicted_theta_bar"].items():
            if not rel_ok(t, y.get("predicted_theta_bar", {}).get(h, np.nan)):
                bad.append(f"{c}.theta_bar.{h}")
    for c, x in a["budget"].items():
        if not rel_ok(x["B_c_ms"], b["budget"].get(c, {}).get("B_c_ms", np.nan)):
            bad.append(f"{c}.B_c")
    for c, x in a["S1_by_rule"]["far40"].items():
        if not rel_ok(x["median_px"], b["S1_by_rule"]["far40"].get(c, {}).get("median_px", np.nan)):
            bad.append(f"{c}.S1")
    return {"pass": not bad, "failures": bad}


def table_check(pa, pb):
    x, y = pd.read_parquet(pa), pd.read_parquet(pb)
    res = {"rows": [len(x), len(y)], "columns_equal": list(x.columns) == list(y.columns)}
    if not res["columns_equal"] or len(x) != len(y):
        return dict(res, pass_=False, differing_columns=None)
    keys = [k for k in ("segment", "frame", "camera", "object_id", "camera_object_id") if k in x.columns]
    x = x.sort_values(keys, kind="mergesort").reset_index(drop=True)
    y = y.sort_values(keys, kind="mergesort").reset_index(drop=True)
    diff, maxabs = [], {}
    for c in x.columns:
        if pd.api.types.is_numeric_dtype(x[c]) and not pd.api.types.is_bool_dtype(x[c]):
            a, b = x[c].to_numpy(float), y[c].to_numpy(float)
            close = np.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True)
            fin = np.isfinite(a) & np.isfinite(b)
            maxabs[c] = float(np.abs(a[fin] - b[fin]).max()) if fin.any() else 0.0
            if not close.all():
                diff.append(c)
        elif not x[c].equals(y[c]):
            diff.append(c)
    return dict(res, pass_=not diff, differing_columns=diff,
                max_abs_diff={k: v for k, v in maxabs.items() if v > 0})


def main(argv):
    fa, fb, da, db, out = argv[1:6]
    a, b = json.load(open(fa, encoding="utf-8")), json.load(open(fb, encoding="utf-8"))
    res = {"summary": summary_check(a, b), "tables": {}}
    for t in TABLES:
        res["tables"][t] = table_check(os.path.join(da, t), os.path.join(db, t))
    res["pass"] = bool(res["summary"]["pass"] and all(res["tables"][t]["pass_"] for t in JUDGED))
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
