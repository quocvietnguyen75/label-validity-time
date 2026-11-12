"""XF-C7 primary analysis run (protocol v1.0 section 5), executed ONCE, with the cheap pre-specified robustness runs.

Usage: PYTHONPATH=src python -m careport.xfc7_primary_run <moving_analysis_set.parquet> <g5_summary.json> <protocol.md> <out_dir>
Checks that the moving analysis set and the frozen protocol match their recorded SHA-256 before computing anything.
Writes primary_results.json (aggregates only).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from careport.xfc7_estimators import weighted_mean_cr1, wls_slope_cr1
from careport.xfc7_primary import CAM, HYP, add_tau_hat, analyse

PROTOCOL_SHA256 = "1ddc751a241509f0c38a28244506aa639e024043b4e482e6679950e7ddeadd6d"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _trim(x, frac=0.10):
    x = np.sort(np.asarray(x, dtype=float))
    k = int(np.floor(frac * len(x)))
    return float(x[k:len(x) - k].mean()) if len(x) > 2 * k else float("nan")


def jackknife_delta(d, h_col):
    segs = d["segment"].unique()
    full = weighted_mean_cr1(d["tau_hat"] - d[h_col], d["w"], d["segment"])[0]
    reps = []
    for s in segs:
        g = d[d["segment"] != s]
        reps.append(weighted_mean_cr1(g["tau_hat"] - g[h_col], g["w"], g["segment"])[0])
    reps = np.asarray(reps)
    n = len(segs)
    return float(full), float(np.sqrt((n - 1) / n * ((reps - reps.mean()) ** 2).sum()))


def main(argv):
    if len(argv) != 5:
        print(__doc__)
        return 64
    mo_path, g5_path, proto_path, out_dir = argv[1:5]
    g5 = json.load(open(g5_path, encoding="utf-8"))
    if sha(mo_path) != g5["moving_analysis_set"]["sha256"]:
        raise SystemExit("moving analysis set does not match the G5 record")
    if sha(proto_path) != PROTOCOL_SHA256:
        raise SystemExit("frozen protocol hash mismatch")
    mo = pd.read_parquet(mo_path)
    dc, sc = g5["decision_cameras"], g5["eligible_slope_cells"]
    primary = analyse(mo, dc, sc)
    d = add_tau_hat(mo)
    # dose-response display: r = sdot * tau_hat regressed on sdot * tau_H (common intercept), weights 1/(sigma_b^2 n_track)
    d["r_px"] = d["sdot"] * d["tau_hat"]
    w_px = 1.0 / (d["sigma_b"] ** 2 * d["n_track"])
    dose = {h: dict(zip(["slope", "se"], wls_slope_cr1(d["r_px"], d["sdot"] * d[col], w_px, np.zeros(len(d)), d["segment"])))
            for h, col in HYP.items()}
    # robustness (secondary, pre-specified in section 7)
    rob = {"median_and_trimmed": {}, "jackknife_delta_se": {}}
    for c, g in d.groupby("camera"):
        name = CAM[int(c)]
        rob["median_and_trimmed"][name] = {h: {"median": float(np.median(g["tau_hat"] - g[col])), "trimmed10": _trim(g["tau_hat"] - g[col])}
                                           for h, col in HYP.items()}
        rob["jackknife_delta_se"][name] = {h: jackknife_delta(g, col)[1] for h, col in HYP.items()}
    subsets = {"margin_flag_excluded": mo[(mo["margin"] >= 0.2) | np.isinf(mo["margin"])],
               "top_to_total_ge_0_8": mo[mo["top_points"] >= 0.8 * mo["all_points"]],
               "zero_line_flagged_excluded": mo[~mo["zl_flagged"]]}
    for name, sub in subsets.items():
        sub = sub.copy()
        sub["n_track"] = sub.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
        sub["w"] = sub["sdot"] ** 2 / (sub["sigma_b"] ** 2 * sub["n_track"])
        r = analyse(sub, dc, sc)
        rob[name] = {"rows": int(len(sub)), "decision": r["decision"]["outcome"],
                     "delta_lidar": {k: v["delta"]["lidar"] for k, v in r["cameras"].items()},
                     "slopes_lidar": {k: v["lidar"] for k, v in r["slopes"].items() if isinstance(v, dict) and "lidar" in v}}
    res = {"run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "inputs": {"moving_analysis_set_sha256": g5["moving_analysis_set"]["sha256"], "protocol_sha256": PROTOCOL_SHA256,
                      "g5_summary_sha256": sha(g5_path), "rows": int(len(mo))},
           "primary": primary, "budget_ms": g5["budget"], "dose_response": dose, "robustness": rob,
           "tau_hat_summary": {"median": float(d["tau_hat"].median()), "p05": float(d["tau_hat"].quantile(0.05)),
                               "p95": float(d["tau_hat"].quantile(0.95))}}
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "primary_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    print("decision:", primary["decision"]["outcome"])
    for c, v in primary["cameras"].items():
        dl = v["delta"]
        print(f"{c:12s} n={v['obs']:5d} tracks={v['tracks']:4d} tau_bar={v['tau_bar']['est']:6.2f} [{v['tau_bar']['lo']:6.2f},{v['tau_bar']['hi']:6.2f}] | "
              + " ".join(f"d_{h}={dl[h]['est']:6.2f}[{dl[h]['lo']:6.2f},{dl[h]['hi']:6.2f}]" for h in HYP))
    for r, v in primary["slopes"].items():
        print(f"slope {r}: " + " ".join(f"{h}={v[h]['est']:6.2f}[{v[h]['lo']:6.2f},{v[h]['hi']:6.2f}]" for h in HYP))
    print("results:", out, "sha256", sha(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
