"""XF-C7 primary analysis under protocol v2.0 (rejection design), executed ONCE, with the pre-specified robustness runs.

Usage: PYTHONPATH=src python -m careport.xfc7_primary_v2_run <moving_analysis_set.parquet> <g5_summary.json>
       <protocol_v2.md> <expected_protocol_v2_sha256> <out_dir>
Verifies the analysis set against the G5 record and the frozen v2.0 protocol against its hash before computing anything.
Cameras: >= 30 moving tracks. B_c = max of the three pre-stated budget components (G5). Primary rule additive
(CI outside +-(5 + B_c)); verdict requires the margin-flag-excluded run to agree. Writes primary_results_v2.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from careport.xfc7_decision_v2 import combine_runs, decide_rejection
from careport.xfc7_estimators import wls_slope_cr1
from careport.xfc7_primary import CAM, HYP, add_tau_hat, analyse
from careport.xfc7_primary_run import _trim, jackknife_delta


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def verdicts(res, cams, budgets, rule="additive"):
    delta = {h: {c: (res["cameras"][c]["delta"][h]["est"], res["cameras"][c]["delta"][h]["lo"], res["cameras"][c]["delta"][h]["hi"])
                 for c in cams if c in res["cameras"]} for h in HYP}
    return decide_rejection(delta, budgets, [c for c in cams if c in res["cameras"]], rule=rule)


def main(argv):
    if len(argv) != 6:
        print(__doc__)
        return 64
    mo_path, g5_path, proto_path, proto_sha, out_dir = argv[1:6]
    g5 = json.load(open(g5_path, encoding="utf-8"))
    if sha(mo_path) != g5["moving_analysis_set"]["sha256"]:
        raise SystemExit("moving analysis set does not match the G5 record")
    if sha(proto_path) != proto_sha:
        raise SystemExit("frozen v2.0 protocol hash mismatch")
    cams = [c for c, v in g5["cameras"].items() if v["tracks"] >= 30]
    budgets = {c: max(v["neighbour_bins_ms"], v["ego_class_spread_ms"], v["ego_speed_slope_ms"]) for c, v in g5["budget"].items()}
    mo = pd.read_parquet(mo_path)
    res = analyse(mo, [], [])           # estimates and intervals; the v1.0 equivalence decision is stopped (no cameras)
    v2 = verdicts(res, cams, budgets)
    v2_pair = verdicts(res, cams, budgets, rule="pair")
    score = {h: float(sum(abs(res["cameras"][c]["delta"][h]["est"]) / budgets[c] for c in cams if c in res["cameras"])) for h in HYP}
    d = add_tau_hat(mo)
    d["r_px"] = d["sdot"] * d["tau_hat"]
    w_px = 1.0 / (d["sigma_b"] ** 2 * d["n_track"])
    dose = {}
    for h, col in HYP.items():
        if np.allclose(d[col].to_numpy(), 0.0):        # H_frame predicts tau = 0 everywhere: the regressor is degenerate
            dose[h] = {"slope": None, "se": None, "note": "undefined: prediction identically zero"}
        else:
            dose[h] = dict(zip(["slope", "se"], wls_slope_cr1(d["r_px"], d["sdot"] * d[col], w_px, np.zeros(len(d)), d["segment"])))
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
        r = analyse(sub, [], [])
        rob[name] = {"rows": int(len(sub)), "verdicts": verdicts(r, cams, budgets), "verdicts_pair": verdicts(r, cams, budgets, rule="pair"),
                     "delta": {c: {h: r["cameras"][c]["delta"][h] for h in HYP} for c in r["cameras"]}}
    final = combine_runs(v2, rob["margin_flag_excluded"]["verdicts"])
    out = {"run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "inputs": {"moving_analysis_set_sha256": g5["moving_analysis_set"]["sha256"], "protocol_v2_sha256": proto_sha,
                      "g5_summary_sha256": sha(g5_path), "rows": int(len(mo))},
           "cameras_in_analysis": cams, "budgets_ms": budgets, "final_verdicts": final, "verdicts_primary_run": v2, "verdicts_pair_rule_secondary": v2_pair,
           "best_single_convention_descriptive": {"scores_sum_abs_delta_over_B": score, "best": min(score, key=score.get)}, "estimates": res["cameras"],
           "slopes_descriptive": res["slopes"], "cell_slopes_display": res["cell_slopes_display"],
           "v1_equivalence_decision": res["decision"]["outcome"], "dose_response": dose, "robustness": rob,
           "tau_hat_summary": {"median": float(d["tau_hat"].median()), "p05": float(d["tau_hat"].quantile(0.05)),
                               "p95": float(d["tau_hat"].quantile(0.95))}}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "primary_results_v2.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    for h, v in v2.items():
        print(f"{h:6s} FINAL={final[h]:40s} primary rejected_in={v['rejected_in']} | pair rule rejected_in={v2_pair[h]['rejected_in']}")
    print("best single convention (descriptive):", min(score, key=score.get), {h: round(x, 2) for h, x in score.items()})
    for c, v in res["cameras"].items():
        tb = v["tau_bar"]
        print(f"{c:12s} n={v['obs']:5d} tracks={v['tracks']:4d} tau_bar={tb['est']:6.2f} [{tb['lo']:6.2f},{tb['hi']:6.2f}] B={budgets.get(c, float('nan')):5.1f} | "
              + " ".join(f"d_{h}={v['delta'][h]['est']:6.1f}[{v['delta'][h]['lo']:6.1f},{v['delta'][h]['hi']:6.1f}]" for h in HYP))
    for name in subsets:
        print(name, {h: rob[name]["verdicts"][h]["ruled_out"] for h in HYP})
    print("results:", path, "sha256", sha(path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
