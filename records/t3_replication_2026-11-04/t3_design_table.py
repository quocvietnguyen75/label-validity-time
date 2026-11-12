"""XF-C7 T3, section 9 before the freeze: the weights-only design table and the hw_BM gate.

Reads from the moving analysis set ONLY the keys, margin, sdot, sigma_b, w and the label-side predictions theta_start,
theta_A, theta_B and theta_mid (never p0, u_ref, d-hat, s or z_b, so no theta-hat can be formed), and from the S3 summary
the budgets, design effects and label-side predicted theta-bar. Writes, per decision camera and for the primary and
margin-flag-excluded runs: tracks, logs, the Bell-McCaffrey df, Kish effective logs, top-log and top-five-track weight
shares, the predicted theta-bar under that run's weights, hw with t(G - 1) and with t(nu_BM), the thresholds and the
capacity table under each, plus the root-sum-square and summed compositions (S-a). The primary run's gate reads the S3
summary's theta-bar, after a check that the recomputed one agrees with it; the margin-flag-excluded run's gate reads its
own, as T2's t2_margin_capacity.py did (changed 4 Nov 2026 after the code audit, before any T3 design value existed).
Usage: PYTHONPATH=src python t3_design_table.py <S3_summary.json> <moving_analysis_set.parquet> <out.json>"""
import hashlib
import json
import sys

import pandas as pd

from careport.xfc7_t2_posthoc import composition_budgets
from careport.xfc7_t2_static import CAM_T2, HYP_T2
from careport.xfc7_t3_run import RUNS, design_table, gates_both

COLUMNS = ["segment", "object_id", "camera", "frame", "frame_type", "margin", "sdot", "sigma_b", "w"] + list(HYP_T2.values())
THETA_TOL_MS = 1e-6


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main(argv):
    s3_path, mo_path, out_path = argv[1:4]
    s3 = json.load(open(s3_path, encoding="utf-8"))
    mo = pd.read_parquet(mo_path, columns=COLUMNS)
    cams = list(s3["decision_cameras"])
    des = design_table(mo, CAM_T2, cams)
    worst = 0.0
    for c in cams:
        if c not in des["primary"]:
            continue
        for h in HYP_T2:
            want, got = float(s3["cameras"][c]["predicted_theta_bar"][h]), des["primary"][c]["theta_bar"][h]
            if pd.isna(want) and pd.isna(got):
                continue
            diff = abs(got - want)
            if not diff <= THETA_TOL_MS:                   # also catches NaN on one side only
                raise SystemExit(f"{c} {h}: recomputed primary theta-bar {got} differs from the S3 summary's {want}; stop")
            worst = max(worst, diff)
    comps = composition_budgets({c: s3["budget"][c] for c in cams})
    out = {"inputs": {"s3_summary_sha256": sha(s3_path), "moving_set_sha256": sha(mo_path), "columns_read": COLUMNS},
           "decision_cameras": cams, "design": des, "primary_theta_bar_check_max_abs_ms": worst,
           "gates": {tag: {} for tag in RUNS}}
    for name in ("max", "rss", "sum"):
        g = gates_both(s3, des, cams, {c: comps[c][name] for c in cams})
        for tag in RUNS:
            out["gates"][tag][name] = g[tag]
    out["gates"]["primary"]["t_G_minus_1_thresholds_ms"] = {c: float(s3["thresholds_ms"][c]) for c in cams if c in des["primary"]}
    out["gates"]["margin_flag_excluded"]["t_G_minus_1_thresholds_ms"] = None
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=1, default=float)
    print(json.dumps({"decision_cameras": cams,
                      "thresholds_bm_max": {c: round(x, 2) for c, x in out["gates"]["primary"]["max"]["thresholds_ms"].items()},
                      "decidable_bm_max": out["gates"]["primary"]["max"]["capacity"]["dataset"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
