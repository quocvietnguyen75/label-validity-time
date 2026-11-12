"""Pre-freeze check (1 Nov 2026; T2 audit finding 4): the capacity table of the margin-flag-excluded run.

The run drops moving rows with association margin < 0.2 (keeping +inf) and recomputes n_track and the weights
(`xfc7_t2_run.verdicts_t2`). This script does the same on the pinned moving sets, reading only keys, margin, sdot,
sigma_b and the label-side theta predictions; DE and B_c come from the S3 summary. It computes no theta-hat, residual
or displacement. Output: per score and decision camera, rows kept, logs, hw, threshold 5 + B + hw, the predicted
theta-bar per hypothesis, and the capacity flags; then the dataset-level decidability.
Usage (repository root, env loaded): PYTHONPATH=src python <this> <out.json>"""
import json
import os
import sys

import numpy as np
import pandas as pd

from careport.xfc7_t2_static import CAM_T2, HYP_T2, capacity, t_crit

T2 = "newpaper/autonomous_driving_crossmodal_fit_selection_2026-10-13/tier2_2026-11-01"
COLS = ["segment", "frame", "camera", "object_id", "margin", "sdot", "sigma_b"] + list(HYP_T2.values())


def block(mo, summ):
    keep = mo[(mo["margin"] >= 0.2) | np.isinf(mo["margin"])].copy()
    keep["n_track"] = keep.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
    keep["w"] = keep["sdot"] ** 2 / (keep["sigma_b"] ** 2 * keep["n_track"])
    keep["var_theta"] = keep["sigma_b"] ** 2 / keep["sdot"] ** 2
    cams, theta, thr = {}, {}, {}
    for c, g in keep.groupby("camera"):
        name = CAM_T2[int(c)]
        if name not in summ["decision_cameras"]:
            continue
        G = int(g["segment"].nunique())
        v = float((g["w"] ** 2 * g["var_theta"]).sum() / g["w"].sum() ** 2)
        de = summ["cameras"][name]["DE"]
        hw = float(t_crit(G) * np.sqrt(de * v))
        B = summ["budget"][name]["B_c_ms"]
        thr[name] = 5.0 + B + hw
        theta[name] = {h: float(np.average(g[col], weights=g["w"])) for h, col in HYP_T2.items()}
        n_all = int((mo["camera"] == c).sum())
        cams[name] = {"rows_primary": n_all, "rows_kept": int(len(g)), "logs": G,
                      "tracks": int(g[["segment", "object_id"]].drop_duplicates().shape[0]),
                      "hw_ms": hw, "hw_primary_ms": summ["cameras"][name]["predicted_halfwidth_ms"],
                      "threshold_ms": thr[name], "threshold_primary_ms": summ["thresholds_ms"][name]}
    cap = capacity(theta, thr)
    return {"dropped_rows": int(len(mo) - len(keep)), "cameras": cams, "capacity": cap}


def main(argv):
    root = os.environ["AV2_SENSOR_ROOT"]
    out = {}
    for s in ("050", "030"):
        mo = pd.read_parquet(os.path.join(root, "derived", "t2_static", f"all50_s{s}", "moving_analysis_set_t2.parquet"),
                             columns=COLS)
        with open(f"{T2}/S3_summary_all50_s{s}.json", encoding="utf-8") as f:
            summ = json.load(f)
        out[f"s{s}"] = block(mo, summ)
        r = out[f"s{s}"]
        print(f"score 0.{s[1]}: dropped {r['dropped_rows']} rows")
        for c, v in r["cameras"].items():
            gaps = {k: (round(x["gap"], 1), x["decidable"]) for k, x in r["capacity"]["cameras"][c].items()}
            print(f"  {c:17s} kept {v['rows_kept']}/{v['rows_primary']} logs {v['logs']} hw {v['hw_ms']:.2f} "
                  f"(primary {v['hw_primary_ms']:.2f}) thr {v['threshold_ms']:.2f} (primary {v['threshold_primary_ms']:.2f}) {gaps}")
        print("  dataset:", r["capacity"]["dataset"])
    with open(argv[1], "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main(sys.argv)
