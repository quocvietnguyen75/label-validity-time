"""Dry run 3 (after the code audit's fixes): xfc7_t3_run.analysis on T2's own frozen tables must still reproduce T2's
CR1 verdicts and R4's CR1, CR2-BM and bootstrap intervals; prints the new outputs (both-run decidability with each
run's own theta-bar, C2's CR2-BM interval, the readings and the failure labels). T2's outcomes are known; nothing here
is a T3 quantity. Run from the repository root: PYTHONPATH=src python -X utf8 <this>"""
import json
import time

import pandas as pd

from careport import xfc7_t3_run as t3

base = "D:/paper/newpaper/autonomous_driving_crossmodal_fit_selection_2026-10-13/tier2_2026-11-01/"
s3 = json.load(open(base + "S3_summary_all50_s050.json", encoding="utf-8"))
t2 = json.load(open(base + "T2_RESULTS_2026-11-01.json", encoding="utf-8"))
r4 = json.load(open(base + "posthoc_2026-11-03/r4_results.json", encoding="utf-8"))["R4_1_AV2"]
mo = pd.read_parquet("D:/datasets/argoverse2/derived/t2_static/all50_s050/moving_analysis_set_t2.parquet")
t0 = time.time()
res = t3.analysis(mo, s3, components={c: s3["budget"][c] for c in s3["decision_cameras"]})
print("seconds", round(time.time() - t0))
print("T2 CR1 final reproduced:", res["t2_block"]["verdicts"]["final"] == t2["primary"]["verdicts"]["final"])
worst = max(abs(float(res["three_method"]["intervals"][tag][c][h][m][k]) - float(ints[m][k]))
            for tag in t3.RUNS for c, by_h in r4["cells"][tag].items() for h, ints in by_h.items()
            for m in t3.METHODS for k in ("lo", "hi"))
print("max |difference| from R4's CR1, CR2-BM and bootstrap intervals (ms):", worst)
print("three-method final:", res["three_method"]["three"]["final"])
print("compositions final:", {k: v["verdicts"]["three"]["final"]["start"] for k, v in res["compositions"].items()})
print("excluded-run theta-bar A (own weights):", {c: round(d["theta_bar"]["A"], 2) for c, d in res["design"]["margin_flag_excluded"].items()})
print("decidable, both runs, after timing:", [k for k, v in res["decidable_bm"]["after_timing"]["decidable"].items() if v])
print("reading:", res["reading"])
print("composition readings:", {k: v["reading"] for k, v in res["compositions"].items()})
c2, c2_cr1 = res["C2_pooled_cr2_bm"], res["t2_block"]["C2_pooled_theta_bar"]
print("C2 pooled theta-bar: CR1 %.2f [%.2f, %.2f]; CR2-BM [%.2f, %.2f], df %.1f" % (c2_cr1["est"], c2_cr1["lo"], c2_cr1["hi"], c2["lo"], c2["hi"], c2["df"]))
print("failure labels, primary, start:", {c: v["start"]["three"] for c, v in res["failure_labels"]["primary"].items()})
print("failure labels, primary, start, by method:", {c: {m: v["start"]["by_method"][m]["label"] for m in t3.METHODS} for c, v in res["failure_labels"]["primary"].items()})
