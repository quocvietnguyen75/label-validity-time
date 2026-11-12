"""Dry run 2 (after decidable_both): as t3_repro_dry.py, plus decidability under the hw_BM gate in both runs."""
import json, sys, time
import pandas as pd
from careport import xfc7_t3_run as t3
base = "D:/paper/newpaper/autonomous_driving_crossmodal_fit_selection_2026-10-13/tier2_2026-11-01/"
s3 = json.load(open(base + "S3_summary_all50_s050.json", encoding="utf-8"))
t2 = json.load(open(base + "T2_RESULTS_2026-11-01.json", encoding="utf-8"))
r4 = json.load(open(base + "posthoc_2026-11-03/r4_results.json", encoding="utf-8"))["R4_1_AV2"]
mo = pd.read_parquet("D:/datasets/argoverse2/derived/t2_static/all50_s050/moving_analysis_set_t2.parquet")
t0 = time.time()
comps = {c: s3["budget"][c] for c in s3["decision_cameras"]}
res = t3.analysis(mo, s3, components=comps)
print("seconds", round(time.time() - t0))
print("T2 CR1 final reproduced:", res["t2_block"]["verdicts"]["final"] == t2["primary"]["verdicts"]["final"])
print("CR1 by-method final equals frozen:", res["three_method"]["by_method"]["cr1"]["final"] == t2["primary"]["verdicts"]["final"])
worst = 0.0
for tag in ("primary", "margin_flag_excluded"):
    for c, by_h in r4["cells"][tag].items():
        for h, ints in by_h.items():
            for m in ("cr1", "cr2_bm", "bootstrap"):
                got = res["three_method"]["intervals"][tag][c][h][m]
                for k in ("lo", "hi"):
                    worst = max(worst, abs(float(got[k]) - float(ints[m][k])))
print("max |difference| from R4 intervals (ms):", worst)
print("R4 per-method finals:", {m: r4["decisions"][m]["final"]["start"] for m in ("cr1", "cr2_bm", "bootstrap")})
print("T3 per-method finals:", {m: res["three_method"]["by_method"][m]["final"]["start"] for m in t3.METHODS})
print("three-method final:", res["three_method"]["three"]["final"])
print("three-method primary start rejected_in:", res["three_method"]["three"]["runs"]["primary"]["start"]["rejected_in"])
print("compositions:", {k: v["verdicts"]["three"]["final"]["start"] for k, v in res["compositions"].items()})
print("design primary:", json.dumps({c: {k: round(v, 3) for k, v in d.items()} for c, d in res["design"]["primary"].items()}))
print("decidable_bm all cameras:", res["decidable_bm"]["all_decision_cameras"]["decidable"])
print("decidable_bm per run:", res["decidable_bm"]["all_decision_cameras"]["runs"])
print("decidable_bm after timing:", res["decidable_bm"]["after_timing"]["decidable"], "timing failures:", res["t2_block"]["timing_model_failures"])
for k, v in res["compositions"].items():
    print("composition", k, "decidable_bm:", v["decidable_bm"]["decidable"])
