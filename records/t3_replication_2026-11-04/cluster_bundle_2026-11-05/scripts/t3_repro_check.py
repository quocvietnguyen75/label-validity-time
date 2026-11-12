"""XF-C7 T3, SAMPLE_PIN addendum 7, cross-platform check 2: xfc7_t3_run.analysis on T2's frozen tables must reproduce
T2's frozen CR1 verdicts and R4's CR1, CR2-BM and bootstrap intervals within 0.01 ms (exit 1 otherwise).
Usage: python t3_repro_check.py <S3_summary_all50_s050.json> <T2_RESULTS.json> <r4_results.json> <moving_set.parquet> <out.json>"""
import json
import sys

import pandas as pd

from careport import xfc7_t3_run as t3

TOL_MS = 0.01


def main(argv):
    s3 = json.load(open(argv[1], encoding="utf-8"))
    t2 = json.load(open(argv[2], encoding="utf-8"))
    r4 = json.load(open(argv[3], encoding="utf-8"))["R4_1_AV2"]
    mo = pd.read_parquet(argv[4])
    res = t3.analysis(mo, s3, components={c: s3["budget"][c] for c in s3["decision_cameras"]})
    worst = 0.0
    for tag in ("primary", "margin_flag_excluded"):
        for c, by_h in r4["cells"][tag].items():
            for h, ints in by_h.items():
                for m in t3.METHODS:
                    got = res["three_method"]["intervals"][tag][c][h][m]
                    worst = max(worst, *(abs(float(got[k]) - float(ints[m][k])) for k in ("lo", "hi")))
    out = {"t2_cr1_final_reproduced": res["t2_block"]["verdicts"]["final"] == t2["primary"]["verdicts"]["final"],
           "cr1_by_method_equals_frozen": res["three_method"]["by_method"]["cr1"]["final"] == t2["primary"]["verdicts"]["final"],
           "max_abs_diff_from_r4_ms": worst,
           "per_method_final_start": {m: res["three_method"]["by_method"][m]["final"]["start"] for m in t3.METHODS},
           "three_method_final": res["three_method"]["three"]["final"]}
    out["pass"] = bool(out["t2_cr1_final_reproduced"] and out["cr1_by_method_equals_frozen"] and worst <= TOL_MS)
    json.dump(out, open(argv[5], "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
