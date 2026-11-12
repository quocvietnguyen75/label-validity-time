"""Code check for t3_fill_s3.mixture_band: on T2's frozen PC S3 summaries (t_{G-1} thresholds), it must give T2 s.9's
published mixture band (0.61-0.99 at score >= 0.5; 0.52-0.55 at >= 0.3) and per-camera cut points (front-left < 0.36,
side-left < 0.61, rear-left > 0.07, rear-right > 0.99), and |delta_A| of 19-20 ms in the band.
Run from the T3 folder: python -X utf8 code_checks/t3_fill_s3_t2_check.py"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import t3_fill_s3 as f  # noqa: E402

for tag in ("s050", "s030"):
    p = os.path.join(HERE, "..", "tier2_2026-11-01", f"S3_summary_all50_{tag}.json")
    s = json.load(open(p, encoding="utf-8"))
    cams = s["decision_cameras"]
    m = f.mixture_band({c: s["cameras"][c]["predicted_theta_bar"] for c in cams}, {c: s["thresholds_ms"][c] for c in cams})
    print(tag, hashlib.sha256(open(p, "rb").read()).hexdigest()[:16], "band", [[round(a, 3), round(b, 3)] for a, b in m["band"]])
    for c, v in m["per_camera"].items():
        print("  ", c, [[round(a, 3), round(b, 3)] for a, b in v])
    print("   max |delta_A| in band", {c: round(v, 1) for c, v in m["max_abs_delta_A_in_band"].items()},
          "5 + B^det", {c: round(5 + s["budget"][c]["B_c_ms"], 1) for c in cams})
