"""Code check for t3_design_table.py (4 Nov evening change): on T2's frozen PC tables, the margin-flag-excluded run's
capacity gaps must equal those of T2's own tier2_2026-11-01/diagnostics/t2_margin_capacity.py output, and the recomputed
primary theta-bar must equal the S3 summary's. Weights and label-side predictions only.
Run from the repository root: PYTHONPATH=src python -X utf8 <this>"""
import contextlib
import io
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import t3_design_table as dt  # noqa: E402

T2 = os.path.join(HERE, "..", "tier2_2026-11-01")
ref = json.load(open(os.path.join(T2, "diagnostics", "t2_margin_capacity_output.json"), encoding="utf-8"))
with tempfile.TemporaryDirectory() as d:
    for s in ("050", "030"):
        out = os.path.join(d, f"design_s{s}.json")
        with contextlib.redirect_stdout(io.StringIO()):
            dt.main(["t3_design_table.py", os.path.join(T2, f"S3_summary_all50_s{s}.json"),
                     f"D:/datasets/argoverse2/derived/t2_static/all50_s{s}/moving_analysis_set_t2.parquet", out])
        des = json.load(open(out, encoding="utf-8"))
        worst = max(abs(des["gates"]["margin_flag_excluded"]["max"]["capacity"]["cameras"][c][k]["gap"] - cell["gap"])
                    for c, by in ref[f"s{s}"]["capacity"]["cameras"].items() for k, cell in by.items())
        both = {k: des["gates"]["primary"]["max"]["capacity"]["dataset"][k]
                and des["gates"]["margin_flag_excluded"]["max"]["capacity"]["dataset"][k]
                for k in des["gates"]["primary"]["max"]["capacity"]["dataset"]}
        print(f"score 0.{s[1]}: primary theta-bar vs S3 max |diff| {des['primary_theta_bar_check_max_abs_ms']} ms; "
              f"excluded-run gaps vs t2_margin_capacity max |diff| {worst} ms")
        print(f"   hw_BM gate, decidable in both runs: {[k for k, v in both.items() if v] or 'none'}")
