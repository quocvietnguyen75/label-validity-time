"""Tests for t3_design_table.py (T3 section 9), written before its 4 Nov evening change.
Run from this folder: PYTHONPATH=<repo>/src python -m unittest test_t3_design_table -v"""
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import t3_design_table as dt
from careport.xfc7_t2_run import _reweight

THETA_HAT_INPUTS = {"p0_u", "p0_v", "u_ref", "v_ref", "d_u", "d_v", "s", "z_b", "theta_hat", "camera_object_id"}


def moving_set():
    rows = []
    for cam in (2, 6):
        for seg in range(6):
            for obj in range(2):
                for fr in range(2):
                    low = obj == 1 and fr == 0 and seg < 3
                    rows.append({"segment": f"s{seg}", "object_id": f"o{obj}", "camera": cam, "frame": fr,
                                 "frame_type": "A" if fr == 0 else "B", "margin": 0.1 if low else 0.6,
                                 "sdot": 1.0 + 0.5 * obj + 0.1 * seg, "sigma_b": 2.0,
                                 "theta_start": 0.0, "theta_A": 90.0 if low else 30.0 + 10.0 * (cam == 6),
                                 "theta_B": 80.0, "theta_mid": 50.0,
                                 "p0_u": 1e9, "p0_v": 1e9, "u_ref": 0.0, "v_ref": 0.0, "d_u": 1.0, "d_v": 0.0,
                                 "s": 0.0, "z_b": 0.0})
    return _reweight(pd.DataFrame(rows))


def summary(mo, shift=0.0):
    cams = {}
    for c, name in ((2, "ring_front_left"), (6, "ring_rear_left")):
        g = mo[mo["camera"] == c]
        tb = {h: float(np.average(g[col], weights=g["w"])) for h, col in
              (("start", "theta_start"), ("A", "theta_A"), ("B", "theta_B"), ("mid", "theta_mid"))}
        tb["A"] += shift
        cams[name] = {"predicted_theta_bar": tb, "DE": 2.0}
    comp = {"neighbour_bins_ms": 3.0, "ego_class_spread_ms": 4.0, "ego_speed_slope_ms": 1.0, "B_c_ms": 4.0}
    return {"decision_cameras": ["ring_front_left", "ring_rear_left"], "cameras": cams,
            "budget": {"ring_front_left": comp, "ring_rear_left": comp},
            "thresholds_ms": {"ring_front_left": 20.0, "ring_rear_left": 20.0}}


class DesignTableScript(unittest.TestCase):
    def run_main(self, mo, s3):
        d = tempfile.mkdtemp()
        mp, sp, op = (os.path.join(d, n) for n in ("mo.parquet", "s3.json", "out.json"))
        mo.to_parquet(mp, index=False)
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(s3, f)
        dt.main(["t3_design_table.py", sp, mp, op])
        with open(op, encoding="utf-8") as f:
            return json.load(f)

    def test_reads_no_theta_hat_input(self):
        self.assertFalse(THETA_HAT_INPUTS & set(dt.COLUMNS))
        self.assertTrue({"theta_start", "theta_A", "theta_B", "theta_mid"} <= set(dt.COLUMNS))

    def test_excluded_run_gate_uses_that_runs_theta_bar(self):
        mo = moving_set()
        out = self.run_main(mo, summary(mo))
        keep = _reweight(mo[mo["margin"] >= 0.2])
        g = keep[keep["camera"] == 2]
        want = float(np.average(g["theta_A"], weights=g["w"]))
        e = out["design"]["margin_flag_excluded"]["ring_front_left"]["theta_bar"]["A"]
        self.assertAlmostEqual(e, want)
        self.assertAlmostEqual(e, 30.0)
        for comp in ("max", "rss", "sum"):
            gap = out["gates"]["margin_flag_excluded"][comp]["capacity"]["cameras"]["ring_front_left"]["start-A"]["gap"]
            self.assertAlmostEqual(gap, want)
            gp = out["gates"]["primary"][comp]["capacity"]["cameras"]["ring_front_left"]["start-A"]["gap"]
            self.assertAlmostEqual(gp, summary(mo)["cameras"]["ring_front_left"]["predicted_theta_bar"]["A"])
        self.assertGreater(gp, 30.0)

    def test_refuses_a_summary_whose_theta_bar_disagrees(self):
        mo = moving_set()
        with self.assertRaises(SystemExit):
            self.run_main(mo, summary(mo, shift=0.01))


if __name__ == "__main__":
    unittest.main()
