"""Tests for t3_fill_s3.py (written before it). Run from this folder: python -m unittest test_t3_fill_s3 -v"""
import json
import os
import unittest

import t3_fill_s3 as f

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "cluster_results", "derived", "t2h100_static")
FIX2 = os.path.join(HERE, "cluster_results", "fetch2", "derived", "t2h100_static")   # design tables with per-run theta-bar


def load(name):
    folder = FIX2 if name.startswith("design_") else FIX
    with open(os.path.join(folder, name), encoding="utf-8") as fh:
        return json.load(fh)


class MixtureBand(unittest.TestCase):
    def setUp(self):
        # X rejects H_start for pi < 0.5, Y for pi > 0.4, Z for pi > 0.2 (gap = |pi*A + (1 - pi)*B - start|).
        self.theta = {"X": {"start": 0.0, "A": 0.0, "B": 100.0},
                      "Y": {"start": 0.0, "A": 100.0, "B": 0.0},
                      "Z": {"start": 0.0, "A": 100.0, "B": 50.0}}
        self.thr = {"X": 50.0, "Y": 40.0, "Z": 60.0}

    def test_per_camera_sets(self):
        out = f.mixture_band(self.theta, self.thr)
        (lo, hi), = out["per_camera"]["X"]
        self.assertAlmostEqual(lo, 0.0, places=4)
        self.assertAlmostEqual(hi, 0.5, places=4)
        (lo, hi), = out["per_camera"]["Z"]
        self.assertAlmostEqual(lo, 0.2, places=4)
        self.assertAlmostEqual(hi, 1.0, places=4)

    def test_band_where_fewer_than_two_reject(self):
        out = f.mixture_band(self.theta, self.thr)
        (lo, hi), = out["band"]
        self.assertAlmostEqual(lo, 0.0, places=4)
        self.assertAlmostEqual(hi, 0.2, places=4)

    def test_negative_theta_uses_absolute_gap(self):
        out = f.mixture_band({"N": {"start": 0.0, "A": -100.0, "B": 0.0}}, {"N": 50.0})
        (lo, hi), = out["per_camera"]["N"]
        self.assertAlmostEqual(lo, 0.5, places=4)
        self.assertAlmostEqual(hi, 1.0, places=4)

    def test_no_band_when_two_always_reject(self):
        theta = {"P": {"start": 0.0, "A": 100.0, "B": 100.0}, "Q": {"start": 0.0, "A": 90.0, "B": 90.0}}
        out = f.mixture_band(theta, {"P": 10.0, "Q": 10.0})
        self.assertEqual(out["band"], [])

    def test_delta_a_in_band(self):
        # In the band pi in [0, 0.2], delta_A = (1 - pi)(B - A) peaks at pi = 0: X 100, Y 100, Z 50.
        out = f.mixture_band(self.theta, self.thr)
        self.assertAlmostEqual(out["max_abs_delta_A_in_band"]["X"], 100.0, places=3)
        self.assertAlmostEqual(out["max_abs_delta_A_in_band"]["Z"], 50.0, places=3)


class FromT2H100Fixtures(unittest.TestCase):
    """The all-H100 T2 static outputs (job 64097) have the same layout as T3's."""

    def setUp(self):
        self.s3 = load("S3_summary_s050.json")
        self.des = load("design_s050_v2.json")

    def test_rows_follow_the_design_table(self):
        rows = f.camera_rows(self.s3, self.des)
        self.assertEqual([r["camera"] for r in rows], self.des["decision_cameras"])
        for r in rows:
            c = r["camera"]
            self.assertEqual(r["threshold_ms"], self.des["gates"]["primary"]["max"]["thresholds_ms"][c])
            self.assertEqual(r["threshold_excluded_ms"], self.des["gates"]["margin_flag_excluded"]["max"]["thresholds_ms"][c])
            self.assertEqual(r["B_c_ms"], self.s3["budget"][c]["B_c_ms"])
            self.assertEqual(r["hw_t_ms"], self.s3["cameras"][c]["predicted_halfwidth_ms"])
            self.assertEqual(r["nu_bm"], self.des["design"]["primary"][c]["nu_bm"])
            for k in f.CONTRASTS:
                cell = self.des["gates"]["primary"]["max"]["capacity"]["cameras"][c][k]
                self.assertEqual(r["gaps"][k], cell["gap"])
                self.assertEqual(r["over"][k], cell["decidable"])

    def test_decidable_and_gate(self):
        dec = f.decidable(self.des, "primary", "max")
        self.assertEqual(dec, ["start-B"])
        self.assertFalse(f.gate_failed(self.des["gates"]["primary"]["max"]["capacity"]["dataset"]))
        self.assertTrue(f.gate_failed({k: False for k in f.CONTRASTS}))

    def test_excluded_run_mixture_uses_its_own_theta_bar(self):
        s = f.summarise(self.s3, self.des)
        des = json.loads(json.dumps(self.des))
        des["design"]["margin_flag_excluded"]["ring_side_left"]["theta_bar"]["B"] += 20.0
        s2 = f.summarise(self.s3, des)
        self.assertEqual(s2["mixture"]["primary"], s["mixture"]["primary"])
        self.assertNotEqual(s2["mixture"]["margin_flag_excluded"]["per_camera"]["ring_side_left"],
                            s["mixture"]["margin_flag_excluded"]["per_camera"]["ring_side_left"])

    def test_refuses_a_design_table_without_per_run_theta_bar(self):
        des = json.loads(json.dumps(self.des))
        del des["design"]["margin_flag_excluded"]["ring_rear_left"]["theta_bar"]
        with self.assertRaises(SystemExit):
            f.summarise(self.s3, des)

    def test_decidable_for_t3_needs_both_runs(self):
        self.assertEqual(f.decidable_both_runs(self.des, "max"), ["start-B"])
        des = json.loads(json.dumps(self.des))
        des["gates"]["margin_flag_excluded"]["max"]["capacity"]["dataset"]["start-B"] = False
        self.assertEqual(f.decidable_both_runs(des, "max"), [])
        s = f.summarise(self.s3, des)
        self.assertTrue(s["t3_gate"]["max"]["gate_failed"])           # primary alone would pass
        self.assertFalse(s["gates"]["primary/max"]["gate_failed"])

    def test_decision_cameras_need_tracks_and_s1(self):
        self.assertEqual(f.decision_cameras(self.s3), sorted(self.s3["decision_cameras"]))
        s3 = json.loads(json.dumps(self.s3))
        s3["S1_projection_check"]["ring_rear_right"]["pass"] = False
        self.assertNotIn("ring_rear_right", f.decision_cameras(s3))

    def test_markdown_bolds_only_cleared_gaps(self):
        md = f.table_markdown(f.camera_rows(self.s3, self.des))
        line = [ln for ln in md.splitlines() if ln.startswith("| ring_front_left")][0]
        self.assertIn("**77.2**", line)
        self.assertNotIn("**27.5**", line)


class Measured(unittest.TestCase):
    def test_parse(self):
        text = "x\nMEASURED module=careport.xfc7_t2_static peak_rss_GB=7.48 wall_s=319 exit=0\nSTATIC_DONE\n"
        self.assertEqual(f.parse_measured(text),
                         [{"module": "careport.xfc7_t2_static", "peak_rss_GB": 7.48, "wall_s": 319.0, "exit": 0}])

    def test_parse_none(self):
        self.assertEqual(f.parse_measured("nothing here\n"), [])


if __name__ == "__main__":
    unittest.main()
