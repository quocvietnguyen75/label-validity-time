"""Synthetic tests for the T2 post hoc set V1-V5 (pilot record, 2026-11-02), written before the module."""
import math
import unittest

import numpy as np
import pandas as pd

from careport import xfc7_t2_posthoc as ph


def _ci(est, half):
    return {"est": est, "lo": est - half, "hi": est + half}


class CompositionTest(unittest.TestCase):
    def test_max_rss_sum(self):
        comp = {"c1": {"neighbour_bins_ms": 3.0, "ego_class_spread_ms": 4.0, "ego_speed_slope_ms": 0.0}}
        b = ph.composition_budgets(comp)["c1"]
        self.assertAlmostEqual(b["max"], 4.0)
        self.assertAlmostEqual(b["rss"], 5.0)
        self.assertAlmostEqual(b["sum"], 7.0)

    def test_verdict_flips_under_sum(self):
        # documented time: intervals clear 5 + max (=15) in two cameras, but not 5 + sum (=25) in one of them
        comp = {"c1": {"neighbour_bins_ms": 10.0, "ego_class_spread_ms": 10.0, "ego_speed_slope_ms": 0.0},
                "c2": {"neighbour_bins_ms": 10.0, "ego_class_spread_ms": 1.0, "ego_speed_slope_ms": 0.0}}
        est = {"c1": {"start": _ci(30.0, 5.0)}, "c2": {"start": _ci(40.0, 5.0)}}
        out = ph.verdicts_by_composition(est, est, comp, ["c1", "c2"], ["start"])
        self.assertEqual(out["max"]["final"]["start"], "ruled out")
        self.assertEqual(out["sum"]["final"]["start"], "not ruled out")
        self.assertEqual(out["sum"]["runs"]["primary"]["start"]["rejected_in"], ["c2"])

    def test_primary_only(self):
        comp = {"c1": {"neighbour_bins_ms": 1.0, "ego_class_spread_ms": 1.0, "ego_speed_slope_ms": 1.0}}
        est = {"c1": {"frame": _ci(20.0, 1.0)}}
        out = ph.rejections_by_composition(est, comp, ["c1"], ["frame"])
        self.assertEqual(out["sum"]["frame"]["rejected_in"], ["c1"])        # 19 > 5 + 3


class Tier1IntervalsTest(unittest.TestCase):
    def test_unwraps_the_delta_level(self):
        e1 = {"estimates": {"FRONT": {"tracks": 3, "obs": 9, "delta": {"frame": _ci(48.0, 2.0)}}}}
        out = ph.tier1_intervals(e1, ["FRONT"])
        self.assertEqual(out["FRONT"]["frame"]["lo"], 46.0)
        r = ph.rejections_by_composition(out, {"FRONT": {"neighbour_bins_ms": 24.0, "ego_class_spread_ms": 22.0,
                                                         "ego_speed_slope_ms": 0.2}}, ["FRONT"], ["frame"])
        self.assertEqual(r["max"]["frame"]["rejected_in"], ["FRONT"])     # 46 > 29
        self.assertEqual(r["sum"]["frame"]["rejected_in"], [])            # 46 < 51.2


class RealizedDecidabilityTest(unittest.TestCase):
    def test_threshold_uses_realized_half_width(self):
        cap = {"c1": {"start-A": {"gap": 80.0}}, "c2": {"start-A": {"gap": 80.0}}}
        budgets = {"c1": 40.0, "c2": 40.0}
        out = ph.realized_decidability(cap, budgets, {"c1": 10.0, "c2": 40.0}, [("start", "A")])
        self.assertTrue(out["cameras"]["c1"]["start-A"]["decidable"])      # 80 > 55
        self.assertFalse(out["cameras"]["c2"]["start-A"]["decidable"])     # 80 < 85
        self.assertFalse(out["dataset"]["start-A"])


class JointSpeedModelTest(unittest.TestCase):
    def test_recovers_slopes_with_camera_effects(self):
        rng = np.random.default_rng(0)
        rows = []
        for seg in range(30):
            for cam, base in ((1, 40.0), (2, 70.0)):
                for i in range(12):
                    v = rng.uniform(3, 20)
                    sdot = rng.uniform(0.3, 3.0)
                    theta = base - 1.5 * v + 4.0 / sdot + rng.normal(0, 0.5)
                    rows.append({"segment": f"s{seg}", "camera": cam, "speed": v, "sdot": sdot, "theta_hat": theta, "w": 1.0})
        mo = pd.DataFrame(rows)
        out = ph.joint_speed_model(mo)
        self.assertAlmostEqual(out["speed"]["coef"], -1.5, delta=0.05)
        self.assertAlmostEqual(out["inv_sdot"]["coef"], 4.0, delta=0.2)
        self.assertLess(out["speed"]["hi"], 0.0)
        self.assertEqual(out["G"], 30)


class ConventionsTest(unittest.TestCase):
    def test_midpoint_and_pooled(self):
        mo = pd.DataFrame({"theta_A": [10.0, 10.0, 10.0], "theta_B": [60.0, 60.0, 60.0],
                           "n_pts_A": [30, 0, 10], "n_pts_B": [10, 25, 0]})
        out = ph.add_conventions(mo)
        self.assertTrue(np.allclose(out["theta_mp"], 35.0))
        self.assertAlmostEqual(out["theta_pool"].iloc[0], (30 * 10 + 10 * 60) / 40)
        self.assertAlmostEqual(out["theta_pool"].iloc[1], 60.0)
        self.assertAlmostEqual(out["theta_pool"].iloc[2], 10.0)


class SeamRowsTest(unittest.TestCase):
    def test_selects_only_seam_rows_that_pass_everything_else(self):
        base = dict(most_visible=True, num_pts=30, inside_all=True, overlap=False, static_proj_ok=True,
                    speed=8.0, sdot=1.0, along_axis_deg=45.0, seam=True)
        lab = pd.DataFrame([dict(base, object_id="a"), dict(base, object_id="b", seam=False),
                            dict(base, object_id="c", speed=2.0), dict(base, object_id="d", overlap=True)])
        self.assertEqual(list(ph.seam_only_moving(lab)["object_id"]), ["a"])


class SpreadAndLooTest(unittest.TestCase):
    def test_weighted_quantiles(self):
        q = ph.weighted_quantiles(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 1.0, 1.0, 1.0]), [0.25, 0.5, 1.0])
        self.assertEqual(q, [1.0, 2.0, 4.0])

    def test_loo_range(self):
        g = pd.DataFrame({"segment": ["a", "a", "b", "c"], "theta_hat": [10.0, 10.0, 20.0, 30.0],
                          "theta_start": 0.0, "w": [1.0, 1.0, 1.0, 1.0]})
        out = ph.loo_range(g, "theta_start")
        self.assertAlmostEqual(out["min"], 40.0 / 3.0)                     # leaving out c: (10 + 10 + 20) / 3
        self.assertAlmostEqual(out["max"], 25.0)                           # leaving out a: (20 + 30) / 2
        self.assertEqual(out["n_logs"], 3)


class VelocityScaleTest(unittest.TestCase):
    def test_ratio(self):
        out = ph.velocity_scale({"diff_ms": 3.9, "lo": 0.1, "hi": 7.7})
        self.assertTrue(math.isclose(out["est"], 0.078) and math.isclose(out["hi"], 0.154))


if __name__ == "__main__":
    unittest.main()
