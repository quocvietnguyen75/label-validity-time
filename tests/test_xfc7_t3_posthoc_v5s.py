"""Synthetic tests for the XF-C7 post hoc set V5S (pilot record entry '2026-11-05, post hoc set V5S'), written first."""
import unittest

from careport import xfc7_t3_posthoc_v5s as v

M3 = ("cr1", "cr2_bm", "bootstrap")


def _ints(lo_by_method, est=50.0, hi=90.0):
    return {m: {"est": est, "lo": lo, "hi": hi} for m, lo in lo_by_method.items()}


class Floor(unittest.TestCase):
    def test_floor_over_cameras_and_methods(self):
        iv = {"c1": {"start": _ints({"cr1": 49.3, "cr2_bm": 48.0, "bootstrap": 49.0})},
              "c2": {"start": _ints({"cr1": 40.0, "cr2_bm": 41.0, "bootstrap": 39.5})}}
        b = {"c1": 27.9, "c2": 10.0}
        cr1 = v.floor_metres(iv, b, ["c1", "c2"], ("cr1",))
        self.assertAlmostEqual(cr1["per_camera_m"]["c1"], (49.3 - 27.9) * 10 / 1000, places=12)
        self.assertAlmostEqual(cr1["floor_m"], (49.3 - 27.9) / 100, places=12)
        three = v.floor_metres(iv, b, ["c1", "c2"], M3)
        self.assertAlmostEqual(three["floor_m"], (48.0 - 27.9) / 100, places=12)
        self.assertEqual(three["binding_camera"], "c1")


class RealisedHalfWidth(unittest.TestCase):
    def test_largest_over_methods_and_both_conventions(self):
        by_h = {"start": {"cr1": {"est": 0, "lo": -2, "hi": 2}, "cr2_bm": {"est": 0, "lo": -3, "hi": 3},
                          "bootstrap": {"est": 0, "lo": -2.5, "hi": 2.5}},
                "A": {"cr1": {"est": 0, "lo": -4, "hi": 4}, "cr2_bm": {"est": 0, "lo": -1, "hi": 1},
                      "bootstrap": {"est": 0, "lo": -1, "hi": 1}}}
        self.assertAlmostEqual(v.realised_hw(by_h, "start", "A", M3), 4.0)
        self.assertAlmostEqual(v.realised_hw(by_h, "start", "start", ("cr1",)), 2.0)


class RealisedDecidability(unittest.TestCase):
    def test_two_cameras_in_both_runs(self):
        th = {"p": {"c1": {"start": 0.0, "A": 40.0}, "c2": {"start": 0.0, "A": 30.0}},
              "m": {"c1": {"start": 0.0, "A": 40.0}, "c2": {"start": 0.0, "A": 20.0}}}
        b = {"c1": 10.0, "c2": 10.0}
        hw = {"p": {"c1": {"start-A": 5.0}, "c2": {"start-A": 5.0}}, "m": {"c1": {"start-A": 5.0}, "c2": {"start-A": 6.0}}}
        d = v.realised_decidability(th, b, hw, contrasts=[("start", "A")])
        self.assertTrue(d["runs"]["p"]["start-A"])            # 40 > 20, 30 > 20
        self.assertFalse(d["runs"]["m"]["start-A"])           # 20 > 21 fails in c2
        self.assertFalse(d["decidable"]["start-A"])


class Screen(unittest.TestCase):
    def test_support_needs_two_passing_rejecting_cameras_in_both_runs(self):
        written = {"c1": {"median_px": 10.0}, "c2": {"median_px": 14.9}, "c3": {"median_px": 15.2}}
        rej = {"p": ["c1", "c3"], "m": ["c1", "c2", "c3"]}
        s = v.screen_support(written, rej, limit=15.0)
        self.assertEqual(s["passing"], ["c1", "c2"])
        self.assertEqual(s["rejecting_and_passing"], {"p": ["c1"], "m": ["c1", "c2"]})
        self.assertFalse(s["rests_on_two_passing_cameras_in_both_runs"])


class Quadrature(unittest.TestCase):
    def test_frozen_rule_on_stored_triples(self):
        cells = {"p": {"F": {"frame": _ints({"cr1": 40.0}, est=45.0, hi=50.0)},
                       "L": {"frame": _ints({"cr1": 30.0}, est=35.0, hi=40.0)}},
                 "m": {"F": {"frame": _ints({"cr1": 40.0}, est=45.0, hi=50.0)},
                       "L": {"frame": _ints({"cr1": 28.0}, est=33.0, hi=38.0)}}}
        out = v.judge_stored(cells, {"F": 30.0, "L": 24.0}, ["F", "L"], ["frame"], "cr1", runs=("p", "m"))
        self.assertEqual(out["runs"]["p"]["frame"]["rejected_in"], ["F", "L"])   # 40 > 35, 30 > 29
        self.assertEqual(out["runs"]["m"]["frame"]["rejected_in"], ["F"])        # 28 > 29 fails
        self.assertEqual(out["final"]["frame"], "association-sensitive, not ruled out")


class Display(unittest.TestCase):
    def test_min_clearance_over_methods_and_runs(self):
        iv = {"p": {"c1": {"start": _ints({"cr1": 47.6, "cr2_bm": 47.0, "bootstrap": 47.3})}},
              "m": {"c1": {"start": _ints({"cr1": 46.5, "cr2_bm": 47.2, "bootstrap": 46.9})}}}
        self.assertAlmostEqual(v.min_clearance(iv, "c1", "start", 20.1, M3, runs=("p", "m")), 46.5 - 20.1)

    def test_interval_text(self):
        self.assertEqual(v.ci_text({"est": 52.06, "lo": 47.61, "hi": 56.54}), "52.1 [47.6, 56.5]")
        self.assertEqual(v.ci_text({"est": -2.04, "lo": -6.46, "hi": 2.4}), "−2.0 [−6.5, 2.4]")


if __name__ == "__main__":
    unittest.main()
