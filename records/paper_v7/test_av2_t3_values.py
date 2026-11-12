"""Tests for av2_t3_values.py (written before it). Run from this folder: python -m unittest test_av2_t3_values -v

The tool writes Table I's AV2 rows and the derived numbers of the v5 text from the T3 result file. Rejection marks
must come from the file's three-method list, never from the numbers.
"""
import unittest

import av2_t3_values as v

HYP = ("start", "A", "B", "mid")
RUNS = ("primary", "margin_flag_excluded")


def iv(est, lo, hi):
    return {"est": est, "se": 1.0, "lo": lo, "hi": hi, "G": 10, "t": 2.0}


def block(rej_start=("ring_front_left",)):
    cams = ["ring_front_left", "ring_side_right"]
    ints = {run: {c: {h: {m: iv(2.0, 1.0, 3.0) if c == "ring_front_left" else iv(-26.04, -30.34, -21.62)
                          for m in ("cr1", "cr2_bm", "bootstrap")} for h in HYP} for c in cams} for run in RUNS}
    three = {run: {h: {"rejected_in": list(rej_start) if h == "start" else []} for h in HYP} for run in RUNS}
    cap = {"cameras": {"ring_front_left": {"start-B": {"gap": 50.0, "threshold": 30.0, "decidable": True}},
                       "ring_side_right": {"start-B": {"gap": 10.0, "threshold": 99.0, "decidable": False}}}}
    design = {c: {"tracks": 1187 if c == "ring_front_left" else 78, "logs": 295 if c == "ring_front_left" else 50,
                  "kish_logs": 68.21 if c == "ring_front_left" else 1.49} for c in cams}
    return {"cameras": cams, "budgets_ms": {"ring_front_left": 15.149, "ring_side_right": 68.64},
            "three_method": {"intervals": ints, "three": {"runs": three}},
            "gate_bm": {"primary": {"capacity": cap}}, "design": {"primary": design}}


class Formatting(unittest.TestCase):
    def test_tex_interval(self):
        self.assertEqual(v.tex_ci(iv(-26.04, -30.34, -21.62)), "$-$26.0 [$-$30.3, $-$21.6]")
        self.assertEqual(v.tex_ci(iv(52.06, 47.63, 56.47)), "52.1 [47.6, 56.5]")  # 52.05 is stored below .05


class TableRows(unittest.TestCase):
    def test_marks_follow_the_file(self):
        rows = v.table_rows(block())
        fl = [r for r in rows if r.startswith("Front left")][0]
        sr = [r for r in rows if r.startswith("Side right")][0]
        # front left is listed as rejecting although its interval [1, 3] lies inside the band: bold anyway
        self.assertIn("\\textbf{2.0 [1.0, 3.0]}", fl)
        self.assertIn("1,187 (295; 68.2)", fl)
        self.assertTrue(fl.endswith("& 20.1 \\\\"))
        # side right clears no contrast: dagger; not listed: no bold
        self.assertTrue(sr.startswith("Side right$^\\dagger$"))
        self.assertNotIn("\\textbf", sr)

    def test_ring_order(self):
        rows = v.table_rows(block())
        self.assertEqual([r.split(" &")[0] for r in rows], ["Front left", "Side right$^\\dagger$"])


class Clearances(unittest.TestCase):
    def test_second_largest_then_min_over_runs(self):
        b = {"cameras": ["a", "b", "c"], "budgets_ms": {"a": 15.0, "b": 15.0, "c": 15.0},
             "three_method": {"intervals": {}, "three": {"runs": {}}}}
        lows = {"primary": {"a": (31.0, 30.0, 32.0), "b": (50.0, 51.0, 52.0), "c": (45.0, 46.0, 47.0)},
                "margin_flag_excluded": {"a": (48.0, 49.0, 50.0), "b": (32.0, 33.0, 34.0)}}
        for run, cams in lows.items():
            b["three_method"]["intervals"][run] = {
                c: {"start": {m: iv(60.0, lo, 70.0) for m, lo in zip(("cr1", "cr2_bm", "bootstrap"), los)}}
                for c, los in cams.items()}
            b["three_method"]["three"]["runs"][run] = {"start": {"rejected_in": list(cams)}}
        out = v.clearances(b)
        self.assertEqual(out["per_run"]["primary"], {"a": 10.0, "b": 30.0, "c": 25.0})
        self.assertEqual(out["threshold"]["primary"], 25.0)
        self.assertEqual(out["threshold"]["margin_flag_excluded"], 12.0)
        self.assertEqual(out["shift_to_remove_ms"], 12.0)


class Metres(unittest.TestCase):
    def test_arithmetic(self):
        b = {"cameras": ["x"], "budgets_ms": {"x": 15.1},
             "three_method": {"intervals": {"primary": {"x": {"start": {"cr1": iv(52.0, 47.6, 56.5),
                                                                         "cr2_bm": iv(52.0, 47.5, 56.6),
                                                                         "bootstrap": iv(52.0, 47.55, 56.6)}}}},
                              "three": {"runs": {"primary": {"start": {"rejected_in": ["x"]}}}}}}
        m = v.metres(b, speed=10.0)["x"]
        self.assertAlmostEqual(m["cr1_lo_m"], 0.476)
        self.assertAlmostEqual(m["cr1_hi_m"], 0.565)
        self.assertAlmostEqual(m["after_budget_min_m"], 0.324)


if __name__ == "__main__":
    unittest.main()
