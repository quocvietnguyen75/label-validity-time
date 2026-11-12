"""Tests for make_figures.py. v5 tests (panel (b) reads the T3 result file) kept; v6 tests (written before the v6
changes): IEEEtran journal widths, and the bold and dagger row labels taken from the result files.
Run from this folder: python -m unittest test_make_figures -v"""
import unittest

import make_figures as mf


def iv(est, lo, hi):
    return {"est": est, "lo": lo, "hi": hi}


class AV2Rows(unittest.TestCase):
    def test_rows_from_t3(self):
        cams = [c for c, _ in mf.AV2_CAMS]
        ints = {c: {"start": {"cr1": iv(52.0, 47.0, 57.0)}, "A": {"cr1": iv(23.0, 19.0, 28.0)},
                    "B": {"cr1": iv(-26.0, -30.0, -21.0)}, "mid": {"cr1": iv(2.0, -2.0, 6.0)}} for c in cams}
        t3 = {"primary": {"budgets_ms": {c: 15.0 for c in cams},
                          "three_method": {"intervals": {"primary": ints}}}}
        rows = mf.av2_rows(t3)
        self.assertEqual([r["name"] for r in rows], [n for _, n in mf.AV2_CAMS])
        r = rows[0]
        self.assertEqual(r["start"], ints[cams[0]]["start"]["cr1"])
        self.assertEqual(r["tol"], 20.0)
        self.assertEqual(r["pred_A"], 52.0 - 23.0)
        self.assertEqual(r["pred_B"], 52.0 + 26.0)

    def test_seven_cameras_in_ring_order(self):
        self.assertEqual([c for c, _ in mf.AV2_CAMS],
                         ["ring_front_center", "ring_front_left", "ring_front_right", "ring_side_left",
                          "ring_side_right", "ring_rear_left", "ring_rear_right"])


class Widths(unittest.TestCase):
    def test_ieeetran_journal_widths(self):
        # \columnwidth 252 pt and \textwidth 516 pt, measured from IEEEtran v1.8b [journal] on 6 Nov 2026
        self.assertAlmostEqual(mf.COLW, 252.0 / 72.27)
        self.assertAlmostEqual(mf.TEXTW, 516.0 / 72.27)


class WodFlags(unittest.TestCase):
    def test_bold_from_primary_rejections_and_detector_dagger(self):
        cams = [c for c, _ in mf.WOD_CAMS]
        v2 = {"verdicts_primary_run": {"frame": {"rejected_in": cams}}}
        t1 = {"primary": {"E1": {"verdicts_primary_run": {"frame": {"rejected_in": ["FRONT", "SIDE_LEFT"]}}}}}
        f = mf.wod_row_flags(v2, t1)
        self.assertEqual(f[("FRONT_RIGHT", "H")], (True, False))
        self.assertEqual(f[("FRONT_RIGHT", "D")], (False, True))
        self.assertEqual(f[("FRONT_LEFT", "D")], (False, False))
        self.assertEqual(f[("SIDE_LEFT", "D")], (True, False))
        self.assertEqual(len(f), 2 * len(cams))


class Av2Flags(unittest.TestCase):
    def test_bold_from_three_method_and_dagger_from_gate(self):
        cams = [c for c, _ in mf.AV2_CAMS]
        cap = {c: {"start-A": {"decidable": c != "ring_side_right"}, "A-B": {"decidable": False}} for c in cams}
        t3 = {"primary": {"three_method": {"three": {"runs": {"primary": {
                  "start": {"rejected_in": ["ring_front_left", "ring_rear_right"]}}}}},
              "gate_bm": {"primary": {"capacity": {"cameras": cap}}}}}
        f = mf.av2_label_flags(t3)
        self.assertEqual(f["ring_front_left"], (True, False))
        self.assertEqual(f["ring_rear_right"], (True, False))
        self.assertEqual(f["ring_side_right"], (False, True))
        self.assertEqual(f["ring_side_left"], (False, False))


class LabelText(unittest.TestCase):
    def test_dagger_suffix(self):
        self.assertEqual(mf.label_text("Side right", True), "Side right†")
        self.assertEqual(mf.label_text("Front", False), "Front")


if __name__ == "__main__":
    unittest.main()
