"""Tests for t3_results_tables.py (written before it). Run from this folder: python -m unittest test_t3_results_tables -v

The tool transcribes the T3 result file into markdown tables. It must copy verdicts and rejection sets from
the file, never recompute them, and refuse to overwrite its output.
"""
import json
import os
import tempfile
import unittest

import t3_results_tables as t

HYP = ("start", "A", "B", "mid")
CAMS = ("ring_a", "ring_b")


def iv(est, lo, hi, G=10):
    return {"est": est, "se": 1.0, "lo": lo, "hi": hi, "G": G, "t": 2.0}


def method_runs(rej):
    return {run: {h: {"rejected_in": list(rej.get(h, [])), "ruled_out": len(rej.get(h, [])) >= 2} for h in HYP}
            for run in ("primary", "margin_flag_excluded")}


def block():
    # ring_a's delta_start interval [1, 3] lies inside its band (+-15), yet ring_a is listed as rejecting:
    # the table must follow the file's rejection list, not the numbers.
    ints = {run: {c: {h: {m: iv(2.0, 1.0, 3.0) if c == "ring_a" else iv(-40.25, -50.0, -30.5)
                          for m in ("cr1", "cr2_bm", "bootstrap")} for h in HYP} for c in CAMS}
            for run in ("primary", "margin_flag_excluded")}
    rej = {"start": ["ring_a"]}
    three_runs = method_runs(rej)
    for run in three_runs:
        for h in HYP:
            three_runs[run][h]["rejected_in_by_method"] = {m: list(rej.get(h, [])) for m in ("cr1", "cr2_bm", "bootstrap")}
    final = {"start": "not ruled out", "A": "not ruled out", "B": "not ruled out", "mid": "not ruled out"}
    return {"cameras": list(CAMS), "budgets_ms": {"ring_a": 10.0, "ring_b": 20.0},
            "three_method": {"intervals": ints, "three": {"runs": three_runs, "final": dict(final)},
                             "by_method": {m: {"runs": method_runs(rej), "final": dict(final)}
                                           for m in ("cr1", "cr2_bm", "bootstrap")},
                             "undefined_intervals": []}}


class Numbers(unittest.TestCase):
    def test_minus_sign_and_none(self):
        self.assertEqual(t.num(-37.649), "−37.6")
        self.assertEqual(t.num(2.8949, 2), "2.89")
        self.assertEqual(t.num(-2.8949, 2), "−2.89")
        self.assertEqual(t.num(None), "–")

    def test_interval(self):
        self.assertEqual(t.ci(iv(39.44, 27.36, 51.54)), "39.4 [27.4, 51.5]")
        self.assertEqual(t.ci(iv(-0.5, -43.51, 42.4)), "−0.5 [−43.5, 42.4]")


class Transcription(unittest.TestCase):
    def test_bold_follows_file_rejections(self):
        md = t.estimates_table(block(), "primary")
        row_a = [ln for ln in md.splitlines() if ln.startswith("| ring_a | start")][0]
        row_b = [ln for ln in md.splitlines() if ln.startswith("| ring_b | start")][0]
        self.assertIn("**", row_a)        # listed as rejecting, though inside its band
        self.assertNotIn("**", row_b)     # outside its band, but not listed
        self.assertIn("±15.0", row_a)     # band = 5 + B^det

    def test_finals_copied(self):
        b = block()
        b["three_method"]["three"]["final"]["mid"] = "association-sensitive, not ruled out"
        line = t.finals(b["three_method"]["three"]["final"])
        self.assertIn("mid: association-sensitive, not ruled out", line)
        self.assertIn("start: not ruled out", line)

    def test_rejection_lines_both_runs_and_methods(self):
        md = t.rejection_lines(block())
        self.assertIn("three-method, primary: start ring_a; A none; B none; mid none", md)
        self.assertIn("bootstrap, margin_flag_excluded: start ring_a; A none; B none; mid none", md)


class CompactTable(unittest.TestCase):
    # T2's report layout: one row per camera, CR1 intervals, bold where all three methods reject (the file's list).
    def test_bold_follows_three_method_list(self):
        md = t.compact_table(block(), "primary")
        row_a = [ln for ln in md.splitlines() if ln.startswith("| ring_a |")][0]
        row_b = [ln for ln in md.splitlines() if ln.startswith("| ring_b |")][0]
        self.assertEqual(row_a, "| ring_a | 15.0 | **2.0 [1.0, 3.0]** | 2.0 [1.0, 3.0] | 2.0 [1.0, 3.0] | 2.0 [1.0, 3.0] |")
        self.assertNotIn("**", row_b)
        self.assertIn("−40.2 [−50.0, −30.5]", row_b)


class ShareTable(unittest.TestCase):
    def test_rows_and_total(self):
        shares = {"input_sha256": "ab" * 32, "rows": 10, "label_side_shares": {
            "ring_rear_left": {"moving_before_seam_rule": 13237, "removed_by_seam_rule": 4596, "share_removed": 0.347,
                               "included": 8641, "share_theta_A_fallback": 0.0, "share_theta_B_fallback": 0.02},
            "_all": {"moving_before_seam_rule": 47005, "removed_by_seam_rule": 5546, "share_removed": 0.118}}}
        md = t.shares_table(shares)
        self.assertIn("| ring_rear_left | 13,237 | 4,596 (34.7%) | 8,641 | 0.0% | 2.0% |", md)
        self.assertIn("| all cameras | 47,005 | 5,546 (11.8%) | 41,459 | – | – |", md)


class ShiftRecovery(unittest.TestCase):
    def e4(self, ratio, flag):
        cams = {"ring_a": {"recovered_shift_ms": 10.0 * ratio, "recovery_ratio": ratio},
                "ring_b": {"recovered_shift_ms": 9.9, "recovery_ratio": 0.99}}
        return {"baseline_delta_start_ms": {"ring_a": 1.0, "ring_b": 2.0}, "baseline_reproduced": True,
                "zero_shift_changed_share": 0.0, "pass_band_90_110": flag,
                "runs": {"+10ms": {"assignment_changed_share": 0.001, "cameras": cams, "missing_cameras": []}}}

    def test_cell_outside_band_marked(self):
        md = t.e4_table(self.e4(0.784, False), ["ring_a", "ring_b"])
        self.assertIn("**78.4%**", md)
        self.assertIn("99.0%", md)

    def test_inconsistent_flag_stops(self):
        with self.assertRaises(SystemExit):
            t.e4_table(self.e4(0.784, True), ["ring_a", "ring_b"])


class DesignRows(unittest.TestCase):
    def test_rows_match_protocol_table(self):
        b = block()
        b["design"] = {run: {c: {"tracks": 30, "logs": 12, "nu_bm": 5.04, "kish_logs": 6.0, "top_log_share": 0.171,
                                 "top5_track_share": 0.25, "V_ind": 1.0, "theta_bar": {}} for c in CAMS}
                       for run in ("primary", "margin_flag_excluded")}
        b["halfwidths"] = {c: {"predicted_t_G_minus_1_ms": 3.0, "predicted_bm_ms": 3.5,
                               "realised_start_ms": {"cr1": 1, "cr2_bm": 1, "bootstrap": 1}} for c in CAMS}
        caps = {run: {"cameras": {c: {k: {"gap": 50.0 if k == "start-B" else 1.0, "threshold": 18.5,
                                          "decidable": k == "start-B"} for k in t.CONTRASTS} for c in CAMS}}
                for run in ("primary", "margin_flag_excluded")}
        b["gate_bm"] = {run: {"hw_bm_ms": {c: 3.5 for c in CAMS}, "thresholds_ms": {c: 18.5 for c in CAMS},
                              "capacity": caps[run]} for run in ("primary", "margin_flag_excluded")}
        rows = t.design_rows(b)
        self.assertEqual(rows[0], "| ring_a | 30 (12) | 10.0 | 3.0 | 5.0 | 3.5 | 6.0 | 17% | 25% | 18.5 [18.5] "
                                  "| 1.0 | **50.0** | 1.0 | 1.0 | 1.0 | 1.0 |")
        protocol = "intro\n| Camera | Tracks (logs) |\n|---|---|\n" + "\n".join(rows) + "\n\nnext"
        self.assertEqual(t.protocol_tables(protocol), [rows])


class FailureLabels(unittest.TestCase):
    # The runner's 'capacity failure' does not require the realised half-width to exceed hw^BM (T2 section 10's
    # wording does); the file carries that comparison as a flag, and the table must show it beside each label.
    def test_flag_shown_beside_label(self):
        b = block()
        per = {"cr1": (4.4, False), "cr2_bm": (19.0, True), "bootstrap": (4.6, None)}
        x = {"three": "capacity failure", "predicted_hw_bm_ms": 18.6, "camera_clears": {"A-B": True, "start-A": False},
             "by_method": {m: {"label": "capacity failure", "realised_hw_ms": hw, "realised_exceeds_hw_bm": ex}
                           for m, (hw, ex) in per.items()}}
        b["failure_labels"] = {run: {c: {h: x for h in HYP} for c in CAMS} for run in ("primary", "margin_flag_excluded")}
        row = [ln for ln in t.failure_table(b, "primary").splitlines() if ln.startswith("| ring_a | A ")][0]
        self.assertIn("capacity failure (4.4, not above hw^BM)", row)
        self.assertIn("capacity failure (19.0, above hw^BM)", row)
        self.assertIn("capacity failure (4.6, no prediction)", row)
        self.assertIn("| A-B |", row)


class SecondaryBand(unittest.TestCase):
    # S-f's file holds intervals only, so the tool marks "entirely outside the band" itself, and says so.
    def test_outside(self):
        self.assertTrue(t.outside(iv(30.0, 25.1, 35.0), 25.0))
        self.assertTrue(t.outside(iv(-30.0, -35.0, -25.1), 25.0))
        self.assertFalse(t.outside(iv(30.0, 24.9, 35.0), 25.0))
        self.assertFalse(t.outside(iv(0.0, -30.0, 30.0), 25.0))


class Refusals(unittest.TestCase):
    def test_protocol_must_be_the_runs(self):
        with tempfile.TemporaryDirectory() as d:
            res, proto, out = (os.path.join(d, n) for n in ("r.json", "p.md", "o.md"))
            with open(res, "w", encoding="utf-8") as fh:
                json.dump({"protocol_sha256": "0" * 64, "settings": {"margin_ms": 5.0}}, fh)
            with open(proto, "w", encoding="utf-8") as fh:
                fh.write("not the frozen protocol")
            with self.assertRaises(SystemExit):
                t.main(["t3_results_tables.py", res, proto, out])
            self.assertFalse(os.path.exists(out))


class NoOverwrite(unittest.TestCase):
    def test_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            res = os.path.join(d, "r.json")
            with open(res, "w", encoding="utf-8") as fh:
                json.dump({}, fh)
            out = os.path.join(d, "out.md")
            with open(out, "w", encoding="utf-8") as fh:
                fh.write("keep")
            with self.assertRaises(FileExistsError):
                t.write_once(out, "new text")
            with open(out, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "keep")


if __name__ == "__main__":
    unittest.main()
