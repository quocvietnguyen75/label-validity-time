"""Synthetic tests for careport.xfc7_t3_run (T3 protocol v1.0 draft, sections 9-11), written before the module."""
import json
import math
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from careport import xfc7_t3_run as t3
from careport.xfc7_t2_posthoc_r4 import cr2_bm_weighted_mean
from careport.xfc7_t2_run import _reweight, add_theta_hat, verdicts_t2
from careport.xfc7_t2_static import capacity, predicted_halfwidth

NAN = float("nan")


def iv(lo, hi):
    return ((lo + hi) / 2, lo, hi)


class ThreeMethodRuleTest(unittest.TestCase):
    def test_camera_rejects_only_if_all_three_reject(self):
        out = {"cr1": iv(40, 50), "cr2_bm": iv(31, 60), "bootstrap": iv(35, 55)}
        self.assertTrue(t3.camera_rejects(out, budget=25.0))           # band 30: all three above it
        out["cr2_bm"] = iv(29, 60)
        self.assertFalse(t3.camera_rejects(out, budget=25.0))          # one method reaches inside the band
        neg = {m: iv(-60, -40) for m in t3.METHODS}
        self.assertTrue(t3.camera_rejects(neg, budget=25.0))           # below -band counts as well

    def test_undefined_interval_never_rejects(self):
        out = {"cr1": iv(40, 50), "cr2_bm": (45.0, NAN, NAN), "bootstrap": iv(35, 55)}
        self.assertFalse(t3.camera_rejects(out, budget=25.0))

    def test_per_camera_reading_differs_from_per_method(self):
        # section 10's example: CR1 rejects in SL and RL, CR2-BM in SL and FL, the bootstrap in all three
        cams = ["FL", "SL", "RL"]
        budgets = {c: 20.0 for c in cams}                               # band 25
        inside, outside = iv(10, 40), iv(30, 40)
        by_method = {"cr1": {"start": {"FL": inside, "SL": outside, "RL": outside}},
                     "cr2_bm": {"start": {"FL": outside, "SL": outside, "RL": inside}},
                     "bootstrap": {"start": {"FL": outside, "SL": outside, "RL": outside}}}
        d = t3.decide_three(by_method, budgets, cams)
        self.assertEqual(d["start"]["rejected_in"], ["SL"])
        self.assertFalse(d["start"]["ruled_out"])
        self.assertEqual({m: len(v) for m, v in d["start"]["rejected_in_by_method"].items()},
                         {"cr1": 2, "cr2_bm": 2, "bootstrap": 3})         # per method, each would rule it out

    def test_two_cameras_under_all_three_rule_out(self):
        cams = ["SL", "RL"]
        by_method = {m: {"start": {"SL": iv(30, 40), "RL": iv(-50, -30)}} for m in t3.METHODS}
        d = t3.decide_three(by_method, {"SL": 20.0, "RL": 20.0}, cams)
        self.assertTrue(d["start"]["ruled_out"])
        self.assertEqual(d["start"]["rejected_in"], ["SL", "RL"])


class IntervalTest(unittest.TestCase):
    def test_all_three_undefined_below_two_logs(self):
        y, w = np.array([1.0, 2.0, 3.0]), np.ones(3)
        out = t3.intervals_three(y, w, np.array(["a", "a", "a"]), draws=99)
        for m in t3.METHODS:
            self.assertTrue(math.isnan(out[m]["lo"]) and math.isnan(out[m]["hi"]), m)
            self.assertAlmostEqual(out[m]["est"], 2.0)

    def test_methods_share_the_point_estimate_and_cover_it(self):
        rng = np.random.default_rng(1)
        cl = np.repeat(np.arange(12), 5)
        y, w = rng.normal(10, 3, cl.size), rng.uniform(0.5, 2, cl.size)
        out = t3.intervals_three(y, w, cl, draws=999, seed=7)
        est = float((w * y).sum() / w.sum())
        for m in t3.METHODS:
            self.assertAlmostEqual(out[m]["est"], est, places=9)
            self.assertLess(out[m]["lo"], est)
            self.assertGreater(out[m]["hi"], est)


class DesignTest(unittest.TestCase):
    def test_nu_bm_is_y_free(self):
        rng = np.random.default_rng(2)
        cl = np.repeat(np.arange(9), [1, 2, 3, 4, 5, 6, 7, 8, 30])
        w = rng.uniform(0.1, 5, cl.size)
        a = cr2_bm_weighted_mean(rng.normal(0, 1, cl.size), w, cl)["df"]
        b = cr2_bm_weighted_mean(rng.normal(50, 20, cl.size), w, cl)["df"]
        self.assertAlmostEqual(a, b, places=12)
        self.assertAlmostEqual(t3.nu_bm(w, cl), a, places=12)

    def test_equal_log_weights_give_g_minus_1_and_t2_halfwidth(self):
        cl = np.repeat(np.arange(10), 4)
        w = np.ones(cl.size)
        self.assertAlmostEqual(t3.nu_bm(w, cl), 9.0, places=9)
        self.assertAlmostEqual(t3.hw_bm(3.0, 4.0, 9.0), predicted_halfwidth(3.0, 4.0, 10), places=9)

    def test_concentrated_weights_lower_nu_and_widen_hw(self):
        cl = np.repeat(np.arange(10), 4)
        w = np.ones(cl.size)
        w[cl == 0] = 50.0
        nu = t3.nu_bm(w, cl)
        self.assertLess(nu, 9.0)
        self.assertGreater(t3.hw_bm(3.0, 4.0, nu), predicted_halfwidth(3.0, 4.0, 10))

    def test_design_row_shares(self):
        g = pd.DataFrame({"segment": ["s1"] * 3 + ["s2"] * 3 + ["s3"] * 2,
                          "object_id": ["a", "a", "b", "c", "d", "e", "f", "g"],
                          "w": [4.0, 4.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0],
                          "sdot": 1.0, "sigma_b": 2.0})
        r = t3.design_row(g)
        self.assertEqual((r["tracks"], r["logs"]), (7, 3))
        self.assertAlmostEqual(r["top_log_share"], 10.0 / 16.0)
        self.assertAlmostEqual(r["top5_track_share"], (8 + 2 + 2 + 1 + 1) / 16.0)
        a = np.array([10.0, 3.0, 3.0]) / 16.0
        self.assertAlmostEqual(r["kish_logs"], 1.0 / (a * a).sum())
        self.assertAlmostEqual(r["V_ind"], (g["w"] ** 2 * 4.0).sum() / 16.0 ** 2)

    def test_each_run_gets_its_own_theta_bar(self):
        # T2's t2_margin_capacity.py: the excluded run's predicted theta-bar averages the label-side predictions under
        # that run's weights; the primary run keeps the S3 summary's
        rows = []
        for seg in range(4):
            for obj in range(2):
                for fr in range(2):
                    low = obj == 1 and fr == 0
                    rows.append({"segment": f"s{seg}", "object_id": f"o{obj}", "camera": 2, "frame": fr,
                                 "margin": 0.1 if low else 0.5, "sdot": 1.0 + obj, "sigma_b": 2.0,
                                 "theta_start": 0.0, "theta_A": 70.0 if low else 30.0, "theta_B": 80.0, "theta_mid": 50.0})
        mo = _reweight(pd.DataFrame(rows))
        des = t3.design_table(mo, {2: "ring_front_left"}, ["ring_front_left"])
        p = des["primary"]["ring_front_left"]["theta_bar"]
        e = des["margin_flag_excluded"]["ring_front_left"]["theta_bar"]
        self.assertAlmostEqual(p["A"], float(np.average(mo["theta_A"], weights=mo["w"])))
        self.assertAlmostEqual(e["A"], 30.0)                      # the flagged rows carried theta_A = 70
        self.assertGreater(p["A"], 30.0)
        s3 = {"cameras": {"ring_front_left": {"predicted_theta_bar": dict(p, A=p["A"] + 1.0), "DE": 1.0}}}
        g = t3.gates_both(s3, des, ["ring_front_left"], {"ring_front_left": 10.0})
        self.assertAlmostEqual(g["primary"]["capacity"]["cameras"]["ring_front_left"]["start-A"]["gap"], p["A"] + 1.0)
        self.assertAlmostEqual(g["margin_flag_excluded"]["capacity"]["cameras"]["ring_front_left"]["start-A"]["gap"], 30.0)
        hw_e = t3.hw_bm(1.0, des["margin_flag_excluded"]["ring_front_left"]["V_ind"],
                        des["margin_flag_excluded"]["ring_front_left"]["nu_bm"])
        self.assertAlmostEqual(g["margin_flag_excluded"]["thresholds_ms"]["ring_front_left"], 15.0 + hw_e)

    def test_theta_bar_skips_missing_predictions_as_the_static_step(self):
        g = pd.DataFrame({"w": [1.0, 3.0, 2.0], "theta_start": 0.0, "theta_A": [10.0, np.nan, 40.0],
                          "theta_B": np.nan, "theta_mid": 50.0})
        tb = t3.theta_bar(g)
        self.assertAlmostEqual(tb["A"], (10.0 + 80.0) / 3.0)
        self.assertTrue(math.isnan(tb["B"]))

    def test_gate_bm_thresholds_and_decidability(self):
        theta = {"X": {"start": 0.0, "A": 60.0, "B": 10.0, "mid": 50.0},
                 "Y": {"start": 0.0, "A": 40.0, "B": 20.0, "mid": 50.0}}
        g = t3.gate_bm(theta, budgets={"X": 30.0, "Y": 30.0}, de={"X": 1.0, "Y": 1.0}, v={"X": 4.0, "Y": 4.0},
                       nu={"X": 9.0, "Y": 2.0})
        tx = 5 + 30 + float(t3.stats.t.ppf(0.975, 9.0)) * 2.0
        ty = 5 + 30 + float(t3.stats.t.ppf(0.975, 2.0)) * 2.0
        self.assertAlmostEqual(g["thresholds_ms"]["X"], tx)
        self.assertAlmostEqual(g["thresholds_ms"]["Y"], ty)
        self.assertTrue(g["capacity"]["cameras"]["X"]["start-A"]["decidable"])     # 60 > 39.5
        self.assertFalse(g["capacity"]["cameras"]["Y"]["start-A"]["decidable"])    # 40 < 43.6 (t with 2 df)
        self.assertTrue(40.0 > 5 + 30 + float(t3.stats.t.ppf(0.975, 9.0)) * 2.0)      # Y would clear it with 9 df
        self.assertFalse(g["capacity"]["dataset"]["start-A"])                        # one camera only

    def test_composition_budgets_feed_the_rule(self):
        comp = {"X": {"neighbour_bins_ms": 3.0, "ego_class_spread_ms": 4.0, "ego_speed_slope_ms": 0.0}}
        b = t3.composition_budgets(comp)
        self.assertEqual((b["X"]["max"], b["X"]["rss"], b["X"]["sum"]), (4.0, 5.0, 7.0))

    def test_decidable_needs_two_cameras_in_both_runs(self):
        # section 9 (T3): a contrast is decidable only if two cameras clear its hw_BM threshold in each run
        theta = {c: {"start": 0.0, "A": 60.0, "B": 10.0, "mid": 50.0} for c in ("X", "Y", "Z")}
        one = dict(de={c: 1.0 for c in theta}, v={c: 4.0 for c in theta})
        prim = t3.gate_bm(theta, {"X": 30.0, "Y": 30.0, "Z": 60.0}, nu={c: 9.0 for c in theta}, **one)
        excl = t3.gate_bm(theta, {"X": 30.0, "Y": 60.0, "Z": 60.0}, nu={c: 9.0 for c in theta}, **one)
        d = t3.decidable_both({"primary": prim, "margin_flag_excluded": excl}, ["X", "Y", "Z"])
        self.assertTrue(d["runs"]["primary"]["start-A"])                 # X and Y clear 60 > 39.5; Z's 69.5 does not
        self.assertFalse(d["runs"]["margin_flag_excluded"]["start-A"])   # only X does
        self.assertFalse(d["decidable"]["start-A"])
        both = t3.decidable_both({"primary": prim, "margin_flag_excluded": prim}, ["X", "Y", "Z"])
        self.assertTrue(both["decidable"]["start-A"])
        # a camera left out (a timing failure) or missing from a run does not count
        self.assertFalse(t3.decidable_both({"primary": prim, "margin_flag_excluded": prim}, ["X", "Z"])["decidable"]["start-A"])
        self.assertFalse(t3.decidable_both({"primary": prim, "margin_flag_excluded": prim}, ["X", "W"])["decidable"]["start-A"])


class ShiftTest(unittest.TestCase):
    def _sets(self):
        ref = pd.DataFrame({"segment": "s", "frame": [1, 1, 2], "camera": 1,
                            "camera_object_id": ["s/1/1/0", "s/1/1/1", "s/2/1/0"],
                            "u": [100.0, 300.0, 120.0], "v": [50.0, 60.0, 55.0], "w": 40.0})
        mo = pd.DataFrame({"segment": "s", "frame": [1, 2], "camera": 1, "object_id": ["o", "o"],
                           "camera_object_id": ["s/1/1/0", "s/2/1/0"], "u_ref": [100.0, 120.0], "v_ref": [50.0, 55.0],
                           "d_u": [0.6, 0.6], "d_v": [0.8, 0.8], "sdot": [2.0, 2.0], "p0_u": [110.0, 130.0],
                           "p0_v": [52.0, 57.0], "s": [1.0, 1.0], "z_b": [0.0, 0.0], "w": [1.0, 1.0],
                           "theta_start": 0.0})
        return ref, mo

    def test_shift_moves_only_matched_boxes_by_minus_delta_sdot_dhat(self):
        ref, mo = self._sets()
        out = t3.shift_reference(ref, mo, 10.0)
        self.assertEqual(list(out.columns), list(ref.columns))
        np.testing.assert_allclose(out["u"], [100.0 - 10 * 2 * 0.6, 300.0, 120.0 - 10 * 2 * 0.6])
        np.testing.assert_allclose(out["v"], [50.0 - 10 * 2 * 0.8, 60.0, 55.0 - 10 * 2 * 0.8])
        pd.testing.assert_series_equal(out["w"], ref["w"])

    def test_recovery_with_a_static_step_that_keeps_assignments(self):
        ref, mo = self._sets()

        def fake_static(lab, boxes):
            m = mo.drop(columns=["u_ref", "v_ref"]).merge(
                boxes[["camera_object_id", "u", "v"]].rename(columns={"u": "u_ref", "v": "v_ref"}), on="camera_object_id")
            return {"mo": m, "assoc": m[["segment", "frame", "camera", "object_id", "camera_object_id"]]}

        res = t3.e4_av2(None, ref, mo, cams=["ring_front_center"], names={1: "ring_front_center"},
                        deltas=(-20.0, 10.0), static_fn=fake_static)
        for k in ("-20ms", "+10ms"):
            self.assertAlmostEqual(res["runs"][k]["cameras"]["ring_front_center"]["recovery_ratio"], 1.0, places=9)
            self.assertEqual(res["runs"][k]["assignment_changed_share"], 0.0)
        self.assertTrue(res["baseline_reproduced"])

    def test_zero_shift_check_ignores_rows_of_other_cameras(self):
        # run_reference_t2 returns every camera's moving rows; the baseline holds the decision cameras only
        ref, mo = self._sets()
        other = mo.assign(camera=3, camera_object_id=["s/1/3/0", "s/2/3/0"])
        ref = pd.concat([ref, ref.iloc[[0, 2]].assign(camera=3, camera_object_id=["s/1/3/0", "s/2/3/0"])], ignore_index=True)

        def fake_static(lab, boxes):
            full = pd.concat([mo, other], ignore_index=True)
            m = full.drop(columns=["u_ref", "v_ref"]).merge(
                boxes[["camera_object_id", "u", "v"]].rename(columns={"u": "u_ref", "v": "v_ref"}), on="camera_object_id")
            return {"mo": m, "assoc": m[["segment", "frame", "camera", "object_id", "camera_object_id"]]}

        res = t3.e4_av2(None, ref, mo, cams=["ring_front_center"], names={1: "ring_front_center", 3: "ring_front_right"},
                        deltas=(10.0,), static_fn=fake_static)
        self.assertTrue(res["baseline_reproduced"])
        self.assertEqual(list(res["runs"]["+10ms"]["cameras"]), ["ring_front_center"])


class AnalysisTest(unittest.TestCase):
    def test_a_decision_camera_without_rows_is_skipped_not_fatal(self):
        # T2's P01 subgroup has no rear-right rows; the half-width, gate and composition tables must skip such a camera
        rng = np.random.default_rng(4)
        rows = []
        for seg in range(6):
            for obj in range(3):
                for fr in range(2):
                    rows.append({"segment": f"s{seg}", "object_id": f"o{obj}", "camera": 2, "frame": fr,
                                 "margin": 0.5, "sdot": 1.0, "sigma_b": 2.0, "p0_u": rng.normal(10, 2), "p0_v": 0.0,
                                 "u_ref": 0.0, "v_ref": 0.0, "d_u": 1.0, "d_v": 0.0, "s": 0.0, "z_b": 0.0,
                                 "theta_start": 0.0, "theta_A": 30.0, "theta_B": 80.0, "theta_mid": 50.0})
        mo = _reweight(pd.DataFrame(rows))
        cam = {"predicted_theta_bar": {"start": 0.0, "A": 30.0, "B": 80.0, "mid": 50.0}, "DE": 1.0,
               "predicted_halfwidth_ms": 2.0}
        comp = {"neighbour_bins_ms": 3.0, "ego_class_spread_ms": 4.0, "ego_speed_slope_ms": 1.0, "B_c_ms": 4.0}
        s3 = {"decision_cameras": ["ring_front_left", "ring_rear_right"],
              "budget": {"ring_front_left": comp, "ring_rear_right": comp},
              "cameras": {"ring_front_left": cam, "ring_rear_right": cam}}
        out = t3.analysis(mo, s3, components=s3["budget"], draws=99, with_block=False)
        self.assertEqual(list(out["halfwidths"]), ["ring_front_left"])
        self.assertNotIn("ring_rear_right", out["three_method"]["intervals"]["primary"])
        self.assertEqual(list(out["compositions"]["rss"]["budgets_ms"]), ["ring_front_left"])
        # decidability under the hw_BM gate, both runs, for the primary budgets and each composition (S-a)
        self.assertEqual(set(out["decidable_bm"]), {"all_decision_cameras", "after_timing"})
        self.assertFalse(any(out["decidable_bm"]["after_timing"]["decidable"].values()))   # one camera with rows
        for name in ("rss", "sum"):
            self.assertEqual(set(out["compositions"][name]["gates"]), {"primary", "margin_flag_excluded"})
            self.assertIn("start-B", out["compositions"][name]["decidable_bm"]["decidable"])


class CR1PathTest(unittest.TestCase):
    def test_cr1_by_method_final_matches_frozen_verdicts_t2(self):
        rng = np.random.default_rng(3)
        rows = []
        for cam in (2, 4, 6):
            for seg in range(8):
                for obj in range(3):
                    for fr in range(2):
                        rows.append({"segment": f"s{seg}", "object_id": f"o{obj}", "camera": cam, "frame": fr,
                                     "theta_hat": rng.normal(45 if cam != 2 else 5, 4), "theta_start": 0.0,
                                     "theta_A": 30.0, "theta_B": 80.0, "theta_mid": 50.0,
                                     "sdot": rng.uniform(0.5, 2), "sigma_b": 3.0,
                                     "margin": rng.choice([0.1, 0.5, np.inf])})
        mo = pd.DataFrame(rows)
        mo["n_track"] = mo.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
        mo["w"] = mo["sdot"] ** 2 / (mo["sigma_b"] ** 2 * mo["n_track"])
        names = {2: "ring_front_left", 4: "ring_side_left", 6: "ring_rear_left"}
        budgets = {n: 20.0 for n in names.values()}
        frozen = verdicts_t2(mo, names, budgets)
        hyp = {"start": "theta_start", "A": "theta_A", "B": "theta_B", "mid": "theta_mid"}
        runs = {"primary": t3.cells(mo, names, hyp, list(budgets)),
                "margin_flag_excluded": t3.cells(_reweight(mo[(mo["margin"] >= 0.2) | np.isinf(mo["margin"])]),
                                                 names, hyp, list(budgets))}
        v = t3.verdicts_three(runs, budgets, list(budgets), draws=199)
        self.assertEqual(v["by_method"]["cr1"]["final"], frozen["final"])
        for h in hyp:                                              # the three-method rule is never looser than CR1
            if v["three"]["final"][h] == "ruled out":
                self.assertEqual(frozen["final"][h], "ruled out")


# ---------- added 4 Nov 2026 after the code audit (written before the code they test) ----------
class CombineRunsTest(unittest.TestCase):
    def test_primary_only_rejection_is_association_sensitive(self):
        rng = np.random.default_rng(8)
        cl = np.repeat(np.arange(8), 3)
        w = np.ones(cl.size)
        far = lambda: (45 + rng.normal(0, 0.5, cl.size), w, cl)          # noqa: E731
        near = lambda: (5 + rng.normal(0, 0.5, cl.size), w, cl)          # noqa: E731
        runs = {"primary": {"X": {"start": far()}, "Y": {"start": far()}},
                "margin_flag_excluded": {"X": {"start": far()}, "Y": {"start": near()}}}
        v = t3.verdicts_three(runs, {"X": 20.0, "Y": 20.0}, ["X", "Y"], draws=199)
        self.assertTrue(v["three"]["runs"]["primary"]["start"]["ruled_out"])
        self.assertFalse(v["three"]["runs"]["margin_flag_excluded"]["start"]["ruled_out"])
        self.assertEqual(v["three"]["final"]["start"], "association-sensitive, not ruled out")
        both = {"primary": runs["primary"], "margin_flag_excluded": runs["primary"]}
        self.assertEqual(t3.verdicts_three(both, {"X": 20.0, "Y": 20.0}, ["X", "Y"], draws=199)["three"]["final"]["start"],
                         "ruled out")

    def test_undefined_intervals_are_listed_and_never_reject(self):
        cl = np.repeat(np.arange(6), 3)
        y = 45 + np.linspace(0, 1, cl.size)
        one_log = (np.array([80.0, 81.0]), np.ones(2), np.array(["a", "a"]))
        runs = {"primary": {"X": {"start": (y, np.ones(cl.size), cl)}, "Z": {"start": one_log}},
                "margin_flag_excluded": {"X": {"start": (y, np.ones(cl.size), cl)}, "Z": {"start": one_log}}}
        v = t3.verdicts_three(runs, {"X": 20.0, "Z": 20.0}, ["X", "Z"], draws=199)
        self.assertEqual([u for u in v["undefined_intervals"] if u[0] == "primary" and u[1] == "Z"],
                         [["primary", "Z", "start", m] for m in sorted(t3.METHODS)])
        self.assertEqual(v["three"]["runs"]["primary"]["start"]["rejected_in"], ["X"])


class FailureLabelTest(unittest.TestCase):
    def test_capacity_estimate_and_undefined_labels(self):
        def iv3(est, lo, hi):
            return {"est": est, "lo": lo, "hi": hi}
        three = {"intervals": {"primary": {
            "X": {"start": {"cr1": iv3(50, 45, 55), "cr2_bm": iv3(50, 30, 70), "bootstrap": iv3(50, 40, 60)}},
            "Y": {"start": {m: iv3(20, 15, 25) for m in t3.METHODS}},
            "Z": {"start": {m: iv3(50, 45, 55) for m in t3.METHODS}},
            "W": {"start": {m: iv3(50, NAN, NAN) for m in t3.METHODS}},
            "V": {"start": {"cr1": iv3(50, 45, 55), "cr2_bm": iv3(50, NAN, NAN), "bootstrap": iv3(50, 45, 55)}}}}}
        lab = t3.failure_labels(three, {c: 30.0 for c in "XYZWV"}, {"primary": {"X": 6.0}})["primary"]
        x = lab["X"]["start"]
        self.assertEqual(x["three"], "capacity failure")                 # |50| > 5 + 30, but CR2 reaches inside
        self.assertEqual(x["by_method"]["cr1"]["label"], "rejected")
        self.assertEqual(x["by_method"]["cr2_bm"]["label"], "capacity failure")
        self.assertAlmostEqual(x["by_method"]["cr2_bm"]["realised_hw_ms"], 20.0)
        self.assertTrue(x["by_method"]["cr2_bm"]["realised_exceeds_hw_bm"])    # 20 > 6
        self.assertFalse(x["by_method"]["cr1"]["realised_exceeds_hw_bm"])      # 5 < 6
        self.assertEqual(x["predicted_hw_bm_ms"], 6.0)
        self.assertEqual(lab["Y"]["start"]["three"], "estimate failure")         # |20| <= 35
        self.assertEqual(lab["Z"]["start"]["three"], "rejected")
        self.assertEqual(lab["W"]["start"]["three"], "undefined interval")
        self.assertEqual(lab["V"]["start"]["three"], "undefined interval")
        self.assertEqual(lab["V"]["start"]["by_method"]["cr2_bm"]["label"], "undefined interval")
        self.assertIsNone(lab["Y"]["start"]["by_method"]["cr1"]["realised_exceeds_hw_bm"])   # no prediction for Y
        self.assertNotIn("camera_clears", x)                                     # no capacity table given

    def test_labels_say_which_contrasts_the_camera_clears(self):
        # section 10 defines the failures for decidable contrasts; each label carries the camera's capacity flags
        ints = {m: {"est": 50.0, "lo": 30.0, "hi": 70.0} for m in t3.METHODS}
        three = {"intervals": {"primary": {"X": {"start": ints, "A": ints}}}}
        cap = {"primary": {"X": {"start-A": {"decidable": True}, "start-B": {"decidable": False},
                                 "start-mid": {"decidable": True}, "A-B": {"decidable": False},
                                 "A-mid": {"decidable": False}, "B-mid": {"decidable": True}}}}
        lab = t3.failure_labels(three, {"X": 30.0}, {}, capacity=cap)["primary"]["X"]
        self.assertEqual(lab["start"]["camera_clears"], {"start-A": True, "start-B": False, "start-mid": True})
        self.assertEqual(lab["A"]["camera_clears"], {"start-A": True, "A-B": False, "A-mid": False})


NAMES3 = {2: "ring_front_left", 4: "ring_side_left", 6: "ring_rear_left"}


def full_set(late_b=(), flagged_shift=0.0, n_logs=8, seed=5):
    """Every column av2_block needs; theta-hat = p0_u / sdot (d-hat (1, 0), u_ref = s = z_b = 0). sdot varies by track
    and theta_A, theta_B by frame (+-2 ms, equal weights within a track), so the weighted theta-bar is exactly 30 and
    80 ms and T2's slope fits are not singular. Cameras in `late_b` get +60 ms on their B-frame rows (a two-frame
    failure); the margin-0.1 rows (o0, frame 0) get `flagged_shift`."""
    rng = np.random.default_rng(seed)
    rows = []
    for cam in NAMES3:
        for seg in range(n_logs):
            for obj, sdot in enumerate((0.8, 1.0, 1.2)):
                for fr, ft, k in ((0, "A", 2.0), (1, "B", -2.0)):
                    flagged = obj == 0 and fr == 0
                    th = rng.normal(45.0, 1.0) + (60.0 if cam in late_b and ft == "B" else 0.0) + \
                        (flagged_shift if flagged else 0.0)
                    rows.append({"segment": f"s{seg}", "object_id": f"o{obj}", "camera": cam, "frame": fr,
                                 "frame_type": ft, "margin": 0.1 if flagged else 0.5, "sdot": sdot, "sigma_b": 2.0,
                                 "p0_u": th * sdot, "p0_v": 0.0, "u_ref": 0.0, "v_ref": 0.0, "d_u": 1.0, "d_v": 0.0,
                                 "s": 0.0, "z_b": 0.0, "speed": rng.uniform(5.0, 15.0),
                                 "theta_start": 0.0, "theta_A": 30.0 + k, "theta_B": 80.0 - k, "theta_mid": 50.0})
    return _reweight(pd.DataFrame(rows))


def s3_for(mo, budgets, components=None):
    """A minimal S3 summary for full_set: predicted theta-bar, DE 1, and the static step's t(G - 1) capacity."""
    cams, thr = {}, {}
    for c, name in NAMES3.items():
        g = mo[mo["camera"] == c]
        hw = predicted_halfwidth(1.0, t3.v_ind(g), g["segment"].nunique())
        cams[name] = {"predicted_theta_bar": t3.theta_bar(g), "DE": 1.0, "predicted_halfwidth_ms": hw}
        thr[name] = 5.0 + budgets[name] + hw
    comp = components or {n: {"neighbour_bins_ms": b, "ego_class_spread_ms": 0.0, "ego_speed_slope_ms": 0.0}
                          for n, b in budgets.items()}
    return {"decision_cameras": sorted(NAMES3.values()), "cameras": cams, "thresholds_ms": thr,
            "budget": {n: dict(comp[n], B_c_ms=budgets[n]) for n in budgets},
            "capacity": capacity({n: cams[n]["predicted_theta_bar"] for n in cams}, thr)}


class FullAnalysisTest(unittest.TestCase):
    """analysis() with T2's av2_block (two-frame check, C2, pi-hat, C3) on full synthetic moving sets."""

    def test_a_timing_failure_leaves_the_rule_decidability_and_c2(self):
        mo = full_set(late_b=(4,))
        s3 = s3_for(mo, {"ring_front_left": 10.0, "ring_side_left": 10.0, "ring_rear_left": 100.0})
        out = t3.analysis(mo, s3, components=s3["budget"], draws=199)
        self.assertEqual(out["t2_block"]["timing_model_failures"], ["ring_side_left"])
        for tag in t3.RUNS:
            self.assertNotIn("ring_side_left", out["three_method"]["intervals"][tag])
        self.assertNotIn("ring_side_left", out["compositions"]["rss"]["budgets_ms"])
        self.assertTrue(out["decidable_bm"]["all_decision_cameras"]["decidable"]["start-B"])   # front-left, side-left
        self.assertFalse(out["decidable_bm"]["after_timing"]["decidable"]["start-B"])          # front-left alone
        self.assertEqual(out["reading"]["h_start_informative_against"], [])
        self.assertFalse(out["reading"]["timing_model_rejected"])
        d = add_theta_hat(mo[mo["camera"] != 4])
        want = cr2_bm_weighted_mean(d["theta_hat"].to_numpy(float), d["w"].to_numpy(float), d["segment"].to_numpy())
        self.assertAlmostEqual(out["C2_pooled_cr2_bm"]["est"], want["est"], places=9)
        self.assertAlmostEqual(out["C2_pooled_cr2_bm"]["hi"], want["hi"], places=9)

    def test_two_failures_reject_the_timing_model(self):
        mo = full_set(late_b=(2, 4))
        out = t3.analysis(mo, s3_for(mo, {n: 10.0 for n in NAMES3.values()}), draws=199)
        self.assertTrue(out["reading"]["timing_model_rejected"])
        self.assertTrue(out["reading"]["all_descriptive"])

    def test_excluded_run_drops_flagged_rows_and_reweights(self):
        mo = full_set(flagged_shift=30.0)
        out = t3.analysis(mo, s3_for(mo, {n: 10.0 for n in NAMES3.values()}), draws=199)
        g = add_theta_hat(mo[mo["camera"] == 2])
        keep = _reweight(g[g["margin"] >= 0.2])
        want_e = float(np.average(keep["theta_hat"] - keep["theta_start"], weights=keep["w"]))
        want_p = float(np.average(g["theta_hat"] - g["theta_start"], weights=g["w"]))
        iv = out["three_method"]["intervals"]
        self.assertAlmostEqual(iv["margin_flag_excluded"]["ring_front_left"]["start"]["cr1"]["est"], want_e, places=9)
        self.assertAlmostEqual(iv["primary"]["ring_front_left"]["start"]["cr1"]["est"], want_p, places=9)
        self.assertGreater(want_p - want_e, 1.0)

    def test_compositions_use_their_own_budgets_and_readings(self):
        mo = full_set()
        comp = {n: {"neighbour_bins_ms": 10.0, "ego_class_spread_ms": 25.0, "ego_speed_slope_ms": 20.0}
                for n in NAMES3.values()}
        s3 = s3_for(mo, {n: 25.0 for n in NAMES3.values()}, components=comp)
        out = t3.analysis(mo, s3, components=s3["budget"], draws=199)
        self.assertEqual(out["three_method"]["three"]["final"]["start"], "ruled out")                  # 45 vs 5 + 25
        self.assertEqual(out["compositions"]["rss"]["verdicts"]["three"]["final"]["start"], "ruled out")  # vs 5 + 33.5
        self.assertEqual(out["compositions"]["sum"]["verdicts"]["three"]["final"]["start"], "not ruled out")  # 5 + 55
        self.assertAlmostEqual(out["compositions"]["sum"]["budgets_ms"]["ring_front_left"], 55.0)
        self.assertEqual(out["reading"]["h_start_informative_against"], ["B", "mid"])
        self.assertIsNone(out["compositions"]["sum"]["reading"]["label"])
        self.assertEqual(out["compositions"]["sum"]["reading"]["h_start_informative_against"], ["B"])
        self.assertEqual(out["failure_labels"]["primary"]["ring_front_left"]["start"]["three"], "rejected")
        self.assertEqual(out["failure_labels"]["primary"]["ring_front_left"]["A"]["three"], "estimate failure")
        clears = out["failure_labels"]["primary"]["ring_front_left"]["start"]["camera_clears"]
        self.assertEqual(clears, {"start-A": False, "start-B": True, "start-mid": True})   # gap 30 < 30.7; 80, 50 clear

    def test_a_composition_with_nothing_decidable_reads_undecidable(self):
        mo = full_set()
        comp = {n: {"neighbour_bins_ms": 10.0, "ego_class_spread_ms": 25.0, "ego_speed_slope_ms": 80.0}
                for n in NAMES3.values()}
        s3 = s3_for(mo, {n: 80.0 for n in NAMES3.values()}, components=comp)
        out = t3.analysis(mo, s3, components=s3["budget"], draws=199)
        self.assertEqual(out["compositions"]["sum"]["reading"]["label"], "undecidable under B_sum")
        self.assertTrue(out["compositions"]["sum"]["reading"]["undecidable"])
        self.assertEqual(out["compositions"]["sum"]["verdicts"]["three"]["final"]["start"], "not ruled out")  # kept

    def test_subgroups_are_descriptive_throughout(self):
        mo = full_set()
        sub = t3.subgroup_analyses(mo, s3_for(mo, {n: 10.0 for n in NAMES3.values()}),
                                   {"first": ["s0", "s1", "s2", "s3"]}, draws=199)["first"]
        three = sub["three_method"]
        self.assertNotIn("final", three["three"])
        self.assertIn("final_descriptive", three["three"])
        for m in t3.METHODS:
            self.assertNotIn("final", three["by_method"][m])
            self.assertIn("final_descriptive", three["by_method"][m])
        self.assertEqual(sorted({s for s in mo["segment"]} & {"s0", "s1", "s2", "s3"}), ["s0", "s1", "s2", "s3"])
        self.assertEqual(three["intervals"]["primary"]["ring_front_left"]["start"]["cr1"]["G"], 4)


class E4PassBandTest(unittest.TestCase):
    def test_pass_band_fails_if_a_camera_loses_its_rows(self):
        ref = pd.DataFrame({"segment": "s", "frame": [1, 2, 1, 2], "camera": [1, 1, 3, 3],
                            "camera_object_id": ["s/1/1/0", "s/2/1/0", "s/1/3/0", "s/2/3/0"],
                            "u": [100.0, 120.0, 100.0, 120.0], "v": [50.0, 55.0, 50.0, 55.0], "w": 40.0})
        base = pd.DataFrame({"segment": "s", "frame": [1, 2, 1, 2], "camera": [1, 1, 3, 3], "object_id": "o",
                             "camera_object_id": ["s/1/1/0", "s/2/1/0", "s/1/3/0", "s/2/3/0"],
                             "u_ref": [100.0, 120.0, 100.0, 120.0], "v_ref": [50.0, 55.0, 50.0, 55.0],
                             "d_u": 0.6, "d_v": 0.8, "sdot": 2.0, "p0_u": [110.0, 130.0, 110.0, 130.0],
                             "p0_v": [52.0, 57.0, 52.0, 57.0], "s": 1.0, "z_b": 0.0, "w": 1.0, "theta_start": 0.0})

        def fake_static(lab, boxes):
            moved = not np.allclose(boxes["u"].to_numpy(float), ref["u"].to_numpy(float))
            full = base[base["camera"] == 1] if moved else base          # camera 3 loses its rows once boxes move
            m = full.drop(columns=["u_ref", "v_ref"]).merge(
                boxes[["camera_object_id", "u", "v"]].rename(columns={"u": "u_ref", "v": "v_ref"}), on="camera_object_id")
            return {"mo": m, "assoc": m[["segment", "frame", "camera", "object_id", "camera_object_id"]]}

        res = t3.e4_av2(None, ref, base, cams=["ring_front_center", "ring_front_right"],
                        names={1: "ring_front_center", 3: "ring_front_right"}, deltas=(10.0,), static_fn=fake_static)
        self.assertTrue(res["baseline_reproduced"])
        self.assertAlmostEqual(res["runs"]["+10ms"]["cameras"]["ring_front_center"]["recovery_ratio"], 1.0, places=9)
        self.assertEqual(res["runs"]["+10ms"]["missing_cameras"], ["ring_front_right"])
        self.assertFalse(res["pass_band_90_110"])


class MainRefusalTest(unittest.TestCase):
    """main() stops before reading any analysis input on each of its refusals (section 15)."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.root = os.path.join(self.d, "root")
        os.makedirs(self.root)
        self.data = os.path.join(self.root, "x.json")
        with open(self.data, "w", encoding="utf-8") as f:
            f.write("{}")
        self.proto = os.path.join(self.d, "P_v1.md")
        with open(self.proto, "w", encoding="utf-8") as f:
            f.write("protocol")
        self.out = os.path.join(self.d, "out.json")

    def cfg(self, **over):
        c = {"protocol": "P_v1.md", "inputs": [{"what": "x.json", "path": "$AV2_SENSOR_ROOT/x.json", "sha256": t3.sha(self.data)}],
             "analyses": {"a": {"reads": {"s3_summary": "$AV2_SENSOR_ROOT/x.json"}, "expected": {}}},
             "comparison": {}, "bootstrap": {"draws": t3.DRAWS, "seed": t3.SEED, "weights": "webb6"}}
        c.update(over)
        p = os.path.join(self.d, "cfg.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(c, f)
        return p

    def run_main(self, cfg, cfg_sha=None, proto_sha=None):
        argv = ["xfc7_t3_run", cfg, cfg_sha or t3.sha(cfg), self.proto, proto_sha or t3.sha(self.proto), self.out]
        with mock.patch.dict(os.environ, {"AV2_SENSOR_ROOT": self.root}):
            return t3.main(argv)

    def test_refuses_an_existing_output(self):
        open(self.out, "w").close()
        with self.assertRaisesRegex(SystemExit, "exists"):
            self.run_main(self.cfg())

    def test_refuses_a_changed_config_or_protocol(self):
        with self.assertRaisesRegex(SystemExit, "^config"):
            self.run_main(self.cfg(), cfg_sha="0" * 64)
        with self.assertRaisesRegex(SystemExit, "^protocol"):
            self.run_main(self.cfg(), proto_sha="0" * 64)

    def test_refuses_another_protocol_file(self):
        with self.assertRaisesRegex(SystemExit, "not the one the config names"):
            self.run_main(self.cfg(protocol="P_v2.md"))

    def test_refuses_an_unpinned_read(self):
        reads = {"s3_summary": "$AV2_SENSOR_ROOT/x.json", "moving_set": "$AV2_SENSOR_ROOT/y.parquet"}
        with self.assertRaisesRegex(SystemExit, "not pinned"):
            self.run_main(self.cfg(analyses={"a": {"reads": reads}}))

    def test_refuses_a_changed_input(self):
        cfg = self.cfg()
        with open(self.data, "w", encoding="utf-8") as f:
            f.write('{"changed": 1}')
        with self.assertRaisesRegex(SystemExit, "^x.json"):
            self.run_main(cfg)

    def test_refuses_other_bootstrap_settings(self):
        with self.assertRaisesRegex(SystemExit, "bootstrap"):
            self.run_main(self.cfg(bootstrap={"draws": 999, "seed": t3.SEED, "weights": "webb6"}))
        self.assertFalse(os.path.exists(self.out))


if __name__ == "__main__":
    unittest.main()
