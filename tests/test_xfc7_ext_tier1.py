import unittest

import numpy as np
import pandas as pd

from careport.xfc7_ext_tier1 import classify_pair, copy_bound, e2_pair, overall_pair, wls_intercept_slope_cr1


class ClassifyTest(unittest.TestCase):
    def test_agree_disagree_inconclusive(self):
        self.assertEqual(classify_pair(-3.0, 4.0), "agree")
        self.assertEqual(classify_pair(5.5, 9.0), "disagree")
        self.assertEqual(classify_pair(-12.0, -6.0), "disagree")
        self.assertEqual(classify_pair(2.0, 7.0), "inconclusive")

    def test_overall_needs_two_agreeing_and_no_disagreeing_camera(self):
        self.assertEqual(overall_pair({"A": "agree", "B": "agree", "C": "inconclusive"}), "agree")
        self.assertEqual(overall_pair({"A": "agree", "B": "inconclusive"}), "inconclusive")
        self.assertEqual(overall_pair({"A": "agree", "B": "agree", "C": "disagree"}), "disagree")


class CopyBoundTest(unittest.TestCase):
    def test_upper_ci_end_over_separation_floored_at_zero(self):
        self.assertAlmostEqual(copy_bound(-2.0, 4.9, 49.0), 0.1)
        self.assertEqual(copy_bound(-6.0, -1.0, 40.0), 0.0)


class WlsTest(unittest.TestCase):
    def test_recovers_intercept_and_slope(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(0.05, 0.5, 400)
        y = 2.0 + 30.0 * x
        seg = np.repeat(np.arange(40), 10).astype(str)
        (b0, se0), (b1, se1) = wls_intercept_slope_cr1(y, x, np.ones(400), seg)
        self.assertAlmostEqual(b0, 2.0, places=8)
        self.assertAlmostEqual(b1, 30.0, places=8)
        self.assertLess(se0, 1e-6)


def _moving(tau, seg):
    n = len(tau)
    return pd.DataFrame({"segment": seg, "frame": np.arange(n), "camera": 1, "object_id": [f"o{i % 4}" for i in range(n)],
                         "p0_u": 500.0, "p0_v": 300.0, "u_ref": 500.0 - np.asarray(tau) * 10.0, "v_ref": 300.0,
                         "d_u": 1.0, "d_v": 0.0, "s": 1.0, "z_b": 0.0, "sdot": 10.0, "sigma_b": 2.0})


class E2PairTest(unittest.TestCase):
    def test_paired_difference_of_tau_hat_on_common_rows(self):
        seg = [f"s{i % 6}" for i in range(60)]
        h = _moving(np.full(60, 3.0), seg)
        d = _moving(np.full(60, 1.0), seg).iloc[:50]
        out = e2_pair(h, d, min_tracks=1)
        cam = out["cameras"]["FRONT"]
        self.assertEqual(cam["common_obs"], 50)
        self.assertAlmostEqual(cam["delta_pair"]["est"], 2.0)
        self.assertEqual(cam["decision"], "agree")

    def test_copy_bound_uses_the_common_set_separation(self):
        """EXT v1.0 section 6 (1 Nov): rho's denominator is the E2-weighted mean of tau_lidar - tau_frame on the common
        rows, from the human set's timing columns, not the v2.0 human-set constant."""
        seg = [f"s{i % 6}" for i in range(60)]
        h = _moving(np.full(60, 3.0), seg).assign(tau_frame=0.0, tau_lidar=np.where(np.arange(60) < 50, 40.0, 90.0))
        d = _moving(np.full(60, 1.0), seg).iloc[:50]
        cam = e2_pair(h, d, min_tracks=1)["cameras"]["FRONT"]
        self.assertAlmostEqual(cam["frame_lidar_separation_ms"], 40.0)   # rows 50-59 are not common
        self.assertAlmostEqual(cam["copy_bound_rho"], cam["delta_pair"]["hi"] / 40.0)

    def test_capture_time_seed_bound(self):
        """Audit finding 10: seeds drawn at the labelled position match capture time, so rho is also bounded with the
        common-set tau_cap - tau_frame separation."""
        seg = [f"s{i % 6}" for i in range(60)]
        h = _moving(np.full(60, 3.0), seg).assign(tau_frame=0.0, tau_lidar=40.0, tau_cap=50.0)
        d = _moving(np.full(60, 1.0), seg).iloc[:50]
        cam = e2_pair(h, d, min_tracks=1)["cameras"]["FRONT"]
        self.assertAlmostEqual(cam["frame_capture_separation_ms"], 50.0)
        self.assertAlmostEqual(cam["copy_bound_rho_capture_seed"], cam["delta_pair"]["hi"] / 50.0)


class CheckShaTest(unittest.TestCase):
    """Audit finding 7 (1 Nov): every input the run depends on is checked against a pinned hash."""

    def test_match_returns_hash_and_mismatch_stops(self):
        import hashlib
        import os
        import tempfile
        from careport.xfc7_ext_tier1 import check_sha
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.json")
            open(p, "wb").write(b"abc")
            h = hashlib.sha256(b"abc").hexdigest()
            self.assertEqual(check_sha(p, h, "x"), h)
            with self.assertRaises(SystemExit):
                check_sha(p, "0" * 64, "x")

    def test_pinned_e0_hashes(self):
        from careport import xfc7_ext_tier1 as t
        self.assertEqual(t.E0_PRIMARY_SHA[:8], "12238bd6")
        self.assertEqual(t.E0_SENSITIVITY_SHA[:8], "f8cfd7d8")
        self.assertEqual(t.ASSOC_PRIMARY_SHA[:8], "0951961a")

    def test_wls_is_the_frozen_estimator(self):
        from careport import xfc7_estimators, xfc7_ext_tier1
        self.assertIs(xfc7_ext_tier1.wls_intercept_slope_cr1, xfc7_estimators.wls_intercept_slope_cr1)


def _synthetic_with_margin(truth):
    from test_xfc7_primary import _synthetic
    df = _synthetic(truth)
    df["margin"] = np.inf
    df.loc[df.index[::50], "margin"] = 0.1                       # a few margin-flagged rows
    return df


class E1VerdictsTest(unittest.TestCase):
    def test_lidar_truth_rules_out_frame_in_both_runs(self):
        from careport.xfc7_ext_tier1 import e1_verdicts
        cams = ["FRONT", "FRONT_LEFT", "SIDE_LEFT"]
        e0 = {"detector": {"cameras": {c: {"tracks": 100} for c in cams} | {"SIDE_RIGHT": {"tracks": 18}},
                           "budget": {c: {"B_c_ms": 10.0} for c in cams + ["SIDE_RIGHT"]}}}
        out = e1_verdicts(_synthetic_with_margin("lidar"), e0)
        self.assertEqual(out["cameras_in_analysis"], cams)        # SIDE_RIGHT (< 30 tracks) excluded
        self.assertEqual(out["final_verdicts"]["frame"], "ruled out")
        self.assertEqual(out["final_verdicts"]["lidar"], "not ruled out")
        self.assertFalse(out["capacity_exception"])                 # ruled out, so no exception is needed


class E4PassBandTest(unittest.TestCase):
    def test_pass_needs_every_decision_camera_within_90_to_110_percent(self):
        from careport.xfc7_ext_tier1 import e4_pass
        runs = {"+10ms": {"cameras": {"FRONT": {"recovery_ratio": 0.97}, "SIDE_LEFT": {"recovery_ratio": 1.08},
                                      "SIDE_RIGHT": {"recovery_ratio": 0.5}}}}
        self.assertTrue(e4_pass(runs, ["FRONT", "SIDE_LEFT"]))      # SIDE_RIGHT is not a decision camera
        runs["-20ms"] = {"cameras": {"FRONT": {"recovery_ratio": 0.88}, "SIDE_LEFT": {"recovery_ratio": 1.0}}}
        self.assertFalse(e4_pass(runs, ["FRONT", "SIDE_LEFT"]))


def _res(cams):
    return {"cameras": {c: {"delta": {"frame": {"est": e, "lo": lo, "hi": hi}}} for c, (e, lo, hi) in cams.items()}}


class FailureLabelTest(unittest.TestCase):
    """EXT v1.0 section 10: capacity failure (|est| beyond 5 + B, CI crosses) vs estimate failure (|est| within)."""

    def test_labels_and_point_estimate_rule(self):
        from careport.xfc7_ext_tier1 import frame_failure_labels, frame_ruled_out_at_point_estimates
        res = _res({"A": (30.0, 20.0, 40.0), "B": (30.0, 12.0, 48.0), "C": (10.0, 0.0, 20.0)})
        budgets = {"A": 10.0, "B": 10.0, "C": 10.0}                      # thresholds 15 ms
        self.assertEqual(frame_failure_labels(res, ["A", "B", "C"], budgets),
                         {"A": "rejected", "B": "capacity failure", "C": "estimate failure"})
        self.assertTrue(frame_ruled_out_at_point_estimates(res, ["A", "B", "C"], budgets))
        self.assertFalse(frame_ruled_out_at_point_estimates(res, ["A", "C"], budgets))


class GateTest(unittest.TestCase):
    """EXT v1.0 section 11 with the capacity exception (pinned 1 Nov)."""

    def test_gate_routes(self):
        from careport.xfc7_ext_tier1 import tier2_gate
        agree = {"overall": "agree", "cameras": {"FRONT": {"decision": "agree"}, "SIDE_LEFT": {"decision": "inconclusive"}}}
        incon = {"overall": "inconclusive", "cameras": {"FRONT": {"decision": "inconclusive"}}}
        disag = {"overall": "disagree", "cameras": {"FRONT": {"decision": "agree"}, "FRONT_LEFT": {"decision": "disagree"}}}
        self.assertTrue(tier2_gate(True, False, agree)["pass"])
        self.assertTrue(tier2_gate(True, False, incon)["pass"])                  # replicated, nobody disagrees
        self.assertFalse(tier2_gate(True, False, disag)["pass"])
        g = tier2_gate(False, True, agree)
        self.assertTrue(g["pass"])
        self.assertEqual(g["route"], "capacity exception")
        self.assertFalse(tier2_gate(False, True, incon)["pass"])                 # exception needs E2 'agree'
        self.assertFalse(tier2_gate(False, False, agree)["pass"])                # estimate failure


class E4ShiftsTest(unittest.TestCase):
    """The E4 glue with stand-ins for association and set building: a shift of matched boxes by -delta x sdot along
    d-hat must move the estimate by +delta (recovery 1.0) and change no assignment."""

    def test_recovery_is_one_with_identity_association(self):
        from careport import xfc7_ext_tier1 as t
        mo = _synthetic_with_margin("lidar")
        mo["camera_object_id"] = mo["segment"] + "/" + mo["frame"].astype(str) + "/" + mo["camera"].astype(str) + "/" + mo["object_id"]
        assoc = mo[["segment", "frame", "camera", "object_id", "camera_object_id"]].copy()
        ref = mo[["segment", "frame", "camera", "camera_object_id"]].assign(u=mo["u_ref"], v=mo["v_ref"], w=50.0)
        mo_d = mo.drop(columns=["camera_object_id"])

        def fake_assoc(lab, boxes, gate):
            a = assoc.merge(boxes, on=["segment", "frame", "camera", "camera_object_id"])
            return a.rename(columns={"u": "u_ref", "v": "v_ref", "w": "w_ref"}).assign(cost=0.0, margin=np.inf)

        def fake_sets(lab, a2):
            m = mo_d.drop(columns=["u_ref", "v_ref"]).merge(a2[["segment", "frame", "camera", "object_id", "u_ref", "v_ref"]],
                                                            on=["segment", "frame", "camera", "object_id"])
            return None, m, None

        saved = t.associate_images, t.build_sets
        t.associate_images, t.build_sets = fake_assoc, fake_sets
        try:
            runs = t.e4_shifts(None, ref, mo_d, assoc, deltas=(-10.0, 20.0))
        finally:
            t.associate_images, t.build_sets = saved
        for run in runs.values():
            self.assertEqual(run["assignment_changed_share"], 0.0)
            for cam in run["cameras"].values():
                self.assertAlmostEqual(cam["recovery_ratio"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
