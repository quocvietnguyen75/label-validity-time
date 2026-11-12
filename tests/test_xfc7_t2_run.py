import unittest

import numpy as np
import pandas as pd


def _moving(theta_true, n_seg=20, seed=0, pi=None):
    """Synthetic moving set whose theta-hat equals theta_true (+ noise); cameras 2 and 4 (FL, SL), two frame types."""
    rng = np.random.default_rng(seed)
    rows = []
    for sg in range(n_seg):
        for trk in range(4):
            cam = [2, 4, 6][trk % 3] if pi is not None else (2 if trk % 2 == 0 else 4)
            th_a, th_b = {2: (28.0, 77.0), 4: (12.0, 62.0), 6: (91.0, 48.0)}[cam]
            use_a = pi is not None and rng.random() < pi
            for fr in range(6):
                sdot = rng.uniform(0.4, 1.5)
                ang = rng.uniform(0, 2 * np.pi)
                d = np.array([np.cos(ang), np.sin(ang)])
                z_b = rng.normal(1.0, 0.5)
                p0 = rng.uniform(300, 1700, 2)
                if theta_true == "mix":
                    t = th_a if use_a else th_b
                else:
                    t = {"start": 0.0, "A": th_a, "B": th_b}[theta_true]
                t = t + rng.normal(0, 3.0)
                u = p0 - (sdot * t + z_b) * d
                rows.append({"segment": f"log{sg}", "object_id": f"o{trk}", "camera": cam, "frame": fr,
                             "frame_type": "A" if fr % 2 == 0 else "B", "p0_u": p0[0], "p0_v": p0[1],
                             "u_ref": u[0], "v_ref": u[1], "d_u": d[0], "d_v": d[1], "s": 1.0, "z_b": z_b,
                             "sdot": sdot, "sigma_b": 4.0, "theta_start": 0.0, "theta_A": th_a, "theta_B": th_b,
                             "theta_mid": 50.0, "speed": rng.uniform(4, 20), "margin": np.inf})
    df = pd.DataFrame(rows)
    df["n_track"] = df.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
    df["w"] = df["sdot"] ** 2 / (df["sigma_b"] ** 2 * df["n_track"])
    return df


NAMES = {2: "ring_front_left", 4: "ring_side_left"}
BUD = {"ring_front_left": 10.0, "ring_side_left": 10.0}


class MeanCiTest(unittest.TestCase):
    def test_interval_uses_t_with_g_minus_1(self):
        from careport.xfc7_estimators import weighted_mean_cr1
        from careport.xfc7_t2_run import mean_ci_t
        y = np.arange(20.0)
        cl = np.repeat(np.arange(5), 4)
        est, se = weighted_mean_cr1(y, np.ones(20), cl)
        r = mean_ci_t(y, np.ones(20), cl)
        self.assertAlmostEqual(r["est"], est)
        self.assertAlmostEqual(r["hi"] - r["est"], 2.7764 * se, places=3)       # t(0.975, 4)


class VerdictTest(unittest.TestCase):
    def test_b_truth_rules_out_start_and_a_not_b(self):
        from careport.xfc7_t2_run import add_theta_hat, verdicts_t2
        mo = add_theta_hat(_moving("B"))
        v = verdicts_t2(mo, NAMES, BUD)
        self.assertEqual(v["final"]["start"], "ruled out")
        self.assertEqual(v["final"]["A"], "ruled out")
        self.assertEqual(v["final"]["B"], "not ruled out")
        self.assertAlmostEqual(v["estimates"]["ring_side_left"]["B"]["est"], 0.0, delta=1.5)


class TwoFrameTest(unittest.TestCase):
    def test_no_difference_between_frame_types_when_the_model_is_right(self):
        from careport.xfc7_t2_run import add_theta_hat, two_frame_check
        tf = two_frame_check(add_theta_hat(_moving("B")), NAMES)
        for c in NAMES.values():
            self.assertLess(abs(tf[c]["diff_ms"]), 2.0)
            self.assertFalse(tf[c]["timing_model_failure"])


class MixtureTest(unittest.TestCase):
    def test_pi_hat_recovers_the_share_fitted_to_lidar_a(self):
        from careport.xfc7_t2_run import add_theta_hat, mixture_pi
        r = mixture_pi(add_theta_hat(_moving("mix", n_seg=60, pi=0.3, seed=3)))
        self.assertAlmostEqual(r["pi_hat"], 0.3, delta=0.12)


class CameraZeroLineTest(unittest.TestCase):
    def test_median_and_scaled_mad_per_camera_with_minimum(self):
        from careport.xfc7_t2_run import camera_zero_line
        st = pd.DataFrame({"camera": [3] * 31 + [1] * 5, "e": list(np.arange(31.0)) + [0.0] * 5})
        z = camera_zero_line(st, min_n=30)
        self.assertIn(3, z)
        self.assertNotIn(1, z)
        self.assertAlmostEqual(z[3]["z_b"], 15.0)
        self.assertAlmostEqual(z[3]["sigma_b"], 1.4826 * 8.0, places=6)


class TruckScenesBlockTest(unittest.TestCase):
    def test_each_hypothesis_uses_its_own_rows_and_only_pinned_cameras_are_reported(self):
        from careport.xfc7_t2_run import ts_block
        mo = _moving("B", n_seg=7, seed=5).rename(columns={"theta_B": "theta_L1"})
        mo["camera"] = 3
        mo["theta_sample"] = 0.0
        mo["theta_L2"] = np.nan                                     # this LiDAR never hits these boxes
        mo["theta_L3"] = np.where(mo["frame"] < 3, mo["theta_L1"], np.nan)
        mo["theta_L4"] = np.where(np.arange(len(mo)) < 5, mo["theta_L1"], np.nan)  # too few rows for an interval
        extra = mo.iloc[:2].copy()
        extra["camera"] = 4
        mo = pd.concat([mo, extra], ignore_index=True).drop(columns=["z_b", "sigma_b", "w", "n_track"])
        z_true = 1.0
        st = pd.DataFrame({"camera": [3] * 40 + [4] * 40, "e": np.r_[np.full(20, z_true - 4), np.full(20, z_true + 4)] .tolist() * 2})
        r = ts_block(st, mo, ["sample", "L1", "L2", "L3", "L4"], {3: "LB", 4: "RB"}, report=["LB"])
        self.assertEqual(sorted(r["theta_bar"]), ["LB"])
        self.assertEqual(r["counts"]["RB"]["obs"], 2)
        self.assertNotIn("L2", r["delta"]["LB"])                    # no rows for this hypothesis
        self.assertNotIn("L4", r["delta"]["LB"])
        self.assertEqual(r["delta"]["LB"]["_n"]["L4"], 5)
        self.assertEqual(r["delta"]["LB"]["_n"]["L2"], 0)
        self.assertEqual(r["delta"]["LB"]["_n"]["L3"], int((mo["camera"].eq(3) & mo["frame"].lt(3)).sum()))
        self.assertAlmostEqual(r["delta"]["LB"]["L1"]["est"], 0.0, delta=3.0)
        self.assertIn(r["C2_sign"]["LB"], ("positive", "negative", "undetermined"))
        self.assertEqual(r["C2_sign"]["LB"], "positive")                # theta near 62-77 ms against sample time 0


def _sha(p):
    import hashlib
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _write_json(p, obj):
    import json
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def _static_zero_lined(mo, offsets=None):
    """Zero-lined parked rows: 20 rows from 4 tracks per (segment, camera) of `mo`, r = offsets.get(cell, 0)."""
    rows = []
    for (sg, cam), _ in mo.groupby(["segment", "camera"]):
        for i in range(20):
            rows.append({"segment": sg, "camera": cam, "object_id": f"p{i % 4}", "r": (offsets or {}).get((sg, cam), 0.0)})
    return pd.DataFrame(rows)


def _e2e_fixture(d, protocol_name="proto.md", drop_input=None, s1_rule="far40"):
    """A pinned synthetic run: AV2 moving set and S3 summary, TruckScenes tables, config and protocol files."""
    import os
    av2, ts = os.path.join(d, "av2"), os.path.join(d, "ts")
    os.makedirs(av2)
    os.makedirs(ts)
    mo = _moving("B", n_seg=12, seed=7)
    mo.to_parquet(os.path.join(av2, "mo.parquet"), index=False)
    _static_zero_lined(mo).to_parquet(os.path.join(av2, "zl.parquet"), index=False)
    summ = os.path.join(d, "s3.json")
    cap = {c: {"start-A": {"decidable": False}, "start-B": {"decidable": True}} for c in ("ring_front_left", "ring_side_left")}
    _write_json(summ, {"S1_rule": s1_rule, "decision_cameras": ["ring_front_left", "ring_side_left"],
                       "budget": {"ring_front_left": {"B_c_ms": 10.0}, "ring_side_left": {"B_c_ms": 10.0}},
                       "capacity": {"cameras": cap}})
    tmo = _moving("B", n_seg=7, seed=8).rename(columns={"theta_B": "theta_L1"}).drop(columns=["z_b", "sigma_b", "w"])
    tmo["camera"] = 3
    tmo["theta_sample"] = 0.0
    tmo.to_parquet(os.path.join(ts, "ma.parquet"), index=False)
    pd.DataFrame({"camera": [3] * 40, "e": np.r_[np.full(20, -3.0), np.full(20, 5.0)]}).to_parquet(
        os.path.join(ts, "st.parquet"), index=False)
    inputs = [{"what": w, "path": p, "sha256": _sha(r)} for w, p, r in (
        ("mo", "$AV2_SENSOR_ROOT/mo.parquet", os.path.join(av2, "mo.parquet")),
        ("zl", "$AV2_SENSOR_ROOT/zl.parquet", os.path.join(av2, "zl.parquet")),
        ("s3", summ, summ),
        ("st", "$TRUCKSCENES_DERIVED/st.parquet", os.path.join(ts, "st.parquet")),
        ("ma", "$TRUCKSCENES_DERIVED/ma.parquet", os.path.join(ts, "ma.parquet"))) if w != drop_input]
    blk = {"s3_summary": summ, "moving_set": "$AV2_SENSOR_ROOT/mo.parquet", "static_zero_lined": "$AV2_SENSOR_ROOT/zl.parquet"}
    exp = {"S1_rule": "far40", "decision_cameras": {t: ["ring_front_left", "ring_side_left"] for t in ("primary", "sensitivity_score_0_3")}}
    cfg = {"protocol": protocol_name, "inputs": inputs, "expected": exp, "primary": blk, "sensitivity_score_0_3": blk,
           "p01_logs": [f"log{i}" for i in range(6)],
           "truckscenes": {"static_all": "$TRUCKSCENES_DERIVED/st.parquet", "moving_assigned": "$TRUCKSCENES_DERIVED/ma.parquet",
                           "hypotheses": ["sample", "L1"], "camera_names": {"3": "LB"}, "report_cameras": ["LB"]}}
    cp, pp, out = os.path.join(d, "cfg.json"), os.path.join(d, "proto.md"), os.path.join(d, "out.json")
    _write_json(cp, cfg)
    with open(pp, "w", encoding="utf-8") as f:
        f.write("protocol")
    return {"argv": ["x", cp, _sha(cp), pp, _sha(pp), out], "env": {"AV2_SENSOR_ROOT": av2, "TRUCKSCENES_DERIVED": ts},
            "out": out}


class MainEndToEndTest(unittest.TestCase):
    def _run(self, **kw):
        import os
        import tempfile
        from unittest import mock
        from careport import xfc7_t2_run as m
        d = tempfile.mkdtemp()
        fx = _e2e_fixture(d, **kw)
        with mock.patch.dict(os.environ, fx["env"]):
            return m, fx, m.main(fx["argv"])

    def test_main_runs_on_a_pinned_synthetic_config(self):
        import json
        import os
        from unittest import mock
        m, fx, rc = self._run()
        self.assertEqual(rc, 0)
        with open(fx["out"], encoding="utf-8") as f:
            r = json.load(f)
        self.assertEqual(r["primary"]["verdicts"]["final"]["start"], "ruled out")
        self.assertIn("p01_subgroup", r)
        self.assertIn("L1", r["truckscenes_descriptive"]["delta"]["LB"])
        self.assertEqual(r["truckscenes_descriptive"]["C2_sign"]["LB"], "positive")
        self.assertNotIn("final", r["p01_subgroup"]["verdicts"])                      # descriptive, no verdict
        self.assertIn("final_descriptive", r["p01_subgroup"]["verdicts"])
        self.assertIn("segment_intercept", r["primary"]["secondary"])
        with mock.patch.dict(os.environ, fx["env"]):
            with self.assertRaises(SystemExit):                                       # run once: never overwrite
                m.main(fx["argv"])
            os.remove(fx["out"])
            bad = list(fx["argv"])
            bad[2] = "0" * 64
            with self.assertRaises(SystemExit):                                       # a wrong pin stops the run
                m.main(bad)
        self.assertFalse(os.path.exists(fx["out"]))

    def test_protocol_file_must_be_the_one_the_config_names(self):
        with self.assertRaises(SystemExit):
            self._run(protocol_name="PILOT_PROTOCOL_OTHER.md")

    def test_every_file_read_must_be_pinned(self):
        with self.assertRaises(SystemExit):
            self._run(drop_input="ma")

    def test_summary_must_carry_the_expected_s1_rule(self):
        with self.assertRaises(SystemExit):
            self._run(s1_rule="written")


class EdgeCaseTest(unittest.TestCase):
    """Cases from the T2 implementation check (t2_check_synthetic.py), promoted to tests."""

    def test_timing_model_failure_is_flagged_and_dropped_from_the_two_camera_rule(self):
        from careport.xfc7_t2_run import add_theta_hat, two_frame_check, verdicts_t2
        mo = _moving("B", n_seg=20, seed=1)
        sel = (mo["camera"] == 2) & (mo["frame_type"] == "A")
        mo.loc[sel, ["u_ref", "v_ref"]] = (mo.loc[sel, ["u_ref", "v_ref"]].to_numpy()
                                            - (mo.loc[sel, "sdot"].to_numpy() * 30.0)[:, None] * mo.loc[sel, ["d_u", "d_v"]].to_numpy())
        mo = add_theta_hat(mo)
        tf = two_frame_check(mo, NAMES)
        fail = [c for c, v in tf.items() if v["timing_model_failure"]]
        self.assertEqual(fail, ["ring_front_left"])
        v = verdicts_t2(mo, NAMES, BUD, exclude=fail)
        self.assertEqual(v["cameras"], ["ring_side_left"])
        self.assertEqual(v["final"]["start"], "not ruled out")                       # one camera cannot rule out
        self.assertIn("ring_front_left", v["estimates"])                              # still reported

    def test_margin_exclusion_drops_rows_and_reweights(self):
        from careport.xfc7_t2_run import _reweight
        mo = _moving("B", n_seg=20, seed=2)
        mo.loc[mo.index[:7], "margin"] = 0.1
        sub = _reweight(mo[(mo["margin"] >= 0.2) | np.isinf(mo["margin"])])
        self.assertEqual(len(sub), len(mo) - 7)
        self.assertTrue((sub["n_track"] != mo.loc[sub.index, "n_track"]).any())

    def test_absent_camera_and_single_log_camera_in_a_subset(self):
        from careport.xfc7_t2_run import av2_block
        names = {2: "ring_front_left", 4: "ring_side_left", 6: "ring_rear_left", 7: "ring_rear_right"}
        mo = _moving("B", n_seg=12, seed=3)
        extra = mo[(mo["camera"] == 2) & (mo["segment"] == "log0")].copy()
        extra["camera"] = 6                                                          # one log only: G = 1
        mo = pd.concat([mo, extra], ignore_index=True)
        out = av2_block(mo, names, {n: 10.0 for n in names.values()}, list(names.values()))
        est = out["verdicts"]["estimates"]
        self.assertNotIn("ring_rear_right", est)
        self.assertTrue(np.isnan(est["ring_rear_left"]["start"]["lo"]))
        self.assertIn(["ring_rear_left", "start"], out["verdicts"]["undefined_intervals"])

    def test_camera_with_zero_sigma_is_not_given_a_zero_line(self):
        from careport.xfc7_t2_run import camera_zero_line
        st = pd.DataFrame({"camera": [3] * 40 + [4] * 40, "e": [1.0] * 40 + list(np.arange(40.0))})
        self.assertEqual(sorted(camera_zero_line(st)), [4])


def _shift_a_frames(mo, cams, ms=30.0):
    mo = mo.copy()
    sel = mo["camera"].isin(cams) & (mo["frame_type"] == "A")
    mo.loc[sel, ["u_ref", "v_ref"]] = (mo.loc[sel, ["u_ref", "v_ref"]].to_numpy()
                                        - (mo.loc[sel, "sdot"].to_numpy() * ms)[:, None] * mo.loc[sel, ["d_u", "d_v"]].to_numpy())
    return mo


CAP = {c: {"start-A": {"decidable": False}, "start-B": {"decidable": True}} for c in NAMES.values()}


class TimingConsequenceTest(unittest.TestCase):
    def test_failing_camera_leaves_pooled_estimates_and_contrasts_are_recounted(self):
        from careport.xfc7_t2_run import add_theta_hat, av2_block, mean_ci_t
        mo = _shift_a_frames(_moving("B", n_seg=20, seed=1), [2])
        out = av2_block(mo, NAMES, BUD, list(BUD), capacity_cams=CAP)
        self.assertEqual(out["timing_model_failures"], ["ring_front_left"])
        self.assertFalse(out["timing_model_rejected"])
        self.assertFalse(out["decidable_after_timing"]["start-B"])                   # one camera left
        ok = add_theta_hat(mo[mo["camera"] == 4])
        self.assertAlmostEqual(out["C2_pooled_theta_bar"]["est"], mean_ci_t(ok["theta_hat"], ok["w"], ok["segment"])["est"])

    def test_two_failures_reject_the_timing_model(self):
        from careport.xfc7_t2_run import av2_block
        out = av2_block(_shift_a_frames(_moving("B", n_seg=20, seed=1), [2, 4]), NAMES, BUD, list(BUD), capacity_cams=CAP)
        self.assertTrue(out["timing_model_rejected"])

    def test_no_failure_keeps_capacity(self):
        from careport.xfc7_t2_run import av2_block
        out = av2_block(_moving("B", n_seg=20, seed=1), NAMES, BUD, list(BUD), capacity_cams=CAP)
        self.assertEqual(out["timing_model_failures"], [])
        self.assertTrue(out["decidable_after_timing"]["start-B"])
        self.assertFalse(out["decidable_after_timing"]["start-A"])


class C3Test(unittest.TestCase):
    def test_slopes_on_speed_and_inverse_image_speed_pooled_and_per_camera(self):
        from careport.xfc7_t2_run import av2_block
        out = av2_block(_moving("B", n_seg=20, seed=2), NAMES, BUD, list(BUD))
        self.assertIn("slope", out["C3_inv_sdot"])
        self.assertEqual(sorted(out["C3_speed_by_camera"]), sorted(NAMES.values()))


class SecondaryTest(unittest.TestCase):
    def test_per_log_intercept_correction_removes_a_log_offset_and_jackknife_is_reported(self):
        from careport.xfc7_t2_run import av2_block
        mo = _moving("B", n_seg=12, seed=9)
        sel = (mo["segment"] == "log0") & (mo["camera"] == 4)
        mo.loc[sel, ["u_ref", "v_ref"]] = mo.loc[sel, ["u_ref", "v_ref"]].to_numpy() - 20.0 * mo.loc[sel, ["d_u", "d_v"]].to_numpy()
        st = _static_zero_lined(mo, {("log0", 4): 20.0})
        out = av2_block(mo, NAMES, BUD, list(BUD), st=st)
        prim = out["verdicts"]["estimates"]["ring_side_left"]["B"]["est"]
        corr = out["secondary"]["segment_intercept"]["ring_side_left"]["B"]["est"]
        self.assertGreater(abs(prim), 1.5)
        self.assertLess(abs(corr), 1.0)
        jk = out["secondary"]["jackknife"]["ring_side_left"]["B"]
        self.assertGreater(jk["se"], 0.0)
        self.assertLess(jk["lo"], jk["hi"])


if __name__ == "__main__":
    unittest.main()
