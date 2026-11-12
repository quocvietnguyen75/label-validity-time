import unittest

import numpy as np
import pandas as pd

from careport.xfc7_zero_line import lookup_zero_line, tau_hat, zero_line_tables

BINS = ["camera", "viewpoint_bin", "range_bin", "column_third", "ego_class"]


def _static():
    rng = np.random.default_rng(1)
    rows = []
    # bin A (cam 1, vp 6, range 1, col 1, ego 0): 40 obs around +3 px
    for e in 3.0 + rng.normal(0, 1, 40):
        rows.append((1, 6, 1, 1, 0, e))
    # bin B (cam 1, vp 6, range 1, col 1, ego 2): only 10 obs around -5 -> forces ego collapse
    for e in -5.0 + rng.normal(0, 1, 10):
        rows.append((1, 6, 1, 1, 2, e))
    # bin C (cam 1, vp 6, range 1, col 0, any ego): 25 obs -> with col collapse joins A and B
    for e in 1.0 + rng.normal(0, 1, 25):
        rows.append((1, 6, 1, 0, 1, e))
    return pd.DataFrame(rows, columns=BINS + ["e"])


class ZeroLineTest(unittest.TestCase):
    def test_levels_and_fallback(self):
        t = zero_line_tables(_static())
        moving = pd.DataFrame([(1, 6, 1, 1, 0), (1, 6, 1, 1, 2), (1, 6, 1, 0, 1), (2, 3, 0, 0, 0)], columns=BINS)
        z = lookup_zero_line(moving, t, min_n=30)
        self.assertEqual(list(z["zl_level"]), [0, 1, 2, -1])
        self.assertAlmostEqual(z["z_b"].iloc[0], float(_static().query("ego_class == 0 and column_third == 1")["e"].median()))
        self.assertTrue(z["no_zero_line"].iloc[3])
        self.assertTrue(np.isnan(z["z_b"].iloc[3]))
        self.assertTrue(z["zl_flagged"].iloc[1] and z["zl_flagged"].iloc[2] and not z["zl_flagged"].iloc[0])
        sel = _static().query("column_third == 1")["e"]
        self.assertAlmostEqual(z["z_b"].iloc[1], float(sel.median()))
        self.assertAlmostEqual(z["sigma_b"].iloc[1], float(1.4826 * np.median(np.abs(sel - sel.median()))))


class SegmentInterceptTest(unittest.TestCase):
    def test_intercepts_need_enough_static_support_and_correct_tau(self):
        from careport.xfc7_zero_line import apply_segment_correction, segment_intercepts
        st = pd.DataFrame({
            "segment": ["a"] * 30 + ["b"] * 30 + ["c"] * 10,
            "camera": [1] * 70,
            "object_id": [f"t{i % 5}" for i in range(30)] + ["u0"] * 30 + [f"v{i % 4}" for i in range(10)],
            "r": [4.0] * 30 + [-2.0] * 30 + [9.0] * 10,
        })
        ic = segment_intercepts(st, min_obs=20, min_tracks=3).set_index(["segment", "camera"])
        self.assertAlmostEqual(ic.loc[("a", 1), "c_seg"], 4.0)
        self.assertNotIn(("b", 1), ic.index)          # one track only
        self.assertNotIn(("c", 1), ic.index)          # too few observations
        mo = pd.DataFrame({"segment": ["a", "a", "b"], "camera": [1, 1, 1], "tau_hat": [30.0, 30.0, 30.0],
                           "s": [1, -1, 1], "sdot": [2.0, 2.0, 2.0]})
        out = apply_segment_correction(mo, ic.reset_index())
        np.testing.assert_allclose(out["tau_hat_seg"].to_numpy()[:2], [30.0 - 2.0, 30.0 + 2.0])
        self.assertTrue(np.isnan(out["tau_hat_seg"].iloc[2]))


class TauHatTest(unittest.TestCase):
    def test_reversing_car_with_front_biased_silhouette_is_unbiased(self):
        e = np.array([1.0, 0.0])                   # body-forward image direction
        p0 = np.array([500.0, 300.0])
        sdot, tau_true, bias = 0.8, 37.0, 3.0      # static offset +3 px toward the front: (p0 - u_ref).e = +3
        for s in (1, -1):
            d = s * e
            p_tau = p0 - sdot * tau_true * d       # box valid at tau: the camera sees the car tau ms further along d
            u_ref = p_tau - bias * e               # same silhouette bias along the body axis
            th = tau_hat(p0[None], u_ref[None], d[None], np.array([s]), np.array([bias]), np.array([sdot]))
            self.assertAlmostEqual(float(th[0]), tau_true, places=9)


if __name__ == "__main__":
    unittest.main()
