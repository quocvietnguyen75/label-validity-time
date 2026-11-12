import unittest

import numpy as np
import pandas as pd

from careport.xfc7_ext_e0 import common_moving, paired_static, predicted_pair_halfwidth

K4 = ["segment", "frame", "camera", "object_id"]


def _static(r, sigma, keys):
    return pd.DataFrame({"segment": [k[0] for k in keys], "frame": [k[1] for k in keys], "camera": 1,
                         "object_id": [k[2] for k in keys], "r": r, "sigma_b": sigma})


class PairedStaticTest(unittest.TestCase):
    def test_difference_variance_and_weights_on_matched_rows_only(self):
        h = _static([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [("a", 1, "x"), ("a", 2, "x"), ("b", 1, "y")])
        d = _static([0.5, 2.5], [2.0, 2.0], [("a", 1, "x"), ("a", 2, "x")])
        p = paired_static(h, d)
        self.assertEqual(len(p), 2)                                  # ("b", 1, "y") has no detector match
        np.testing.assert_allclose(p.sort_values("frame")["r_pair"], [0.5, -0.5])
        np.testing.assert_allclose(p["var_pair"], [5.0, 5.0])
        np.testing.assert_allclose(p["w_pair"], [1 / (5.0 * 2), 1 / (5.0 * 2)])   # track of two common frames


class CommonMovingTest(unittest.TestCase):
    def test_weights_and_no_reference_positions(self):
        base = {"segment": ["a", "a", "b"], "frame": [1, 2, 1], "camera": [1, 1, 1], "object_id": ["x", "x", "y"],
                "sdot": [10.0, 10.0, 20.0], "u_ref": [5.0, 6.0, 7.0], "v_ref": [1.0, 1.0, 1.0], "z_b": [0.0, 0.0, 0.0]}
        h = pd.DataFrame(dict(base, sigma_b=[1.0, 1.0, 1.0]))
        d = pd.DataFrame(dict(base, sigma_b=[2.0, 2.0, 2.0])).iloc[:2]
        m = common_moving(h, d)
        self.assertEqual(len(m), 2)
        np.testing.assert_allclose(m["w_pair"], [100 / (5.0 * 2), 100 / (5.0 * 2)])
        np.testing.assert_allclose(m["var_pair_ms2"], [5.0 / 100, 5.0 / 100])
        for col in ("u_ref", "v_ref", "u_ref_h", "u_ref_d"):
            self.assertNotIn(col, m.columns)                         # E0 must not carry moving reference positions


class HalfWidthTest(unittest.TestCase):
    def test_independent_static_residuals_give_design_effect_near_one(self):
        rng = np.random.default_rng(0)
        n = 4000
        st = pd.DataFrame({"r_pair": rng.normal(0, 1, n), "w_pair": np.ones(n), "var_pair": np.ones(n),
                           "segment": np.arange(n).astype(str)})
        mv = pd.DataFrame({"w_pair": np.ones(100), "var_pair_ms2": np.full(100, 4.0)})
        hw, de = predicted_pair_halfwidth(st, mv)
        self.assertAlmostEqual(de, 1.0, delta=0.1)
        self.assertAlmostEqual(hw, 1.96 * np.sqrt(de * 4.0 / 100), places=6)


if __name__ == "__main__":
    unittest.main()
