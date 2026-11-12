"""Synthetic tests for the XF-C7 post hoc set V5R (pilot record entry '2026-11-05, post hoc set V5R'), written first."""
import math
import unittest

import numpy as np
import pandas as pd

from careport import xfc7_t3_posthoc_v5r as v
from careport.xfc7_t2_run import slope_ci_t

NAMES = {1: "cam_a", 2: "cam_b"}


def _frame(seed=0, n_logs=40, per_log=6, beta=12.0, alphas=(30.0, 70.0), inv_means=(0.5, 2.0)):
    """Two cameras with different true means and different typical 1/sdot; theta_hat = alpha_c + beta / sdot + noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for cam, (alpha, im) in enumerate(zip(alphas, inv_means), start=1):
        for g in range(n_logs):
            log_shift = rng.normal(0, 2.0)
            for _ in range(per_log):
                inv = abs(rng.normal(im, 0.3)) + 0.05
                rows.append({"camera": cam, "segment": f"log{g:03d}", "sdot": 1.0 / inv, "w": rng.uniform(0.5, 2.0),
                             "theta_hat": alpha + beta * inv + log_shift + rng.normal(0, 1.0)})
    return pd.DataFrame(rows)


class WithinSlope(unittest.TestCase):
    def test_single_camera_equals_slope_ci_t(self):
        f = _frame()
        g = f[f["camera"] == 1]
        got = v.within_slope(g)
        ref = slope_ci_t(g["theta_hat"], 1.0 / g["sdot"], g["w"], g["segment"])
        for k_got, k_ref in (("slope", "slope"), ("lo", "slope_lo"), ("hi", "slope_hi")):
            self.assertAlmostEqual(got[k_got], ref[k_ref], places=9)
        self.assertEqual(got["G"], ref["G"])

    def test_recovers_common_beta_where_pooled_is_confounded(self):
        f = _frame(beta=12.0)
        within = v.within_slope(f)
        pooled = slope_ci_t(f["theta_hat"], 1.0 / f["sdot"], f["w"], f["segment"])
        self.assertLess(abs(within["slope"] - 12.0), 1.0)
        self.assertGreater(abs(pooled["slope"] - 12.0), 5.0)     # camera mix inflates the pooled slope here


class Adjust(unittest.TestCase):
    def test_subtracts_beta_over_sdot(self):
        f = pd.DataFrame({"theta_hat": [10.0, 20.0], "sdot": [2.0, 4.0]})
        out = v.adjust(f, 8.0)
        self.assertEqual(list(out["theta_hat"]), [6.0, 18.0])
        self.assertEqual(list(f["theta_hat"]), [10.0, 20.0])     # input untouched


class Midpass(unittest.TestCase):
    def test_mean_of_passes_and_dropped_count(self):
        f = pd.DataFrame({"theta_A": [40.0, 10.0, np.nan], "theta_B": [90.0, 60.0, 50.0]})
        out, dropped = v.add_midpass(f)
        self.assertEqual(dropped, 1)
        self.assertEqual(list(out["theta_midpass"]), [65.0, 35.0])


class SharesAndBias(unittest.TestCase):
    def test_shares_sum_to_one(self):
        f = _frame()
        s = v.camera_shares(f, NAMES)
        self.assertAlmostEqual(sum(s.values()), 1.0, places=12)

    def test_implied_bias(self):
        f = pd.DataFrame({"camera": [1, 1], "w": [1.0, 3.0], "sdot": [2.0, 4.0]})
        b = v.implied_bias(f, NAMES, 10.0)
        self.assertAlmostEqual(b["cam_a"], 10.0 * (1.0 / 2 + 3.0 / 4) / 4.0, places=12)


class ShiftThreshold(unittest.TestCase):
    def test_second_largest_clearance_smaller_run(self):
        budgets = {"c1": 10.0, "c2": 10.0, "c3": 10.0}
        # clearance = lo - (5 + 10): c1 20, c2 8, c3 -3 in run p; c1 18, c2 9 in run m (min over methods)
        iv = {"p": {"c1": {"start": {"cr1": (50, 35, 65), "x": (50, 37, 63)}},
                    "c2": {"start": {"cr1": (40, 23, 57), "x": (40, 24, 56)}},
                    "c3": {"start": {"cr1": (20, 12, 28), "x": (20, 13, 27)}}},
              "m": {"c1": {"start": {"cr1": (50, 33, 67), "x": (50, 34, 66)}},
                    "c2": {"start": {"cr1": (40, 24, 56), "x": (40, 25, 55)}}}}
        self.assertAlmostEqual(v.shift_threshold(iv, budgets, methods=("cr1", "x")), 8.0)
        self.assertAlmostEqual(v.shift_threshold(iv, budgets, methods=("cr1",)), 8.0)
        self.assertAlmostEqual(v.shift_threshold(iv, budgets, methods=("x",)), 9.0)


class Crossing(unittest.TestCase):
    @staticmethod
    def _profile(seam_deg, period=100.0):
        """Median offsets per 10-degree bin for a LiDAR that spins clockwise (offset falls with azimuth), starting at
        seam_deg: offset(az) = ((seam_deg - az) mod 360) * period / 360."""
        prof = {}
        for lo in range(-180, 180, 10):
            c = lo + 5
            prof[f"{lo}..{lo + 10}"] = round(((seam_deg - c) % 360) * period / 360.0, 4)
        return prof

    def test_interpolates_between_bin_centres(self):
        prof = self._profile(140.0)
        got = v.crossing_offset(prof, 0.0)                       # (140 - 0) * 100/360
        self.assertAlmostEqual(got, 140.0 * 100 / 360, places=3)

    def test_wraps_in_azimuth_and_time(self):
        prof = self._profile(140.0)
        self.assertAlmostEqual(v.crossing_offset(prof, 179.0), ((140 - 179) % 360) * 100 / 360, places=3)
        self.assertAlmostEqual(v.crossing_offset(prof, -179.0), ((140 + 179) % 360) * 100 / 360, places=3)

    def test_residuals_nearest_mode_mod_100(self):
        g3 = {"cameras": {"ring_front_center": {"azimuth_deg_median": 0.0}},
              "lidar_profile": {"lower_laser_numbers": self._profile(140.0), "upper_laser_numbers": self._profile(-40.0)},
              "frame_offsets_ms": {"ring_front_center": {"modes_ms": [140 * 100 / 360 + 0.3, (-40 % 360) * 100 / 360 - 0.2]}}}
        res = v.crossing_residuals(g3)
        self.assertAlmostEqual(res["pairs"]["ring_front_center"]["A"]["residual_ms"], 0.3, places=3)
        self.assertAlmostEqual(res["pairs"]["ring_front_center"]["B"]["residual_ms"], -0.2, places=3)
        self.assertAlmostEqual(res["r_max_ms"], 0.3, places=3)

    def test_wrapped_difference(self):
        self.assertAlmostEqual(v.wrap_ms(99.0 - 1.0), -2.0)
        self.assertAlmostEqual(v.wrap_ms(1.0 - 99.0), 2.0)


class Decidability(unittest.TestCase):
    def test_two_cameras_in_both_runs(self):
        thr = {"p": {"c1": 10.0, "c2": 10.0, "c3": 10.0}, "m": {"c1": 10.0, "c2": 10.0, "c3": 10.0}}
        th = {"p": {"c1": {"start": 0.0, "midpass": 20.0}, "c2": {"start": 0.0, "midpass": 15.0}, "c3": {"start": 0.0, "midpass": 5.0}},
              "m": {"c1": {"start": 0.0, "midpass": 20.0}, "c2": {"start": 0.0, "midpass": 9.0}, "c3": {"start": 0.0, "midpass": 5.0}}}
        d = v.midpass_decidable(th, thr, contrasts=[("start", "midpass")])
        self.assertTrue(d["runs"]["p"]["start-midpass"])
        self.assertFalse(d["runs"]["m"]["start-midpass"])
        self.assertFalse(d["decidable"]["start-midpass"])


if __name__ == "__main__":
    unittest.main()
