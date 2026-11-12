import unittest

import numpy as np

from careport.xfc7_sweep import column_positions, column_times, fit_sweep


class ColumnPositionsTest(unittest.TestCase):
    def test_median_over_valid_rows_and_nan_for_empty_columns(self):
        img = np.zeros((4, 3, 6))
        img[:, 0, 3:6] = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [9.0, 9.0, 9.0], [1.0, 2.0, 3.0]]
        img[1, 1, 3:6] = [4.0, 5.0, 6.0]
        p = column_positions(img)
        np.testing.assert_allclose(p[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(p[1], [4.0, 5.0, 6.0])
        self.assertTrue(np.isnan(p[2]).all())


class ColumnTimesTest(unittest.TestCase):
    def test_recovers_column_times_on_linear_trajectory(self):
        v = np.array([8.0, 6.0, 0.0])
        x0 = np.array([7000.0, -1500.0, 200.0])
        img_t = np.array([-0.024, 0.026, 0.038, 0.051, 0.064, 0.076, 0.126])
        img_xyz = x0 + np.outer(img_t, v)
        true_t = np.array([0.0002, 0.025, 0.05, 0.0999])
        cols = x0 + np.outer(true_t, v)
        cols = np.vstack([cols, [np.nan, np.nan, np.nan]])
        t = column_times(cols, img_t, img_xyz, min_speed=2.0)
        np.testing.assert_allclose(t[:4], true_t, atol=1e-9)
        self.assertTrue(np.isnan(t[4]))

    def test_slow_ego_gives_nan(self):
        img_t = np.array([0.0, 0.05, 0.1])
        img_xyz = np.tile([1.0, 2.0, 3.0], (3, 1)) + np.outer(img_t, [0.5, 0.0, 0.0])
        t = column_times(np.array([[1.0, 2.0, 3.0]]), img_t, img_xyz, min_speed=2.0)
        self.assertTrue(np.isnan(t).all())


class FitSweepTest(unittest.TestCase):
    def test_intercept_slope_and_residual_spread(self):
        w = 2650
        j = np.arange(w, dtype=float)
        t = 100.0 * (j + 0.5) / w
        t[::7] = np.nan
        a, b, mad, n = fit_sweep(j, t)
        self.assertAlmostEqual(a, 100.0 * 0.5 / w, places=9)
        self.assertAlmostEqual(b * w, 100.0, places=9)
        self.assertAlmostEqual(mad, 0.0, places=9)
        self.assertEqual(n, int(np.isfinite(t).sum()))


if __name__ == "__main__":
    unittest.main()
