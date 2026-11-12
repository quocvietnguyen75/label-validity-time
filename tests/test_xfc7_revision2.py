import unittest

import numpy as np

from careport.xfc7_estimators import weighted_mean_cr1
from careport.xfc7_revision2 import (cells_mean_cov_cr1, clearance, direction_z, percentile_rank, retention_lower_bound,
                                     shift_each, split_camera, stratified_camera, wald_chi2)


class StratifiedKeyTest(unittest.TestCase):
    def test_round_trip(self):
        code = stratified_camera(np.array([1, 3, 5]), np.array([2, 4, 2]))
        np.testing.assert_array_equal(code, [12, 34, 52])
        cam, ro = split_camera(code)
        np.testing.assert_array_equal(cam, [1, 3, 5])
        np.testing.assert_array_equal(ro, [2, 4, 2])


class JointCovarianceTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        n = 120
        self.y = rng.normal(size=n)
        self.w = rng.uniform(0.5, 2.0, size=n)
        self.cl = np.repeat(np.arange(12), 10)
        self.cell = np.tile(["A", "B"], n // 2)

    def test_diagonal_matches_weighted_mean_cr1(self):
        cells, mu, V = cells_mean_cov_cr1(self.y, self.w, self.cell, self.cl)
        self.assertEqual(list(cells), ["A", "B"])
        for k, c in enumerate(cells):
            m = self.cell == c
            est, se = weighted_mean_cr1(self.y[m], self.w[m], self.cl[m])
            self.assertAlmostEqual(mu[k], est)
            self.assertAlmostEqual(V[k, k], se ** 2)    # both cells span the same 12 clusters

    def test_disjoint_clusters_have_zero_covariance(self):
        cell = np.where(self.cl < 6, "A", "B")
        _, _, V = cells_mean_cov_cr1(self.y, self.w, cell, self.cl)
        self.assertAlmostEqual(V[0, 1], 0.0)
        self.assertAlmostEqual(V[1, 0], 0.0)


class TestsTest(unittest.TestCase):
    def test_direction_z(self):
        self.assertAlmostEqual(direction_z(5.0, 3.0, 1.0, 4.0), 4.0 / 5.0)

    def test_wald_chi2_identity_covariance(self):
        stat, df, p = wald_chi2(np.array([1.0, 2.0]), np.eye(2))
        self.assertAlmostEqual(stat, 5.0)
        self.assertEqual(df, 2)
        self.assertAlmostEqual(p, float(np.exp(-2.5)))     # chi-square(2) survival = exp(-x/2)


class BoundsTest(unittest.TestCase):
    def test_retention_bound_subtracts_bias(self):
        self.assertAlmostEqual(retention_lower_bound(47.7, 0.0, 49.1), 47.7 / 49.1)
        self.assertAlmostEqual(retention_lower_bound(47.7, 21.3, 49.1), 26.4 / 49.1)
        self.assertAlmostEqual(retention_lower_bound(43.6, -0.8, 60.0), 44.4 / 60.0)

    def test_clearance_sign(self):
        self.assertAlmostEqual(clearance(43.6, 64.6, 41.9), 43.6 - 41.9)    # above the band
        self.assertAlmostEqual(clearance(-30.0, -20.0, 17.6), 2.4)          # below the band
        self.assertAlmostEqual(clearance(32.3, 67.7, 41.9), -9.6)           # straddles the upper edge: shrink tol by 9.6
        self.assertAlmostEqual(clearance(-5.8, 1.4, 23.8), -25.2)           # inside: the smaller shift that clears it


class ShiftTest(unittest.TestCase):
    def test_per_observation_shift(self):
        p = np.zeros((2, 2))
        d = np.array([[1.0, 0.0], [0.0, 1.0]])
        out = shift_each(p, d, np.array([2.0, 0.5]), np.array([-10.0, 4.0]))
        np.testing.assert_allclose(out, [[20.0, 0.0], [0.0, -2.0]])          # p - delta * sdot * d_hat, row by row


class PercentileTest(unittest.TestCase):
    def test_share_at_or_below(self):
        self.assertAlmostEqual(percentile_rank([1, 2, 3, 4], 3), 75.0)
        self.assertAlmostEqual(percentile_rank([1, 2, 3, 4], 0.5), 0.0)


if __name__ == "__main__":
    unittest.main()
