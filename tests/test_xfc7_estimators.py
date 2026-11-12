import unittest

import numpy as np

from careport.xfc7_estimators import (
    design_effect_mean,
    design_effect_slope,
    ind_var_weighted_mean,
    ind_var_wls_slope,
    weighted_mean_cr1,
    wls_cr1,
    wls_intercept_slope_cr1,
    wls_slope_cr1,
)


class WeightedMeanTest(unittest.TestCase):
    def test_hand_computed_cr1(self):
        est, se = weighted_mean_cr1(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 1.0, 2.0, 2.0]), np.array(["a", "a", "b", "b"]))
        self.assertAlmostEqual(est, 17.0 / 6.0)
        self.assertAlmostEqual(se ** 2, 2.0 * (2 * (8.0 / 3.0) ** 2) / 36.0)

    def test_cr1_matches_sampling_spread_under_clustering(self):
        rng = np.random.default_rng(7)
        ests, ses = [], []
        for _ in range(400):
            G, n = 60, 8
            cl = np.repeat(np.arange(G), n)
            y = 5.0 + np.repeat(rng.normal(0, 2.0, G), n) + rng.normal(0, 3.0, G * n)
            w = rng.uniform(0.5, 2.0, G * n)
            e, s = weighted_mean_cr1(y, w, cl)
            ests.append(e)
            ses.append(s)
        ratio = np.mean(ses) / np.std(ests)
        self.assertGreater(ratio, 0.85)
        self.assertLess(ratio, 1.15)

    def test_independent_variance_formula(self):
        self.assertAlmostEqual(ind_var_weighted_mean(np.ones(50), np.full(50, 4.0)), 4.0 / 50)


class SlopeTest(unittest.TestCase):
    def test_camera_intercepts_correlated_with_column_are_removed(self):
        rng = np.random.default_rng(3)
        xa = rng.uniform(-0.5, 0.0, 400)
        xb = rng.uniform(0.0, 0.5, 400)
        x = np.concatenate([xa, xb])
        grp = np.array([1] * 400 + [2] * 400)
        y = np.where(grp == 1, 10.0, -10.0) + 5.0 * x + rng.normal(0, 0.5, 800)
        cl = rng.integers(0, 80, 800)
        w = rng.uniform(0.5, 2.0, 800)
        b, se = wls_slope_cr1(y, x, w, grp, cl)
        self.assertAlmostEqual(b, 5.0, delta=0.3)
        naive = np.polyfit(x, y, 1)[0]                    # pooled without intercepts is badly biased here
        self.assertLess(naive, -20.0)
        self.assertGreater(se, 0.0)

    def test_slope_cr1_matches_sampling_spread(self):
        rng = np.random.default_rng(11)
        bs, ses = [], []
        for _ in range(300):
            G, n = 50, 10
            cl = np.repeat(np.arange(G), n)
            grp = rng.integers(1, 4, G * n)
            x = rng.uniform(-0.5, 0.5, G * n)
            y = grp * 2.0 + 3.0 * x + np.repeat(rng.normal(0, 1.0, G), n) * x * 4 + rng.normal(0, 2.0, G * n)
            w = rng.uniform(0.5, 2.0, G * n)
            b, s = wls_slope_cr1(y, x, w, grp, cl)
            bs.append(b)
            ses.append(s)
        ratio = np.mean(ses) / np.std(bs)
        self.assertGreater(ratio, 0.85)
        self.assertLess(ratio, 1.15)


class DesignEffectTest(unittest.TestCase):
    def test_near_one_without_clustering_and_large_with_it(self):
        rng = np.random.default_rng(5)
        n = 3000
        cl = rng.integers(0, 300, n)
        var = np.full(n, 1.0)
        w = np.ones(n)
        indep = rng.normal(0, 1, n)
        self.assertAlmostEqual(design_effect_mean(indep, w, var, cl), 1.0, delta=0.25)
        clustered = np.repeat(rng.normal(0, 1, 300), 10) + rng.normal(0, 0.3, 3000)
        cl2 = np.repeat(np.arange(300), 10)
        self.assertGreater(design_effect_mean(clustered, w, np.full(3000, 1.09), cl2), 5.0)

    def test_slope_design_effect_and_ind_var(self):
        rng = np.random.default_rng(9)
        n = 2000
        x = rng.uniform(-0.5, 0.5, n)
        grp = rng.integers(1, 3, n)
        cl = rng.integers(0, 200, n)
        r = rng.normal(0, 1, n)
        v = ind_var_wls_slope(x, np.ones(n), grp, np.ones(n))
        self.assertGreater(v, 0.0)
        self.assertAlmostEqual(design_effect_slope(r, x, np.ones(n), np.ones(n), grp, cl), 1.0, delta=0.3)


class MultiRegressorTest(unittest.TestCase):
    def test_single_column_matches_slope(self):
        rng = np.random.default_rng(3)
        n = 400
        grp = rng.integers(0, 3, n)
        cl = rng.integers(0, 25, n)
        x = rng.normal(size=n) + grp
        y = 2.0 * x + grp * 5.0 + rng.normal(size=n)
        w = rng.uniform(0.5, 2.0, n)
        b, se = wls_cr1(y, x[:, None], w, grp, cl)
        b1, se1 = wls_slope_cr1(y, x, w, grp, cl)
        self.assertAlmostEqual(b[0], b1, places=10)
        self.assertAlmostEqual(se[0], se1, places=10)

    def test_two_regressors_recovered_with_group_intercepts(self):
        rng = np.random.default_rng(4)
        n = 600
        grp = rng.integers(0, 4, n)
        x1 = rng.uniform(-0.5, 0.5, n)
        x2 = x1 * rng.uniform(0, 20, n)
        y = 3.0 * x1 - 0.5 * x2 + 10.0 * grp
        b, se = wls_cr1(y, np.stack([x1, x2], axis=1), np.ones(n), grp, rng.integers(0, 30, n))
        np.testing.assert_allclose(b, [3.0, -0.5], atol=1e-9)
        np.testing.assert_allclose(se, [0.0, 0.0], atol=1e-6)


class InterceptSlopeTest(unittest.TestCase):
    def test_recovers_planted_intercept_and_slope(self):
        rng = np.random.default_rng(11)
        n = 800
        xi = rng.uniform(-0.5, 0.5, n)
        cl = rng.integers(0, 40, n)
        w = rng.uniform(0.5, 2.0, n)
        y = 7.0 + 30.0 * xi + rng.normal(0, 0.5, n)
        (a, sa), (b, sb) = wls_intercept_slope_cr1(y, xi, w, cl)
        self.assertAlmostEqual(a, 7.0, delta=0.1)
        self.assertAlmostEqual(b, 30.0, delta=0.5)
        self.assertGreater(sa, 0.0)
        self.assertGreater(sb, 0.0)

    def test_noiseless_fit_is_exact_and_single_cluster_is_nan(self):
        xi = np.linspace(-0.5, 0.5, 50)
        y = -3.0 + 10.0 * xi
        (a, sa), (b, sb) = wls_intercept_slope_cr1(y, xi, np.ones(50), np.arange(50) % 5)
        self.assertAlmostEqual(a, -3.0, places=9)
        self.assertAlmostEqual(b, 10.0, places=9)
        self.assertAlmostEqual(sa, 0.0, places=6)
        (_, sa1), (_, sb1) = wls_intercept_slope_cr1(y, xi, np.ones(50), np.zeros(50))
        self.assertTrue(np.isnan(sa1) and np.isnan(sb1))

    def test_matches_weighted_mean_when_slope_regressor_is_constant_free(self):
        # with xi centred within the data, the intercept equals the weighted mean of y minus beta * weighted mean of xi
        rng = np.random.default_rng(12)
        xi = rng.uniform(-0.5, 0.5, 300)
        w = rng.uniform(0.5, 2.0, 300)
        y = 2.0 + 5.0 * xi + rng.normal(0, 1.0, 300)
        (a, _), (b, _) = wls_intercept_slope_cr1(y, xi, w, rng.integers(0, 20, 300))
        ym = float((w * y).sum() / w.sum())
        xm = float((w * xi).sum() / w.sum())
        self.assertAlmostEqual(a, ym - b * xm, places=9)


if __name__ == "__main__":
    unittest.main()
