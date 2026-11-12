"""Synthetic tests for the post hoc set R4 (pilot record, 2026-11-03: R18 small-sample intervals, R17 copying threshold,
R29 tolerance-adjusted misplacement), written before the module."""
import math
import unittest

import numpy as np

from careport import xfc7_t2_posthoc_r4 as r4
from careport.xfc7_estimators import weighted_mean_cr1


def _oracle_cr2_bm(y, w, cluster):
    """Brute-force CR2 and Bell-McCaffrey df for the weighted mean, from full matrices in the whitened space
    (working variance proportional to 1/w): y~ = sqrt(w) y, X~ = sqrt(w)."""
    y, w = np.asarray(y, float), np.asarray(w, float)
    cluster = np.asarray(cluster)
    n = len(y)
    x = np.sqrt(w)
    yt = x * y
    S = float(x @ x)
    mu = float(x @ yt) / S
    e = yt - x * mu
    H = np.outer(x, x) / S
    IH = np.eye(n) - H
    groups = [np.where(cluster == g)[0] for g in dict.fromkeys(cluster.tolist())]
    meat = 0.0
    cols = []
    for idx in groups:
        Hgg = H[np.ix_(idx, idx)]
        vals, vecs = np.linalg.eigh(np.eye(len(idx)) - Hgg)
        A = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
        u = x[idx] @ A @ e[idx]
        meat += u * u
        cols.append(IH[idx, :].T @ (A @ x[idx]) / S)
    se = math.sqrt(meat) / S
    Gm = np.column_stack(cols)
    M = Gm.T @ Gm
    df = float(np.trace(M) ** 2 / np.trace(M @ M))
    return mu, se, df


class Cr2BmTest(unittest.TestCase):
    def test_balanced_equals_cr1_and_g_minus_1(self):
        rng = np.random.default_rng(1)
        G, m = 8, 5
        cluster = np.repeat(np.arange(G), m)
        y = rng.normal(10.0, 3.0, G * m)
        w = np.ones(G * m)
        out = r4.cr2_bm_weighted_mean(y, w, cluster)
        mu, se1 = weighted_mean_cr1(y, w, cluster)
        self.assertAlmostEqual(out["est"], mu, places=12)
        self.assertAlmostEqual(out["se"], se1, places=12)
        self.assertAlmostEqual(out["df"], G - 1, places=9)

    def test_matches_matrix_oracle_with_unequal_clusters_and_weights(self):
        rng = np.random.default_rng(2)
        sizes = [1, 2, 7, 3, 12, 4, 1, 6]
        cluster = np.concatenate([np.full(s, g) for g, s in enumerate(sizes)])
        y = rng.normal(0.0, 5.0, len(cluster))
        w = rng.uniform(0.05, 4.0, len(cluster))
        out = r4.cr2_bm_weighted_mean(y, w, cluster)
        mu, se, df = _oracle_cr2_bm(y, w, cluster)
        self.assertAlmostEqual(out["est"], mu, places=10)
        self.assertAlmostEqual(out["se"], se, places=10)
        self.assertAlmostEqual(out["df"], df, places=8)
        self.assertAlmostEqual(out["lo"], mu - out["t"] * se, places=10)

    def test_dominant_cluster_shrinks_df_and_widens_interval(self):
        rng = np.random.default_rng(3)
        G = 12
        cluster = np.repeat(np.arange(G), 4)
        y = rng.normal(0.0, 1.0, G * 4)
        w = np.ones(G * 4)
        w[cluster == 0] = 40.0                                  # one cluster carries most of the weight
        out = r4.cr2_bm_weighted_mean(y, w, cluster)
        self.assertLess(out["df"], 4.0)
        self.assertLess(out["kish_G"], 2.0)
        mu, se1 = weighted_mean_cr1(y, w, cluster)
        cr1_half = 1.96 * se1
        self.assertGreater(out["hi"] - out["lo"], 2 * cr1_half)

    def test_kish_effective_clusters(self):
        cluster = np.array([0, 0, 1, 2])
        w = np.array([1.0, 1.0, 1.0, 1.0])                    # shares 0.5, 0.25, 0.25
        self.assertAlmostEqual(r4.kish_effective_clusters(w, cluster), 1.0 / (0.25 + 0.0625 + 0.0625))


class WebbBootstrapTest(unittest.TestCase):
    def test_webb_weights(self):
        v = r4.webb_weights(np.random.default_rng(0), (200000,))
        self.assertEqual(sorted(set(np.round(np.abs(v), 6))), sorted({round(math.sqrt(0.5), 6), 1.0, round(math.sqrt(1.5), 6)}))
        self.assertAlmostEqual(float(v.mean()), 0.0, places=2)
        self.assertAlmostEqual(float((v * v).mean()), 1.0, places=2)

    def test_deterministic_and_contains_estimate(self):
        rng = np.random.default_rng(4)
        cluster = np.repeat(np.arange(15), 6)
        y = rng.normal(20.0, 4.0, 90)
        w = rng.uniform(0.5, 2.0, 90)
        a = r4.wild_cluster_bootstrap_t(y, w, cluster, draws=2999, seed=20261103)
        b = r4.wild_cluster_bootstrap_t(y, w, cluster, draws=2999, seed=20261103)
        self.assertEqual(a, b)
        self.assertLess(a["lo"], a["est"])
        self.assertGreater(a["hi"], a["est"])

    def test_close_to_cr1_t_interval_when_clusters_are_many_and_balanced(self):
        rng = np.random.default_rng(5)
        G = 120
        cluster = np.repeat(np.arange(G), 5)
        y = rng.normal(0.0, 1.0, G * 5)
        w = np.ones(G * 5)
        bt = r4.wild_cluster_bootstrap_t(y, w, cluster, draws=9999, seed=20261103)
        mu, se1 = weighted_mean_cr1(y, w, cluster)
        self.assertAlmostEqual((bt["hi"] - bt["lo"]) / (2 * 1.98 * se1), 1.0, delta=0.08)


class DecisionTest(unittest.TestCase):
    def test_rule_applied_to_corrected_intervals(self):
        intervals = {"start": {"c1": (40.0, 30.0, 50.0), "c2": (70.0, 46.0, 94.0)}}
        out = r4.decide(intervals, {"c1": 19.0, "c2": 42.0}, ["c1", "c2"])
        self.assertEqual(out["start"]["rejected_in"], ["c1"])          # c2: 46 < 47 = 5 + 42
        self.assertFalse(out["start"]["ruled_out"])


class CopyingThresholdTest(unittest.TestCase):
    def test_binding_value_is_second_largest(self):
        self.assertAlmostEqual(r4.binding_value([0.54, 0.23, 0.03, 0.29]), 0.29)
        self.assertAlmostEqual(r4.binding_value([0.51, 0.25, 0.03, 0.27]), 0.27)

    def test_threshold_formula(self):
        # rho >= (lo - (5 + B_c)) / tau_seed
        self.assertAlmostEqual(r4.copying_threshold(47.75, 21.28, 49.14), (47.75 - 21.28) / 49.14)


class MisplacementTest(unittest.TestCase):
    def test_subtracts_b_c_not_the_negligible_band(self):
        self.assertAlmostEqual(r4.tolerance_adjusted_misplacement_m(34.3, 19.2, 10.0), 0.151)
        self.assertAlmostEqual(r4.tolerance_adjusted_misplacement_m(54.8, 42.2, 10.0), 0.126)


if __name__ == "__main__":
    unittest.main()
