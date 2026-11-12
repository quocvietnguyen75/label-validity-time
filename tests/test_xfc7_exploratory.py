import unittest

import numpy as np
import pandas as pd

from careport.xfc7_exploratory_run import e1


def _static(n, inter, rng):
    rows = []
    for ro, k in ((2, inter[0]), (4, inter[1])):
        xi = rng.uniform(-0.5, 0.5, n)
        v = rng.uniform(0, 15, n)
        cam = rng.integers(1, 4, n)
        rows.append(pd.DataFrame({"readout": ro, "xi": xi, "ego_speed": v, "camera": cam, "segment": rng.integers(0, 40, n),
                                  "w": 1.0, "r": k * xi * v + 3.0 * cam + rng.normal(0, 0.5, n)}))
    return pd.concat(rows, ignore_index=True)


def _moving(rng):
    n = 200
    return pd.DataFrame({"readout": np.repeat([2, 4], n // 2), "w": 1.0, "s": 1.0, "sdot": 2.0, "ego_speed": 10.0})


class E1Test(unittest.TestCase):
    def test_planted_opposite_interactions_appear(self):
        rng = np.random.default_rng(5)
        out = e1(_static(3000, (2.0, -2.0), rng), _moving(rng))
        self.assertTrue(out["appears"])
        self.assertAlmostEqual(out["L2R"]["b_xi_x_ego_speed_px_per_mps"]["est"], 2.0, delta=0.05)
        self.assertAlmostEqual(out["L2R"]["implied_moving_slope_ms_per_xi"], 2.0 * 10.0 / 2.0, delta=0.5)
        self.assertAlmostEqual(out["implied_conflict_ms_per_xi"], 20.0, delta=1.0)

    def test_no_interaction_does_not_appear(self):
        rng = np.random.default_rng(6)
        self.assertFalse(e1(_static(3000, (0.0, 0.0), rng), _moving(rng))["appears"])


if __name__ == "__main__":
    unittest.main()
