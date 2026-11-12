import unittest

import numpy as np
import pandas as pd

from careport.xfc7_budget import ego_spread, neighbour_spread

K0 = ["camera", "viewpoint_bin", "range_bin", "column_third", "ego_class"]


def _table():
    rows = []
    # camera 1, column 1, ego 0: viewpoints 5,6,7 at range 1, plus range 0 and 2 at viewpoint 6
    for vp, rg, z in [(5, 1, 2.0), (6, 1, 3.0), (7, 1, 7.0), (6, 0, 4.0), (6, 2, 3.5)]:
        rows.append((1, vp, rg, 1, 0, z, 40))
    rows.append((1, 8, 1, 1, 0, 50.0, 5))          # too few observations: ignored as a neighbour
    # ego classes for the bin (1, 6, 1, 1): 3.0 (ego 0), 1.0 (ego 1), 6.0 (ego 2, but only 10 obs: ignored)
    rows.append((1, 6, 1, 1, 1, 1.0, 35))
    rows.append((1, 6, 1, 1, 2, 6.0, 10))
    return pd.DataFrame(rows, columns=K0 + ["z_b", "n_b"])


class BudgetTest(unittest.TestCase):
    def test_neighbour_spread_uses_valid_viewpoint_and_range_neighbours(self):
        obs = pd.DataFrame([(1, 6, 1, 1, 0), (1, 7, 1, 1, 0)], columns=K0)
        s = neighbour_spread(_table(), obs, min_n=30)
        # bin (6,1): neighbours vp5 (2.0), vp7 (7.0), rg0 (4.0), rg2 (3.5) -> max |3 - x| = 4.0
        self.assertAlmostEqual(s[0], 4.0)
        # bin (7,1): neighbours vp6 (3.0); vp8 ignored (n=5); no range neighbours at vp7 -> 4.0
        self.assertAlmostEqual(s[1], 4.0)

    def test_ego_spread_within_bin(self):
        obs = pd.DataFrame([(1, 6, 1, 1, 2), (1, 5, 1, 1, 0)], columns=K0)
        s = ego_spread(_table(), obs, min_n=30)
        self.assertAlmostEqual(s[0], 2.0)      # classes 0 (3.0) and 1 (1.0); class 2 ignored
        self.assertTrue(np.isnan(s[1]))        # only one valid class


if __name__ == "__main__":
    unittest.main()
