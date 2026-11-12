import unittest

import numpy as np
import pandas as pd

from careport.xfc7_gate import subsample_tracks_to_lengths


class SubsampleTest(unittest.TestCase):
    def test_contiguous_blocks_with_drawn_lengths(self):
        rows = [("s1", "a", f) for f in range(100)] + [("s1", "b", f) for f in range(5)] + [("s2", "a", f) for f in range(30)]
        df = pd.DataFrame(rows, columns=["segment", "object_id", "frame"])
        out = subsample_tracks_to_lengths(df, ["segment", "object_id"], np.array([10]), seed=1)
        n = out.groupby(["segment", "object_id"])["frame"].agg(["size", "min", "max"])
        self.assertEqual(n.loc[("s1", "a"), "size"], 10)
        self.assertEqual(n.loc[("s1", "b"), "size"], 5)            # shorter than the drawn length: kept whole
        self.assertEqual(n.loc[("s2", "a"), "size"], 10)
        self.assertEqual(n.loc[("s1", "a"), "max"] - n.loc[("s1", "a"), "min"], 9)   # contiguous block

    def test_seeded_and_reproducible(self):
        df = pd.DataFrame([("s", t, f) for t in range(20) for f in range(50)], columns=["segment", "object_id", "frame"])
        a = subsample_tracks_to_lengths(df, ["segment", "object_id"], np.array([3, 7, 12]), seed=20261021)
        b = subsample_tracks_to_lengths(df, ["segment", "object_id"], np.array([3, 7, 12]), seed=20261021)
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
        self.assertTrue(set(a.groupby(["segment", "object_id"]).size()) <= {3, 7, 12})


if __name__ == "__main__":
    unittest.main()
