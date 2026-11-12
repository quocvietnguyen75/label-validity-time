"""Tests for the XF-C7 post hoc set V6R (part 1: av2-api's default sweep-to-image pairing). Written before the code."""
import unittest

import numpy as np
import pandas as pd

from careport import xfc7_t3_posthoc_v6r as v

MS = 1_000_000  # nanoseconds per millisecond


class NearestWithinTests(unittest.TestCase):
    def test_picks_nearest_image_within_tolerance(self):
        sweeps = np.array([0, 100 * MS, 200 * MS], dtype=np.int64)
        images = np.array([-22 * MS, 28 * MS, 78 * MS, 128 * MS, 178 * MS, 228 * MS], dtype=np.int64)
        matched, ok = v.nearest_within(sweeps, images, 25 * MS)
        self.assertTrue(ok.all())
        self.assertEqual(matched.tolist(), [-22 * MS, 78 * MS, 178 * MS])

    def test_no_match_beyond_tolerance(self):
        sweeps = np.array([0], dtype=np.int64)
        images = np.array([-30 * MS, 40 * MS], dtype=np.int64)
        matched, ok = v.nearest_within(sweeps, images, 25 * MS)
        self.assertFalse(ok[0])

    def test_empty_images(self):
        matched, ok = v.nearest_within(np.array([0, MS], dtype=np.int64), np.array([], dtype=np.int64), 25 * MS)
        self.assertFalse(ok.any())

    def test_agrees_with_pandas_merge_asof_as_av2_api_uses_it(self):
        rng = np.random.default_rng(20261108)
        sweeps = np.sort(rng.integers(0, 10_000 * MS, size=400)).astype(np.int64)
        images = np.sort(rng.integers(0, 10_000 * MS, size=700)).astype(np.int64)
        tol = 25 * MS
        matched, ok = v.nearest_within(sweeps, images, tol)
        left = pd.DataFrame({"lidar": pd.to_timedelta(sweeps)})
        right = pd.DataFrame({"cam": pd.to_timedelta(images)})
        ref = pd.merge_asof(left, right, left_on="lidar", right_on="cam", direction="nearest",
                            tolerance=pd.to_timedelta(tol))
        ref_ok = ref["cam"].notna().to_numpy()
        self.assertEqual(ok.tolist(), ref_ok.tolist())
        ref_ns = ref["cam"][ref_ok].astype("int64").to_numpy()
        self.assertEqual(matched[ok].tolist(), ref_ns.tolist())


class SummaryTests(unittest.TestCase):
    def test_summarize(self):
        s = v.summarize(np.array([-22.0, -21.0, -23.0, 10.0]))
        self.assertEqual(s["n"], 4)
        self.assertAlmostEqual(s["median"], -21.5)
        self.assertAlmostEqual(s["frac_negative"], 0.75)
        self.assertAlmostEqual(s["min"], -23.0)
        self.assertAlmostEqual(s["max"], 10.0)

    def test_compare_leave_versus_retime(self):
        c = v.compare(52.1, -21.9)
        self.assertAlmostEqual(c["leave_unchanged_ms"], 74.0)
        self.assertAlmostEqual(c["retime_from_sweep_start_ms"], 52.1)
        self.assertTrue(c["leave_errs_more"])
        c = v.compare(41.1, 13.3)
        self.assertAlmostEqual(c["leave_unchanged_ms"], 27.8)
        self.assertFalse(c["leave_errs_more"])


class ArithmeticTests(unittest.TestCase):
    """V6R-2 to V6R-7 helpers (Addendum 2): arithmetic on stored values only."""

    def test_clearance_positive_side(self):
        self.assertAlmostEqual(v.clearance(47.632172036420194, 56.47056110220198, 20.148874204930923),
                               47.632172036420194 - 20.148874204930923)

    def test_clearance_negative_side(self):
        self.assertAlmostEqual(v.clearance(-30.3, -21.6, 20.1), 1.5)

    def test_clearance_inside_band_is_negative(self):
        self.assertAlmostEqual(v.clearance(-10.0, 10.0, 20.0), -30.0)

    def test_second_largest_and_smallest(self):
        self.assertAlmostEqual(v.second_largest([16.8, 4.7, 7.3]), 7.3)
        self.assertAlmostEqual(v.second_smallest([29.8, 29.1, 52.6, 24.3]), 29.1)

    def test_ci_accepts_dict_or_list(self):
        self.assertEqual(v.ci({"est": 1.0, "lo": 0.5, "hi": 1.5, "se": 0.2}), (1.0, 0.5, 1.5))
        self.assertEqual(v.ci([1.0, 0.5, 1.5]), (1.0, 0.5, 1.5))

    def test_speed_summary(self):
        s = v.speed_summary(np.array([3.0, 4.0, 5.0, 6.0, 100.0]))
        self.assertEqual(s["n"], 5)
        self.assertAlmostEqual(s["median"], 5.0)
        self.assertAlmostEqual(s["p90"], float(np.percentile([3.0, 4.0, 5.0, 6.0, 100.0], 90)))


class CompositionTests(unittest.TestCase):
    """V6R-8 helpers (Addendum 3)."""

    def test_intermediate_compositions(self):
        self.assertAlmostEqual(v.lin_intermediate(16.0, 11.0, 1.5), 27.0)
        self.assertAlmostEqual(v.quad_intermediate(3.0, 4.0, 1.0), 5.0)
        self.assertAlmostEqual(v.lin_intermediate(16.0, 1.0, 11.0), 27.0)

    def test_rejected(self):
        self.assertTrue(v.rejected(30.0, 40.0, 25.0))
        self.assertTrue(v.rejected(-40.0, -30.0, 25.0))
        self.assertFalse(v.rejected(20.0, 40.0, 25.0))

    def test_tipping_is_second_largest_share(self):
        shares = [26.0 / 52.0, 9.0 / 30.0, 1.0 / 40.0, 7.0 / 35.0]
        self.assertAlmostEqual(v.tipping([26.0, 9.0, 1.0, 7.0], [52.0, 30.0, 40.0, 35.0]), sorted(shares)[-2])


class ParseTests(unittest.TestCase):
    def test_parse_timestamps_keeps_only_matching_suffix(self):
        got = v.parse_timestamps(["315969904387425431.jpg", "notes.txt", "315969904437425441.jpg"], ".jpg")
        self.assertEqual(got.tolist(), [315969904387425431, 315969904437425441])
        self.assertEqual(got.dtype, np.int64)


if __name__ == "__main__":
    unittest.main()
