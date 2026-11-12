import unittest

from careport.xfc7_decision import decide

H = ["frame", "lidar", "pose", "cam"]


def _cis(good):
    """good: the hypothesis whose CIs sit inside the margins; others far outside."""
    d, b = {}, {}
    for h in H:
        ok = h in good
        d[h] = {c: ((-1.0, 1.5) if ok else (12.0, 15.0)) for c in ["FRONT", "FRONT_LEFT", "SIDE_LEFT"]}
        b[h] = {r: ((-3.0, 4.0) if ok else (25.0, 35.0)) for r in ["L2R", "R2L"]}
    return d, b


class DecideTest(unittest.TestCase):
    def test_exactly_one(self):
        d, b = _cis({"lidar"})
        out = decide(d, b, ["FRONT", "FRONT_LEFT", "SIDE_LEFT"], ["L2R", "R2L"])
        self.assertEqual(out["supported"], ["lidar"])
        self.assertEqual(out["outcome"], "exactly one supported: lidar")

    def test_more_than_one_and_none(self):
        d, b = _cis({"lidar", "cam"})
        self.assertEqual(decide(d, b, ["FRONT", "FRONT_LEFT"], ["L2R"])["outcome"], "more than one supported: lidar, cam")
        d, b = _cis(set())
        self.assertEqual(decide(d, b, ["FRONT", "FRONT_LEFT"], ["L2R"])["outcome"], "none supported")

    def test_interval_must_lie_inside_margin(self):
        d, b = _cis({"lidar"})
        d["lidar"]["FRONT"] = (-1.0, 5.5)            # crosses +5 ms
        self.assertEqual(decide(d, b, ["FRONT", "FRONT_LEFT"], ["L2R"])["supported"], [])
        d, b = _cis({"lidar"})
        b["lidar"]["R2L"] = (-11.0, 2.0)             # crosses -10 ms per unit xi
        self.assertEqual(decide(d, b, ["FRONT", "FRONT_LEFT"], ["L2R", "R2L"])["supported"], [])
        self.assertEqual(decide(d, b, ["FRONT", "FRONT_LEFT"], ["L2R"])["supported"], ["lidar"])  # R2L not eligible

    def test_stop_rules(self):
        d, b = _cis({"lidar"})
        self.assertEqual(decide(d, b, ["FRONT"], ["L2R"])["outcome"], "stop: fewer than two decision cameras")
        self.assertEqual(decide(d, b, ["FRONT", "FRONT_LEFT"], [])["outcome"], "no slope cell passed G5: no hypothesis can be supported")


if __name__ == "__main__":
    unittest.main()
