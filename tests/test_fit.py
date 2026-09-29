import unittest
from fit import placement_fit, passage_fit

class TestFitLogic(unittest.TestCase):
    def test_placement_fit_obvious_fit(self):
        item = {"ok": True, "L_cm": 100, "W_cm": 50, "H_cm": 50, "sigma_pct": 0.05}
        space = {"ok": True, "width_cm": 200, "depth_cm": 100, "sigma_pct": 0.05}
        res = placement_fit(item, space)
        self.assertEqual(res["verdict"], "fits")
        self.assertGreater(res["margin_cm"], 15)

    def test_placement_fit_obvious_no_fit(self):
        item = {"ok": True, "L_cm": 200, "W_cm": 100, "H_cm": 50, "sigma_pct": 0.05}
        space = {"ok": True, "width_cm": 150, "depth_cm": 80, "sigma_pct": 0.05}
        res = placement_fit(item, space)
        self.assertEqual(res["verdict"], "doesn't fit")
        self.assertLess(res["margin_cm"], 0)

    def test_placement_fit_only_rotated(self):
        item = {"ok": True, "L_cm": 100, "W_cm": 50, "H_cm": 50, "sigma_pct": 0.05}
        space = {"ok": True, "width_cm": 70, "depth_cm": 120, "sigma_pct": 0.05}
        res = placement_fit(item, space)
        self.assertEqual(res["verdict"], "fits")
        self.assertIn("90 degrees", res["detail"])
        
    def test_placement_fit_razor_thin(self):
        item = {"ok": True, "L_cm": 100, "W_cm": 50, "H_cm": 50, "sigma_pct": 0.05}
        space = {"ok": True, "width_cm": 102, "depth_cm": 55, "sigma_pct": 0.05}
        res = placement_fit(item, space)
        self.assertEqual(res["verdict"], "tight")
        self.assertEqual(res["margin_cm"], 2)

    def test_passage_fit_straight_on(self):
        item = {"ok": True, "L_cm": 100, "W_cm": 50, "H_cm": 50, "sigma_pct": 0.05}
        space = {"ok": True, "width_cm": 80, "height_cm": 200, "sigma_pct": 0.05}
        res = passage_fit(item, space)
        self.assertEqual(res["verdict"], "fits")
        self.assertGreater(res["margin_cm"], 15)

    def test_passage_fit_needs_tilt(self):
        # A long object that only fits if tilted a bit (maybe difficult to set up, let's test a simple tilt case)
        item = {"ok": True, "L_cm": 200, "W_cm": 60, "H_cm": 60, "sigma_pct": 0.05}
        space = {"ok": True, "width_cm": 80, "height_cm": 210, "sigma_pct": 0.05}
        res = passage_fit(item, space)
        self.assertEqual(res["verdict"], "fits")

    def test_passage_fit_cannot_pass(self):
        item = {"ok": True, "L_cm": 200, "W_cm": 100, "H_cm": 100, "sigma_pct": 0.05}
        space = {"ok": True, "width_cm": 80, "height_cm": 200, "sigma_pct": 0.05}
        res = passage_fit(item, space)
        self.assertEqual(res["verdict"], "doesn't fit")

    def test_confidence_tiers(self):
        # Big margin -> High
        item = {"ok": True, "L_cm": 50, "W_cm": 50, "H_cm": 50, "sigma_pct": 0.05}
        space = {"ok": True, "width_cm": 200, "depth_cm": 200, "sigma_pct": 0.05}
        res = placement_fit(item, space)
        self.assertEqual(res["tier"], "High")

        # Razor thin -> Low
        space2 = {"ok": True, "width_cm": 52, "depth_cm": 52, "sigma_pct": 0.05}
        res2 = placement_fit(item, space2)
        self.assertEqual(res2["tier"], "Low")

if __name__ == '__main__':
    unittest.main()
