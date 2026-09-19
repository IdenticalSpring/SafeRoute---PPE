import unittest
from saferoute.calibration import Observation, fit_calibration
from saferoute.routing import RoutingConfig, select_action
from saferoute.integration import Box, Detection, integrate


class CoreTests(unittest.TestCase):
    def test_regions_count_as_one_image_and_dataset_ids_are_distinct(self):
        g = ("helmet", "small", "low")
        data = [Observation("a", "1", g, .2, .1),
                Observation("a", "1", g, .7, .1),
                Observation("b", "1", g, .3, .1)]
        cal = fit_calibration(data, min_group_images=3)
        self.assertEqual(cal.n_images, 2)
        self.assertEqual(cal.group_q, {})
        self.assertAlmostEqual(cal.global_q, .6)
        self.assertEqual(cal.upper(.9, g), 1.)

    def test_group_specific_and_unknown_group_fallback(self):
        a, b = ("helmet","small","low"), ("vest","large","high")
        data = [Observation("d",str(i),a,.2,.1) for i in range(20)]
        data += [Observation("d","rare",b,.8,.1)]
        cal = fit_calibration(data, alpha=.01)
        self.assertAlmostEqual(cal.upper(.1,a),.2)
        self.assertAlmostEqual(cal.upper(.1,b),.8)

    def test_gate_acceptance_overrides_specialist_advantage(self):
        cfg = RoutingConfig(gate="gate")
        d = select_action({"gate":.2,"expert":0}, {"expert":0}, 10, cfg)
        self.assertEqual(d.reason, "gate_accepted")

    def test_cost_changes_expert_selection(self):
        cfg = RoutingConfig(gate="gate")
        risks = {"gate":.8,"slow":.1,"fast":.15}
        d = select_action(risks, {"slow":20,"fast":2},10,cfg)
        self.assertEqual(d.action, "fast")

    def test_bad_specialists_and_ties_retain_gate(self):
        cfg = RoutingConfig(gate="gate")
        d = select_action({"gate":.3,"expert":.3}, {"expert":0},10,cfg)
        self.assertEqual(d.action, "gate")
        cfg = RoutingConfig(gate="gate",min_improvement_margin=.2)
        d = select_action({"gate":.4,"expert":.3}, {"expert":0},10,cfg)
        self.assertEqual(d.action, "gate")

    def test_integration_replaces_earlier_specialist_and_keeps_outside(self):
        small = Box(0,0,20,20)
        large = Box(0,0,60,60)
        original = Detection(Box(2,2,8,8),"helmet",.5)
        first = Detection(Box(3,3,9,9),"helmet",.8)
        final = Detection(Box(4,4,10,10),"helmet",.7)
        outside = Detection(Box(80,80,90,90),"helmet",.9)
        out = integrate([original,outside],[(small,[final]),(large,[first])])
        self.assertEqual(set(out),{outside,final})

    def test_nms_retains_different_classes(self):
        box = Box(0,0,10,10)
        ds = [Detection(box,"helmet",.8),Detection(box,"helmet",.7),
              Detection(box,"head",.9)]
        self.assertEqual(len(integrate(ds,[])),2)

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_calibration([])
        with self.assertRaises(ValueError):
            select_action({"gate":.5,"expert":float('nan')},{"expert":1},1,
                          RoutingConfig(gate="gate"))


if __name__ == "__main__":
    unittest.main()
