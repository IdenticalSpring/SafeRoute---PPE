"""Synthetic mechanics demonstration. Contains no trained models or paper results."""
import json
from dataclasses import asdict
from pathlib import Path
from saferoute.calibration import Observation, fit_calibration
from saferoute.routing import RoutingConfig, select_action
from saferoute.integration import Box, Detection, integrate


def main():
    settings = json.loads((Path(__file__).parent / "config.json").read_text())
    cfg = RoutingConfig(**settings["routing"])
    group = ("helmet", "small", "low")
    experts = [cfg.gate, "YOLOv12-S", "RTMDet-tiny", "RT-DETRv2-S"]
    # Entirely artificial held-out residuals: two regions from each of 24 images.
    calibrations = {}
    for expert in experts:
        observations = []
        for i in range(24):
            observations.extend([
                Observation("synthetic", str(i), group, 0.20, 0.18),
                Observation("synthetic", str(i), group, 0.25, 0.20),
            ])
        calibrations[expert] = fit_calibration(observations, **settings["calibration"])
    costs = settings["synthetic_demo_costs_ms"]
    scenarios = [
        ("easy", [0.10, 0.03, 0.04, 0.02]),
        ("difficult", [0.65, 0.12, 0.15, 0.10]),
        ("no_beneficial_specialist", [0.30, 0.30, 0.35, 0.40]),
    ]
    report = {"scope": "SYNTHETIC DEMO ONLY - not manuscript results", "decisions": []}
    for name, predictions in scenarios:
        risks = {a: calibrations[a].upper(p, group) for a, p in zip(experts, predictions)}
        decision = select_action(risks, costs, settings["synthetic_reference_ms"], cfg)
        report["decisions"].append({"scenario": name, "risks": risks, **asdict(decision)})
    gate = [Detection(Box(5,5,15,15), "helmet", 0.60),
            Detection(Box(80,80,90,90), "helmet", 0.85)]
    specialist = [Detection(Box(6,6,16,16), "helmet", 0.95)]
    merged = integrate(gate, [(Box(0,0,40,40), specialist)], settings["final_nms_iou"])
    report["synthetic_integration"] = [asdict(d) for d in merged]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
