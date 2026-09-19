"""Risk and incremental-latency selection; no test-ground-truth access."""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class RoutingConfig:
    gate: str = "PP-PicoDet-S"
    risk_tolerance: float = 0.20
    latency_lambda: float = 0.12
    min_improvement_margin: float = 0.0

    def __post_init__(self):
        vals = (self.risk_tolerance, self.latency_lambda, self.min_improvement_margin)
        if not all(isfinite(v) for v in vals):
            raise ValueError("Configuration must be finite")
        if not 0 <= self.risk_tolerance <= 1 or min(vals[1:]) < 0:
            raise ValueError("Invalid routing configuration")


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    objectives: dict


def select_action(risks, specialist_cost_ms, reference_ms, config=RoutingConfig()):
    """Costs MUST be estimates available before executing candidate specialists.

    Use exactly the same estimated costs during accuracy and timing evaluation.
    The already-executed gate incurs zero incremental detector cost.
    """
    if config.gate not in risks or len(risks) < 2:
        raise ValueError("Supply gate and at least one specialist risk")
    if not isfinite(reference_ms) or reference_ms <= 0:
        raise ValueError("reference_ms must be finite and positive")
    if any(not isfinite(v) or not 0 <= v <= 1 for v in risks.values()):
        raise ValueError("Calibrated risks must lie in [0, 1]")
    objective = {config.gate: float(risks[config.gate])}
    for expert in sorted(set(risks) - {config.gate}):
        cost = specialist_cost_ms[expert]
        if not isfinite(cost) or cost < 0:
            raise ValueError("Specialist costs must be finite and nonnegative")
        objective[expert] = risks[expert] + config.latency_lambda * cost / reference_ms
    if risks[config.gate] <= config.risk_tolerance:
        return Decision(config.gate, "gate_accepted", objective)
    # Gate wins objective ties. Specialist ties use alphabetical order.
    best = min(objective, key=objective.get)
    improvement = objective[config.gate] - objective[best]
    if best == config.gate or improvement < config.min_improvement_margin:
        return Decision(config.gate, "insufficient_benefit", objective)
    return Decision(best, "specialist_selected", objective)
