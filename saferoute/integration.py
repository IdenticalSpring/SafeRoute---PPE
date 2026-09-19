"""Sequential regional replacement in original-image coordinates."""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self):
        if not all(isfinite(v) for v in (self.x1, self.y1, self.x2, self.y2)):
            raise ValueError("Box coordinates must be finite")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Box must have positive width and height")

    @property
    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def contains_center(self, other):
        cx, cy = (other.x1 + other.x2) / 2, (other.y1 + other.y2) / 2
        return self.x1 <= cx < self.x2 and self.y1 <= cy < self.y2


@dataclass(frozen=True)
class Detection:
    box: Box
    label: str
    score: float

    def __post_init__(self):
        if not isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("Detection score must lie in [0, 1]")


def iou(a, b):
    intersection = max(0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(
        0, min(a.y2, b.y2) - max(a.y1, b.y1))
    return intersection / (a.area + b.area - intersection)


def classwise_nms(detections, threshold=0.60):
    if not 0 < threshold <= 1:
        raise ValueError("NMS threshold must lie in (0, 1]")
    pending = sorted(detections, key=lambda d: d.score, reverse=True)
    kept = []
    while pending:
        best = pending.pop(0)
        kept.append(best)
        pending = [d for d in pending
                   if d.label != best.label or iou(d.box, best.box) < threshold]
    return kept


def integrate(gate_detections, routed_outputs, nms_iou=0.60):
    """routed_outputs: (region Box, specialist detections in full-image coords).

    Supply only escalated regions. Larger regions are processed first. Each
    replacement removes overlapping-centre predictions already accumulated,
    including earlier specialist outputs. Equal-area ties preserve input order.
    Crop decoding/resizing/coordinate restoration are the caller's responsibility.
    """
    out = list(gate_detections)
    for region, predictions in sorted(routed_outputs, key=lambda item: item[0].area,
                                      reverse=True):
        out = [d for d in out if not region.contains_center(d.box)]
        out.extend(predictions)
    return classwise_nms(out, nms_iou)
