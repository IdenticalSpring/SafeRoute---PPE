"""Image-level residual correction, independent of detector/regressor backends."""
from dataclasses import dataclass
from math import ceil, isfinite


def clip01(value):
    value = float(value)
    if not isfinite(value):
        raise ValueError("Risk/loss values must be finite")
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class Observation:
    dataset: str
    image_id: str
    group: tuple  # (meta_class, scale_group, crowd_group)
    observed_loss: float
    predicted_risk: float


@dataclass
class Calibration:
    global_q: float
    group_q: dict
    n_images: int
    group_n_images: dict

    def upper(self, predicted_risk, group):
        q = self.group_q.get(tuple(group), self.global_q)
        return clip01(clip01(predicted_risk) + q)


def upper_quantile(values, alpha=0.10):
    """Notebook convention: ceil((n+1)*(1-alpha)), capped at n.

    The capped small-sample rule is not a universal coverage guarantee.
    Empty calibration data is rejected in this public API.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    values = sorted(float(v) for v in values)
    if not values or not all(isfinite(v) for v in values):
        raise ValueError("Calibration requires nonempty finite residuals")
    k = min(len(values), ceil((len(values) + 1) * (1 - alpha)))
    return values[k - 1]


def fit_calibration(observations, alpha=0.10, min_group_images=20):
    """Fit ONE expert's correction using a held-out calibration partition.

    Multiple regions from an image contribute one maximum residual per group;
    global correction uses one maximum across all regions of that image.
    Dataset and image identity jointly identify an image.
    """
    if min_group_images < 1:
        raise ValueError("min_group_images must be positive")
    global_images, grouped = {}, {}
    for obs in observations:
        if not isfinite(obs.observed_loss) or not 0 <= obs.observed_loss <= 1:
            raise ValueError("Observed losses must be bounded in [0, 1]")
        key = (obs.dataset, obs.image_id)
        group = tuple(obs.group)
        if len(group) != 3:
            raise ValueError("group requires class, scale and crowd labels")
        residual = obs.observed_loss - clip01(obs.predicted_risk)
        global_images[key] = max(global_images.get(key, float('-inf')), residual)
        images = grouped.setdefault(group, {})
        images[key] = max(images.get(key, float('-inf')), residual)
    global_q = upper_quantile(global_images.values(), alpha)
    group_q = {
        group: upper_quantile(images.values(), alpha)
        for group, images in grouped.items() if len(images) >= min_group_images
    }
    return Calibration(global_q, group_q, len(global_images),
                       {g: len(images) for g, images in grouped.items()})
