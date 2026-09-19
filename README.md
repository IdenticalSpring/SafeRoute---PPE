# SafeRoute-PPE: partial reference implementation

This repository releases selected mechanics of region-level risk-aware routing.
It is **not the complete experimental pipeline**, and it cannot reproduce the
manuscript tables on its own. The demo uses synthetic numbers and boxes only.

## Included

- Expert-specific residual corrections aggregated by image, with group fallback.
- Gate acceptance and specialist selection using calibrated loss and estimated latency.
- Sequential regional prediction replacement and class-wise NMS.
- Selected routing parameters, a synthetic example, and focused unit tests.
- Dependency-free Python 3.10+ source; no GPU, installation or account required.

## Not included

Detector training/fine-tuning, trained detector or risk-regressor weights,
feature extraction and proposal generation, loss-target construction, dataset
exports and experimental split manifests, benchmark/ablation/transfer suites,
private experiment logs and manuscript-result tables are not part of this release.
This repository neither verifies nor regenerates the manuscript's reported results.

## Run

Windows: double-click `run_demo.bat`, or run `python demo.py`.
Linux/macOS: `bash run_demo.sh`.
Tests: `python -m unittest discover -s tests -v`.

The example demonstrates three outcomes: gate acceptance, specialist selection,
and retention because no specialist offers sufficient benefit. Example costs are
explicitly synthetic, not measured hardware performance. Nothing is downloaded.

## Integration contract

1. Supply numerical predicted losses for every candidate expert and the gate.
   A trained risk estimator is intentionally not included.
2. Fit one calibration object per expert on a held-out calibration partition.
   Supply bounded observed losses, image identities and `(meta_class, scale,
   crowd)` group labels through `Observation`. Do not use test labels here.
3. Apply `Calibration.upper` to the predictions for a new region.
4. Supply **expert-specific estimated costs available before expert execution**
   to `select_action`. Use the same cost estimates in accuracy and latency runs.
   The denominator is a positive reference latency; the gate's incremental cost
   is zero because the gate has already executed.
5. Execute the selected specialist only when the decision calls for it.
6. Restore crop predictions to full-image coordinates, then pass only escalated
   regions and their outputs to `integrate`.

The synthetic demo provides executable examples of these APIs. It does not
include detector adapters, image processing, or a live inference service.

## Method details and scope

Calibration uses the maximum residual per image within each group, followed by
an upper order statistic. Groups need at least 20 distinct images; other groups
use an expert-level global correction. Dataset and image ID jointly identify an
image. Predicted risks and corrected estimates are clipped to [0, 1].

The quantile rank follows the supplied notebook's capped convention:
`min(n, ceil((n+1)*(1-alpha)))`. This convention, sparse fallback, and adaptive
expert selection must not be advertised as an unconditional finite-sample safety
guarantee. Here risk means weighted detection loss, not accident probability.

Gate acceptance uses `risk <= 0.20`. Above this tolerance, objectives are
`risk + 0.12 * estimated_cost / reference_cost`. The gate wins ties; a specialist
must beat the gate, and its improvement must be at least the configured margin.
The released margin is zero. Specialist ties use alphabetical order, making the
reference API deterministic (the notebook uses its configured action order).

Integration processes larger regions first. A later routed region can replace
predictions inserted by an earlier specialist. Equal-area regions keep input
order. NMS is class-wise with IoU threshold 0.60. This is sequential replacement,
not a simultaneous union of every expert's output.

This is a refactored, standalone reference, not a byte-for-byte export of the
research notebook. It adds input validation, rejects empty calibration data, and
accepts caller-supplied cost estimates rather than running candidate experts to
obtain costs. It does not claim to repair or validate the full experimental suite.

## Availability statement for the manuscript

A partial reference implementation of the regional calibration, expert-selection,
and prediction-integration components is available at [repository URL]. The
release includes a synthetic demonstration; trained models and the complete
experimental pipeline are not included.

## Publishing

Upload only this folder's contents to your public repository. The original
notebook, credentials, datasets, checkpoints, and results are absent. The ignore
file also excludes common experiment artifacts, but does not remove files already
tracked in an existing repository. No repository has been created or published by
this package. No software license is imposed; choose a license appropriate to your
ownership and intended reuse before calling the release open source.
