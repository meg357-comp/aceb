# ACEB

ACEB (Anchored Calibrated Empirical Bayes) estimates latent item effects from
multiple noisy, partially overlapping measurement views. The implementation
models view-specific scale and residual noise,

```text
y_ij | theta_i ~ Normal(b_j + a_j theta_i, s_ij^2 + tau_j^2)
```

anchors one reference-view scale, and uses leave-view residual calibration by
default. This repository contains the runnable core implementation and a small,
deterministic synthetic benchmark for reviewing the accompanying submission.

## Install

ACEB requires Python 3.11 or newer.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Quick Start

Generate a tiny deterministic dataset and fit ACEB:

```bash
python examples/tiny/make_data.py
cmveb examples/tiny/config.yaml
```

The fit writes:

- `examples/tiny/output/posterior.parquet`: item-level posterior summaries.
- `examples/tiny/output/view_calibration.csv`: fitted view scales and residual
  noise.

The three input tables are:

| File | Required columns |
| --- | --- |
| `observations.parquet` | `item_id`, `view_id`, `estimate`, `standard_error` |
| `item_features.parquet` | `item_id`, followed by optional numeric features |
| `view_metadata.parquet` | `view_id`, `is_reference_view` (exactly one `True`) |

Each item-view pair must occur at most once, and standard errors must be
strictly positive.

## Verify

Run the core test suite:

```bash
pytest -q
```

Run the compact graph-design smoke benchmark:

```bash
python scripts/run_synthetic_graph.py \
  --config configs/smoke/synthetic_graph.yaml \
  --output-root results/smoke \
  --overwrite
```

This compares a rank-deficient star design with a repaired star-plus-edge
design and writes metrics, graph diagnostics, method diagnostics, and SVG
diagnostic plots under `results/smoke/synthetic_graph_smoke/`.

The tiny example and smoke benchmark are fully self-contained. Other configs
record larger or data-dependent experiment settings and may require optional
dependencies or external datasets.

## Repository Map

```text
src/cmveb/                  ACEB, baselines, graph diagnostics, and evaluation
configs/smoke/              Fast deterministic smoke configuration
examples/tiny/              Minimal end-to-end Parquet example
scripts/                    Reproducible benchmark entry points
stan/                       Optional full-Bayesian reference model
tests/                      Core unit and integration tests
docs/                       Metric definitions and implementation audit
.github/workflows/          Automated install and test check
```

For exact optimizer, calibration, anchoring, and numerical details, see the
[implementation contract](docs/ACEB_IMPLEMENTATION_CONTRACT_AUDIT.md). Metric
definitions are recorded in [docs/METRICS.md](docs/METRICS.md).

The optional Stan baseline requires a separate CmdStan installation; it is not
needed for ACEB or the quick-start workflow.
