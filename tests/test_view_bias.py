from __future__ import annotations

import numpy as np
import pandas as pd

from cmveb.graph_diagnostics import build_coobservation_graph
from cmveb.indexing import build_indexed_data
from cmveb.models.baseline import anchored_calibrated_eb
from cmveb.posterior import compute_posterior_from_prior, marginal_log_likelihood
from cmveb.preprocess import center_estimates_by_view, prepare_data


def make_data(estimates):
    observations = pd.DataFrame(
        {
            "item_id": ["a", "a", "b", "b", "c", "c"],
            "view_id": ["v0", "v1", "v0", "v1", "v0", "v1"],
            "estimate": estimates,
            "standard_error": [0.4, 0.3, 0.5, 0.4, 0.2, 0.3],
        }
    )
    item_features = pd.DataFrame({"item_id": ["a", "b", "c"], "x": 1.0})
    view_metadata = pd.DataFrame({"view_id": ["v0", "v1"], "is_reference_view": [True, False]})
    return build_indexed_data(prepare_data(observations, item_features, view_metadata, standardize_features=False))


def test_posterior_with_bias_equals_manually_centered_posterior() -> None:
    data = make_data([1.0, 3.0, -0.5, 0.7, 2.0, 4.4])
    view_bias = np.array([0.2, 1.5])
    centered = make_data((data.estimate - view_bias[data.view_idx]).tolist())
    prior_mean = np.array([0.1, -0.2, 0.3])
    prior_variance = np.array([1.0, 0.7, 1.2])
    view_scale = np.array([1.0, 0.6])
    extra_noise = np.array([0.1, 0.2])

    with_bias = compute_posterior_from_prior(
        data,
        prior_mean,
        prior_variance,
        view_scale,
        extra_noise,
        view_bias=view_bias,
    )
    manual = compute_posterior_from_prior(centered, prior_mean, prior_variance, view_scale, extra_noise)
    np.testing.assert_allclose(with_bias.mean, manual.mean)
    np.testing.assert_allclose(with_bias.variance, manual.variance)


def test_marginal_likelihood_with_bias_equals_shifted_no_bias_likelihood() -> None:
    data = make_data([1.0, 3.0, -0.5, 0.7, 2.0, 4.4])
    view_bias = np.array([0.2, 1.5])
    centered = make_data((data.estimate - view_bias[data.view_idx]).tolist())
    prior_mean = np.array([0.1, -0.2, 0.3])
    prior_variance = np.array([1.0, 0.7, 1.2])
    view_scale = np.array([1.0, 0.6])
    extra_noise = np.array([0.1, 0.2])

    with_bias = marginal_log_likelihood(data, prior_mean, prior_variance, view_scale, extra_noise, view_bias=view_bias)
    manual = marginal_log_likelihood(centered, prior_mean, prior_variance, view_scale, extra_noise)
    np.testing.assert_allclose(with_bias, manual)


def test_default_fit_view_bias_false_reproduces_explicit_false() -> None:
    data = make_data([0.2, 0.4, -0.3, -0.1, 1.0, 1.4])
    default = anchored_calibrated_eb(data, max_iter=20)
    explicit = anchored_calibrated_eb(data, max_iter=20, fit_view_bias=False)
    np.testing.assert_allclose(default.posterior["posterior_mean"], explicit.posterior["posterior_mean"])
    np.testing.assert_allclose(default.posterior["posterior_var"], explicit.posterior["posterior_var"])
    np.testing.assert_allclose(default.diagnostics["view_bias"], np.zeros(data.n_views))
    assert default.diagnostics["fit_view_bias"] is False


def test_center_estimates_by_view_returns_offsets_and_centered_data() -> None:
    data = make_data([1.0, 3.0, -1.0, 5.0, 2.0, 7.0])
    centered, offsets = center_estimates_by_view(data)
    assert list(offsets.columns) == ["view_id", "view_idx", "view_bias"]
    for view in range(data.n_views):
        rows = centered.view_idx == view
        np.testing.assert_allclose(np.mean(centered.estimate[rows]), 0.0, atol=1e-12)


def test_graph_centered_moments_invariant_to_constant_per_view() -> None:
    item_idx = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    view_idx = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    estimate = np.array([1.0, 2.0, 3.0, 7.0, 5.0, 8.0, 9.0, 12.0])
    shifted = estimate + np.array([10.0, -4.0])[view_idx]
    graph = build_coobservation_graph(
        item_idx=item_idx,
        view_idx=view_idx,
        estimate=estimate,
        n_views=2,
        min_shared_items=2,
        center=True,
    )
    shifted_graph = build_coobservation_graph(
        item_idx=item_idx,
        view_idx=view_idx,
        estimate=shifted,
        n_views=2,
        min_shared_items=2,
        center=True,
    )
    np.testing.assert_allclose(graph.edge_moments[0].edge_moment, shifted_graph.edge_moments[0].edge_moment)
