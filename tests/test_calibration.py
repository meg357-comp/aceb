from __future__ import annotations

import numpy as np
import pandas as pd

from cmveb.calibration import compute_leave_view_posterior_by_row, estimate_tau2_leave_view
from cmveb.indexing import build_indexed_data
from cmveb.models.baseline import anchored_calibrated_eb
from cmveb.posterior import compute_posterior_from_prior
from cmveb.preprocess import prepare_data


def make_data(estimates, standard_errors, item_ids, view_ids):
    observations = pd.DataFrame(
        {
            "item_id": item_ids,
            "view_id": view_ids,
            "estimate": estimates,
            "standard_error": standard_errors,
        }
    )
    items = sorted(set(item_ids))
    views = sorted(set(view_ids))
    item_features = pd.DataFrame({"item_id": items, "x": 1.0})
    view_metadata = pd.DataFrame({"view_id": views, "is_reference_view": [view == views[0] for view in views]})
    return build_indexed_data(prepare_data(observations, item_features, view_metadata, standardize_features=False))


def test_leave_view_posterior_matches_manual_bruteforce() -> None:
    data = make_data(
        estimates=[1.0, 2.0, -1.0],
        standard_errors=[0.5, 0.25, 0.4],
        item_ids=["a", "a", "a"],
        view_ids=["v0", "v1", "v2"],
    )
    prior_mean = np.array([0.3])
    prior_variance = np.array([1.5])
    view_scale = np.array([1.0, 0.5, 1.2])
    extra_noise = np.array([0.1, 0.2, 0.0])
    leave = compute_leave_view_posterior_by_row(data, prior_mean, prior_variance, view_scale, extra_noise)
    sigma2 = data.standard_error**2 + extra_noise[data.view_idx] ** 2
    for row in range(data.n_rows):
        keep = np.arange(data.n_rows) != row
        precision = 1.0 / prior_variance[0] + np.sum(view_scale[data.view_idx[keep]] ** 2 / sigma2[keep])
        natural = prior_mean[0] / prior_variance[0] + np.sum(
            view_scale[data.view_idx[keep]] * data.estimate[keep] / sigma2[keep]
        )
        expected_var = 1.0 / precision
        expected_mean = expected_var * natural
        np.testing.assert_allclose(leave.variance[row], expected_var)
        np.testing.assert_allclose(leave.mean[row], expected_mean)


def test_single_observed_view_leave_view_reduces_to_prior() -> None:
    data = make_data(
        estimates=[1.0],
        standard_errors=[0.5],
        item_ids=["a"],
        view_ids=["v0"],
    )
    leave = compute_leave_view_posterior_by_row(
        data,
        prior_mean=np.array([0.7]),
        prior_variance=np.array([2.3]),
        view_scale=np.array([1.0]),
        extra_noise=np.array([0.2]),
    )
    np.testing.assert_allclose(leave.mean, [0.7])
    np.testing.assert_allclose(leave.variance, [2.3])


def test_leave_view_matches_posterior_after_row_removed() -> None:
    data = make_data(
        estimates=[1.0, 2.0, 4.0],
        standard_errors=[0.5, 0.25, 0.3],
        item_ids=["a", "a", "b"],
        view_ids=["v0", "v1", "v0"],
    )
    prior_mean = np.array([0.2, -0.1])
    prior_variance = np.array([1.1, 0.8])
    view_scale = np.array([1.0, 0.7])
    extra_noise = np.array([0.1, 0.2])
    leave = compute_leave_view_posterior_by_row(data, prior_mean, prior_variance, view_scale, extra_noise)

    row = 1
    reduced = make_data(
        estimates=[data.estimate[0]],
        standard_errors=[data.standard_error[0]],
        item_ids=["a"],
        view_ids=["v0"],
    )
    posterior = compute_posterior_from_prior(
        reduced,
        prior_mean=np.array([prior_mean[0]]),
        prior_variance=np.array([prior_variance[0]]),
        view_scale=np.array([1.0]),
        extra_noise=np.array([extra_noise[0]]),
    )
    np.testing.assert_allclose(leave.mean[row], posterior.mean[0])
    np.testing.assert_allclose(leave.variance[row], posterior.variance[0])


def test_leave_view_tau_calibration_recovers_oracle_tau_approximately() -> None:
    rng = np.random.default_rng(4)
    n_items = 2500
    n_views = 3
    theta = rng.normal(0.0, 1.0, size=n_items)
    true_tau = np.array([0.15, 0.35, 0.55])
    se = 0.1
    rows = []
    for item in range(n_items):
        for view in range(n_views):
            estimate = theta[item] + rng.normal(0.0, np.sqrt(se * se + true_tau[view] ** 2))
            rows.append((f"i{item}", f"v{view}", estimate, se))
    observations = pd.DataFrame(rows, columns=["item_id", "view_id", "estimate", "standard_error"])
    item_features = pd.DataFrame({"item_id": [f"i{item}" for item in range(n_items)], "x": 1.0})
    view_metadata = pd.DataFrame({"view_id": [f"v{view}" for view in range(n_views)], "is_reference_view": [True, False, False]})
    data = build_indexed_data(prepare_data(observations, item_features, view_metadata, standardize_features=False))
    result = estimate_tau2_leave_view(
        data,
        prior_mean=np.zeros(n_items),
        prior_variance=np.ones(n_items),
        view_scale=np.ones(n_views),
        extra_noise=true_tau,
    )
    np.testing.assert_allclose(result.tau_by_view, true_tau, atol=0.08)


def test_default_anchored_calibrated_eb_uses_leave_view() -> None:
    data = make_data(
        estimates=[0.5, 0.8, -0.4, -0.7, 1.1, 1.5],
        standard_errors=[0.3, 0.3, 0.4, 0.4, 0.2, 0.3],
        item_ids=["a", "a", "b", "b", "c", "c"],
        view_ids=["v0", "v1", "v0", "v1", "v0", "v1"],
    )
    output = anchored_calibrated_eb(data, max_iter=20)
    assert output.diagnostics["residual_calibration_method"] == "leave_view"
    assert "tau2_by_view" in output.diagnostics
    assert "tau2_raw_before_positive_part" in output.diagnostics


def test_in_sample_calibration_remains_available_as_ablation() -> None:
    data = make_data(
        estimates=[0.5, 0.8, -0.4, -0.7, 1.1, 1.5],
        standard_errors=[0.3, 0.3, 0.4, 0.4, 0.2, 0.3],
        item_ids=["a", "a", "b", "b", "c", "c"],
        view_ids=["v0", "v1", "v0", "v1", "v0", "v1"],
    )
    output = anchored_calibrated_eb(data, max_iter=20, residual_calibration="in_sample")
    assert output.diagnostics["residual_calibration_method"] == "in_sample"
    assert "tau_by_view" in output.diagnostics
