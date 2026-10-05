from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from cmveb.indexing import build_indexed_data
from cmveb.models.anchored_eb import fit_non_reference_view_scales
from cmveb.posterior import compute_posterior_from_prior, marginal_log_likelihood, residual_moment_extra_noise
from cmveb.preprocess import prepare_data
from cmveb.schemas import PosteriorState


def make_data(
    estimates: list[float],
    standard_errors: list[float],
    item_ids: list[str],
    view_ids: list[str],
    reference_view: str = "v0",
):
    observations = pd.DataFrame(
        {
            "item_id": item_ids,
            "view_id": view_ids,
            "estimate": estimates,
            "standard_error": standard_errors,
        }
    )
    item_features = pd.DataFrame({"item_id": sorted(set(item_ids)), "x": 1.0})
    ordered_views = sorted(set(view_ids))
    view_metadata = pd.DataFrame(
        {"view_id": ordered_views, "is_reference_view": [view_id == reference_view for view_id in ordered_views]}
    )
    return build_indexed_data(prepare_data(observations, item_features, view_metadata, standardize_features=False))


def test_posterior_formula_against_brute_force() -> None:
    data = make_data(
        estimates=[1.4, 0.8, -0.2],
        standard_errors=[0.5, 0.3, 0.7],
        item_ids=["a", "a", "a"],
        view_ids=["v0", "v1", "v2"],
    )
    prior_mean = np.array([0.2])
    prior_variance = np.array([1.7])
    view_scale = np.array([1.0, 0.6, 1.3])
    extra_noise = np.array([0.1, 0.2, 0.0])
    posterior = compute_posterior_from_prior(data, prior_mean, prior_variance, view_scale, extra_noise)

    sigma_sq = np.array([0.5**2 + 0.1**2, 0.3**2 + 0.2**2, 0.7**2])
    precision = 1.0 / prior_variance[0] + np.sum(view_scale**2 / sigma_sq)
    expected_variance = 1.0 / precision
    expected_mean = expected_variance * (prior_mean[0] / prior_variance[0] + np.sum(view_scale * data.estimate / sigma_sq))
    np.testing.assert_allclose(posterior.mean, [expected_mean], atol=1e-12)
    np.testing.assert_allclose(posterior.variance, [expected_variance], atol=1e-12)


def test_marginal_likelihood_for_one_view() -> None:
    data = make_data(
        estimates=[0.1, 1.2, -0.4],
        standard_errors=[0.2, 0.3, 0.4],
        item_ids=["a", "b", "c"],
        view_ids=["v0", "v0", "v0"],
    )
    prior_mean = np.array([0.0, 0.5, -0.2])
    prior_variance = np.array([1.0, 0.7, 2.0])
    view_scale = np.array([0.8])
    extra_noise = np.array([0.1])

    expected = 0.0
    for idx in range(3):
        marginal_sd = np.sqrt(data.standard_error[idx] ** 2 + extra_noise[0] ** 2 + view_scale[0] ** 2 * prior_variance[idx])
        expected += norm.logpdf(data.estimate[idx], loc=view_scale[0] * prior_mean[idx], scale=marginal_sd)
    actual = marginal_log_likelihood(data, prior_mean, prior_variance, view_scale, extra_noise)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_multiview_posterior_with_known_values() -> None:
    data = make_data(
        estimates=[2.0, 1.0],
        standard_errors=[1.0, 2.0],
        item_ids=["a", "a"],
        view_ids=["v0", "v1"],
    )
    posterior = compute_posterior_from_prior(
        data,
        prior_mean=np.array([0.0]),
        prior_variance=np.array([4.0]),
        view_scale=np.array([1.0, 0.5]),
        extra_noise=np.array([0.0, 0.0]),
    )
    expected_variance = 1.0 / (0.25 + 1.0 + 0.25 / 4.0)
    expected_mean = expected_variance * (2.0 + 0.5 * 1.0 / 4.0)
    np.testing.assert_allclose(posterior.mean, [expected_mean])
    np.testing.assert_allclose(posterior.variance, [expected_variance])


def test_residual_extra_noise_calibration_sanity() -> None:
    standard_error = 0.3
    true_extra_noise = np.array([0.4, 0.7])
    data = make_data(
        estimates=[1.0 + np.sqrt(standard_error**2 + true_extra_noise[0] ** 2), -2.0 + np.sqrt(standard_error**2 + true_extra_noise[0] ** 2), 0.5 + np.sqrt(standard_error**2 + true_extra_noise[1] ** 2)],
        standard_errors=[standard_error, standard_error, standard_error],
        item_ids=["a", "b", "c"],
        view_ids=["v0", "v0", "v1"],
    )
    posterior = PosteriorState(
        mean=np.array([1.0, -2.0, 0.5]),
        variance=np.zeros(3),
        second_moment=np.array([1.0, 4.0, 0.25]),
        prior_mean=np.zeros(3),
        prior_variance=np.ones(3),
    )
    calibrated = residual_moment_extra_noise(data, posterior, np.array([1.0, 1.0]), lower=0.0, upper=2.0)
    np.testing.assert_allclose(calibrated, true_extra_noise, atol=1e-12)


def test_reference_view_scale_fixed_to_one() -> None:
    data = make_data(
        estimates=[0.5, 0.2, 1.5, 0.9],
        standard_errors=[0.2, 0.2, 0.3, 0.3],
        item_ids=["a", "a", "b", "b"],
        view_ids=["v0", "v1", "v0", "v1"],
    )
    fitted, _ = fit_non_reference_view_scales(
        data,
        w_mu=np.array([0.0, 0.0]),
        w_v=np.array([0.0, 0.0]),
        view_scale=np.array([0.25, 0.5]),
        extra_noise=np.array([0.0, 0.0]),
        max_iter=20,
    )
    assert fitted[0] == 1.0
