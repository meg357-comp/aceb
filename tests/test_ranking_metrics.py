from __future__ import annotations

import numpy as np
import pandas as pd

from cmveb.benchmarks.synthetic_runner import synthetic_parameter_recovery_metrics
from cmveb.evaluate import evaluate_interval_metrics, evaluate_pairwise_calibration, evaluate_pairwise_scores, ndcg_at_k
from cmveb.ranking import false_best_rate, pairwise_win_probabilities, posterior_rank_entropy


def posterior(means, variances):
    return pd.DataFrame(
        {
            "item_id": [f"i{idx}" for idx in range(len(means))],
            "posterior_mean": means,
            "posterior_var": variances,
        }
    )


def truth(values):
    return pd.DataFrame({"item_id": [f"i{idx}" for idx in range(len(values))], "theta": values})


def test_pairwise_probability_is_half_when_means_equal() -> None:
    pairs = pairwise_win_probabilities(posterior([1.0, 1.0], [0.5, 0.5]))
    np.testing.assert_allclose(pairs["pairwise_win_probability"].iloc[0], 0.5)


def test_pairwise_probability_increases_with_mean_gap() -> None:
    small = pairwise_win_probabilities(posterior([0.1, 0.0], [1.0, 1.0]))["pairwise_win_probability"].iloc[0]
    large = pairwise_win_probabilities(posterior([2.0, 0.0], [1.0, 1.0]))["pairwise_win_probability"].iloc[0]
    assert large > small > 0.5


def test_pairwise_calibration_returns_valid_bins() -> None:
    curve, ece = evaluate_pairwise_calibration(
        posterior([3.0, 2.0, 1.0, 0.0], [0.2, 0.2, 0.2, 0.2]),
        truth([3.0, 2.0, 1.0, 0.0]),
        bins=5,
    )
    assert set(["bin_left", "bin_right", "mean_pred", "empirical_win_rate", "n_pairs"]).issubset(curve.columns)
    assert len(curve) == 5
    assert ece >= 0.0
    assert curve["n_pairs"].sum() == 6


def test_false_best_detects_confidently_wrong_top_model() -> None:
    post = posterior([10.0, 0.0], [1e-6, 1e-6])
    tr = truth([0.0, 1.0])
    assert false_best_rate(post, tr, threshold=0.95, n_samples=1000, random_state=1) == 1.0


def test_rank_entropy_lower_for_tiny_variances_than_large_variances() -> None:
    means = [3.0, 2.0, 1.0, 0.0]
    low = posterior_rank_entropy(posterior(means, [1e-4] * 4), n_samples=1000, random_state=2)
    high = posterior_rank_entropy(posterior(means, [10.0] * 4), n_samples=1000, random_state=2)
    assert low < high


def test_ndcg_at_k_equals_one_for_perfect_ranking() -> None:
    values = np.array([4.0, 3.0, 2.0, 1.0])
    np.testing.assert_allclose(ndcg_at_k(values, values, 3), 1.0)


def test_loading_rmse_zero_when_estimated_loadings_equal_truth() -> None:
    view_parameters = pd.DataFrame(
        {
            "view_id": ["v0", "v1", "v2"],
            "a_true": [1.0, 0.5, 2.0],
            "tau_true": [0.1, 0.2, 0.3],
            "b_true": [0.0, 0.0, 0.0],
        }
    )
    metrics = synthetic_parameter_recovery_metrics(
        {"view_scale": np.array([1.0, 0.5, 2.0]), "extra_noise": np.array([0.1, 0.2, 0.3])},
        true_view_parameters=view_parameters,
    )
    np.testing.assert_allclose(metrics["loading_rmse"], 0.0)
    np.testing.assert_allclose(metrics["tau_rmse"], 0.0)


def test_methods_without_loading_estimates_get_nan_parameter_metrics() -> None:
    view_parameters = pd.DataFrame({"view_id": ["v0", "v1"], "a_true": [1.0, 2.0], "tau_true": [0.1, 0.2]})
    metrics = synthetic_parameter_recovery_metrics({}, true_view_parameters=view_parameters)
    assert np.isnan(metrics["loading_rmse"])
    assert np.isnan(metrics["tau_rmse"])


def test_close_pair_ece_uses_fewer_or_equal_pairs_than_all_pairs() -> None:
    post = posterior([3.0, 2.7, 1.0, 0.9, -1.0], [0.2] * 5)
    tr = truth([3.0, 2.8, 1.2, 1.1, -1.0])
    all_curve, _ = evaluate_pairwise_calibration(post, tr, bins=5, pair_subset="all_pairs")
    close_curve, _ = evaluate_pairwise_calibration(post, tr, bins=5, pair_subset="close_pairs", close_gap_quantile=0.3)
    assert close_curve["n_pairs"].sum() <= all_curve["n_pairs"].sum()


def test_pairwise_log_score_is_finite_due_to_clipping() -> None:
    metrics = evaluate_pairwise_scores(
        posterior([100.0, -100.0], [1e-9, 1e-9]),
        truth([-1.0, 1.0]),
        epsilon=1e-6,
    )
    assert np.isfinite(metrics["pairwise_log_score"])


def test_interval_score_penalizes_uncovered_intervals() -> None:
    covered = evaluate_interval_metrics(posterior([0.0], [1.0]), truth([0.0]), level=0.90)["coverage_width_score"]
    uncovered = evaluate_interval_metrics(posterior([0.0], [1.0]), truth([10.0]), level=0.90)["coverage_width_score"]
    assert uncovered > covered
