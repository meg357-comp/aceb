from __future__ import annotations

import numpy as np
import pandas as pd

from cmveb.indexing import build_indexed_data
from cmveb.models.baseline import (
    BASELINE_METHODS,
    anchored_calibrated_eb,
    bootstrap_inverse_variance_all_views,
    bootstrap_reference_view,
    inverse_variance_all_views,
    joint_eb_full,
    posthoc_variance_inflation,
)
from cmveb.preprocess import prepare_data


REQUIRED_COLUMNS = {
    "item_id",
    "posterior_mean",
    "posterior_var",
    "posterior_sd",
    "posterior_second_moment",
    "lfsr",
    "method_name",
}


def make_tiny_data():
    observations = pd.DataFrame(
        {
            "item_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "view_id": ["ref", "alt", "ref", "alt", "ref", "alt", "ref", "alt"],
            "estimate": [0.2, 0.3, 1.1, 1.8, -0.7, -1.0, 0.0, 0.1],
            "standard_error": [0.4, 0.5, 0.3, 0.4, 0.4, 0.6, 0.5, 0.5],
        }
    )
    item_features = pd.DataFrame({"item_id": ["a", "b", "c", "d"], "x": [0.0, 1.0, -1.0, 0.5]})
    view_metadata = pd.DataFrame({"view_id": ["ref", "alt"], "is_reference_view": [True, False]})
    return build_indexed_data(prepare_data(observations, item_features, view_metadata))


def test_each_baseline_runs_on_tiny_synthetic_dataset() -> None:
    data = make_tiny_data()
    for method_name, method in BASELINE_METHODS.items():
        output = method(data, max_iter=40) if method_name in {"normal_eb_no_features", "feature_normal_eb_no_view_scale", "joint_eb_full", "anchored_calibrated_eb", "adaptive_shrinkage_like"} else method(data)
        assert REQUIRED_COLUMNS.issubset(output.posterior.columns), method_name
        assert output.posterior.shape[0] == data.n_items
        assert set(output.posterior["method_name"]) == {method_name}


def test_baseline_posterior_variances_are_positive() -> None:
    data = make_tiny_data()
    for method_name, method in BASELINE_METHODS.items():
        output = method(data, max_iter=40) if method_name in {"normal_eb_no_features", "feature_normal_eb_no_view_scale", "joint_eb_full", "anchored_calibrated_eb", "adaptive_shrinkage_like"} else method(data)
        assert np.all(output.posterior["posterior_var"].to_numpy(dtype=np.float64) > 0.0), method_name


def make_miscalibrated_data(seed: int = 3):
    rng = np.random.default_rng(seed)
    n_items = 120
    item_ids = [f"item_{idx:03d}" for idx in range(n_items)]
    theta = rng.normal(0.0, 1.0, size=n_items)
    rows = []
    for item_id, value in zip(item_ids, theta, strict=True):
        rows.append(
            {
                "item_id": item_id,
                "view_id": "ref",
                "estimate": float(value + rng.normal(0.0, 0.45)),
                "standard_error": 0.45,
            }
        )
        rows.append(
            {
                "item_id": item_id,
                "view_id": "scaled",
                "estimate": float(2.0 * value + rng.normal(0.0, 0.35)),
                "standard_error": 0.35,
            }
        )
    observations = pd.DataFrame(rows)
    item_features = pd.DataFrame({"item_id": item_ids})
    view_metadata = pd.DataFrame({"view_id": ["ref", "scaled"], "is_reference_view": [True, False]})
    data = build_indexed_data(prepare_data(observations, item_features, view_metadata))
    return data, theta


def coverage_95(frame: pd.DataFrame, theta: np.ndarray) -> float:
    mean = frame["posterior_mean"].to_numpy(dtype=np.float64)
    sd = frame["posterior_sd"].to_numpy(dtype=np.float64)
    return float(np.mean((theta >= mean - 1.96 * sd) & (theta <= mean + 1.96 * sd)))


def test_calibrated_eb_improves_coverage_in_toy_miscalibrated_case() -> None:
    data, theta = make_miscalibrated_data()
    naive = inverse_variance_all_views(data)
    anchored = anchored_calibrated_eb(data, max_iter=120)
    naive_coverage = coverage_95(naive.posterior, theta)
    anchored_coverage = coverage_95(anchored.posterior, theta)
    assert anchored_coverage > naive_coverage + 0.15
    assert anchored.diagnostics["view_scale"][data.view_ids.tolist().index("scaled")] > 1.2


def test_joint_eb_can_be_compared_to_anchored_eb() -> None:
    data = make_tiny_data()
    joint = joint_eb_full(data, max_iter=50)
    anchored = anchored_calibrated_eb(data, max_iter=50)
    merged = joint.posterior.merge(anchored.posterior, on="item_id", suffixes=("_joint", "_anchored"))
    assert merged.shape[0] == data.n_items
    assert "view_scale" in joint.diagnostics
    assert "view_scale" in anchored.diagnostics


def test_posthoc_inflation_increases_variance_when_calibration_z_sd_exceeds_one() -> None:
    data = make_tiny_data()
    base = inverse_variance_all_views(data)
    truth = pd.DataFrame({"item_id": data.item_ids, "theta": np.full(data.n_items, 10.0)})
    calibrated = posthoc_variance_inflation(
        data,
        inverse_variance_all_views,
        truth=truth,
        calibration_fraction=0.5,
        random_state=1,
        method_name="posthoc_test",
    )
    assert calibrated.diagnostics["posthoc_variance_inflation_c2"] > 1.0
    assert np.all(
        calibrated.posterior["posterior_var"].to_numpy(dtype=np.float64)
        >= base.posterior["posterior_var"].to_numpy(dtype=np.float64)
    )


def test_posthoc_inflation_leaves_variance_unchanged_when_z_sd_is_small() -> None:
    data = make_tiny_data()
    base = inverse_variance_all_views(data)
    truth = base.posterior.loc[:, ["item_id", "posterior_mean"]].rename(columns={"posterior_mean": "theta"})
    calibrated = posthoc_variance_inflation(
        data,
        inverse_variance_all_views,
        truth=truth,
        calibration_fraction=0.5,
        random_state=2,
        method_name="posthoc_test",
    )
    assert calibrated.diagnostics["posthoc_variance_inflation_c2"] == 1.0
    np.testing.assert_allclose(
        calibrated.posterior["posterior_var"].to_numpy(dtype=np.float64),
        base.posterior["posterior_var"].to_numpy(dtype=np.float64),
    )


def test_bootstrap_output_has_positive_variance_and_success_count() -> None:
    data = make_tiny_data()
    output = bootstrap_inverse_variance_all_views(data, B=4, random_state=3)
    assert np.all(output.posterior["posterior_var"].to_numpy(dtype=np.float64) > 0.0)
    assert output.diagnostics["bootstrap_successful_fits"] == 4
    assert output.diagnostics["bootstrap_failures"] == 0


def test_bootstrap_reference_view_runs_on_tiny_data() -> None:
    data = make_tiny_data()
    output = bootstrap_reference_view(data, B=3, random_state=4)
    assert REQUIRED_COLUMNS.issubset(output.posterior.columns)
    assert set(output.posterior["method_name"]) == {"bootstrap_reference_view"}
