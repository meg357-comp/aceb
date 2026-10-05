from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from cmveb.evaluate import (
    evaluate_decision_metrics,
    evaluate_decision_metrics_multi,
    evaluate_interval_metrics,
    evaluate_pairwise_calibration,
    evaluate_pairwise_scores,
    evaluate_point_metrics,
    evaluate_ranking_metrics,
    evaluate_stratified_coverage,
)
from cmveb.indexing import build_indexed_data
from cmveb.mash_baseline import mashr_multivariate_eb
from cmveb.models.baseline import (
    adaptive_shrinkage_like,
    anchored_calibrated_eb,
    bootstrap_aceb_variance_correction,
    bootstrap_inverse_variance_all_views,
    bootstrap_reference_view,
    feature_normal_eb_no_view_scale,
    inverse_variance_all_views,
    joint_eb_full,
    normal_eb_no_features,
    posthoc_inverse_variance_all_views,
    raw_single_reference,
)
from cmveb.posterior import softplus_numpy
from cmveb.preprocess import prepare_data
from cmveb.schemas import IndexedData


SCENARIOS = [
    "correct_specification",
    "scale_confounding",
    "underestimated_standard_errors",
    "heavy_tailed_theta",
    "missing_views",
    "noisy_features",
]

SIZES = {
    "small": {"n_items": 500, "n_views": 5, "n_features": 10},
    "medium": {"n_items": 5000, "n_views": 8, "n_features": 20},
}

METHODS: dict[str, Callable[..., object]] = {
    "raw_single_reference": raw_single_reference,
    "inverse_variance_all_views": inverse_variance_all_views,
    "posthoc_inverse_variance_all_views": posthoc_inverse_variance_all_views,
    "normal_eb_no_features": normal_eb_no_features,
    "feature_normal_eb_no_view_scale": feature_normal_eb_no_view_scale,
    "joint_eb_full": joint_eb_full,
    "anchored_calibrated_eb": anchored_calibrated_eb,
    "adaptive_shrinkage_like": adaptive_shrinkage_like,
    "mashr_multivariate_eb": mashr_multivariate_eb,
}

BOOTSTRAP_METHODS: dict[str, Callable[..., object]] = {
    "bootstrap_inverse_variance_all_views": bootstrap_inverse_variance_all_views,
    "bootstrap_reference_view": bootstrap_reference_view,
    "bootstrap_aceb_variance_correction": bootstrap_aceb_variance_correction,
}


@dataclass(slots=True)
class SyntheticDataset:
    data: IndexedData
    truth: pd.DataFrame
    true_view_scale: np.ndarray
    true_extra_noise: np.ndarray


def simulate_scenario(
    *,
    scenario: str,
    n_items: int,
    n_views: int,
    n_features: int,
    seed: int,
) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    item_ids = np.array([f"item_{idx:05d}" for idx in range(n_items)])
    view_ids = np.array([f"view_{idx:02d}" for idx in range(n_views)])

    latent_features = rng.normal(size=(n_items, n_features))
    w_mu = rng.normal(0.0, 0.25, size=n_features)
    w_v = rng.normal(0.0, 0.12, size=n_features)
    w_v[0] = 0.0
    prior_mean = latent_features @ w_mu
    prior_variance = softplus_numpy(latent_features @ w_v) + 1e-6

    if scenario == "heavy_tailed_theta":
        theta = prior_mean + np.sqrt(prior_variance) * t.rvs(df=3, size=n_items, random_state=rng) / math.sqrt(3.0)
    else:
        theta = rng.normal(prior_mean, np.sqrt(prior_variance))

    true_view_scale = np.ones(n_views, dtype=np.float64)
    true_extra_noise = np.linspace(0.02, 0.08, n_views)
    if scenario in {"correct_specification", "heavy_tailed_theta", "missing_views", "noisy_features"}:
        true_view_scale[1:] = np.linspace(0.55, 1.35, n_views - 1)
    elif scenario == "scale_confounding":
        template = np.array([0.35, 0.7, 1.8, 2.4, 1.4, 0.5, 2.1], dtype=np.float64)
        true_view_scale[1:] = template[: n_views - 1]
        true_extra_noise[1:] = np.linspace(0.04, 0.16, n_views - 1)
    elif scenario == "underestimated_standard_errors":
        true_view_scale[1:] = np.linspace(0.65, 1.45, n_views - 1)
        true_extra_noise[:] = np.linspace(0.18, 0.35, n_views)

    observed_features = latent_features.copy()
    if scenario == "noisy_features":
        observed_features = latent_features + rng.normal(0.0, 1.25, size=latent_features.shape)

    item_features = pd.DataFrame(observed_features, columns=[f"x{idx:02d}" for idx in range(n_features)])
    item_features.insert(0, "item_id", item_ids)
    view_metadata = pd.DataFrame({"view_id": view_ids, "is_reference_view": [idx == 0 for idx in range(n_views)]})

    rows = []
    for item_idx, item_id in enumerate(item_ids):
        for view_idx, view_id in enumerate(view_ids):
            if scenario == "missing_views" and view_idx != 0 and rng.random() < 0.35:
                continue
            reported_standard_error = float(rng.uniform(0.12, 0.45))
            true_standard_error = reported_standard_error
            if scenario == "underestimated_standard_errors":
                true_standard_error = reported_standard_error * 1.8
            variance = true_standard_error * true_standard_error + true_extra_noise[view_idx] ** 2
            estimate = rng.normal(true_view_scale[view_idx] * theta[item_idx], math.sqrt(variance))
            rows.append(
                {
                    "item_id": item_id,
                    "view_id": view_id,
                    "estimate": float(estimate),
                    "standard_error": reported_standard_error,
                }
            )
    observations = pd.DataFrame(rows)
    data = build_indexed_data(prepare_data(observations, item_features, view_metadata))
    truth = pd.DataFrame(
        {
            "item_id": item_ids,
            "theta": theta,
            "prior_mean": prior_mean,
            "prior_variance": prior_variance,
        }
    )
    return SyntheticDataset(data, truth, true_view_scale, true_extra_noise)


def _predictive_log_likelihood(data: IndexedData, posterior: pd.DataFrame, diagnostics: dict[str, object]) -> float:
    mean = posterior["posterior_mean"].to_numpy(dtype=np.float64)
    variance = posterior["posterior_var"].to_numpy(dtype=np.float64)
    view_scale = np.asarray(diagnostics.get("view_scale", np.ones(data.n_views)), dtype=np.float64)
    if view_scale.shape != (data.n_views,):
        view_scale = np.ones(data.n_views, dtype=np.float64)
    extra_noise = np.asarray(diagnostics.get("extra_noise", np.zeros(data.n_views)), dtype=np.float64)
    if extra_noise.shape != (data.n_views,):
        extra_noise = np.zeros(data.n_views, dtype=np.float64)
    view_bias = np.asarray(diagnostics.get("view_bias", np.zeros(data.n_views)), dtype=np.float64)
    if view_bias.shape != (data.n_views,):
        view_bias = np.zeros(data.n_views, dtype=np.float64)
    row_mean = view_bias[data.view_idx] + view_scale[data.view_idx] * mean[data.item_idx]
    row_var = (
        np.square(data.standard_error)
        + np.square(extra_noise[data.view_idx])
        + np.square(view_scale[data.view_idx]) * variance[data.item_idx]
    )
    return float(np.mean(norm.logpdf(data.estimate, row_mean, np.sqrt(np.maximum(row_var, 1e-18)))))


def synthetic_parameter_recovery_metrics(
    diagnostics: dict[str, object],
    *,
    true_view_parameters: pd.DataFrame | None = None,
    true_view_scale: np.ndarray | None = None,
    true_extra_noise: np.ndarray | None = None,
) -> dict[str, float]:
    if true_view_parameters is not None:
        true_a = true_view_parameters["a_true"].to_numpy(dtype=np.float64)
        true_tau = true_view_parameters["tau_true"].to_numpy(dtype=np.float64)
        true_bias = (
            true_view_parameters["b_true"].to_numpy(dtype=np.float64)
            if "b_true" in true_view_parameters
            else np.zeros_like(true_a)
        )
    elif true_view_scale is not None and true_extra_noise is not None:
        true_a = np.asarray(true_view_scale, dtype=np.float64)
        true_tau = np.asarray(true_extra_noise, dtype=np.float64)
        true_bias = np.zeros_like(true_a)
    else:
        true_a = true_tau = true_bias = None

    def missing() -> dict[str, float]:
        return {
            "loading_rmse": np.nan,
            "log_loading_rmse": np.nan,
            "loading_mae": np.nan,
            "loading_corr": np.nan,
            "tau_rmse": np.nan,
            "tau_mae": np.nan,
            "bias_rmse": np.nan,
            "prior_scale_error": np.nan,
        }

    if true_a is None or true_tau is None:
        return missing()
    estimated_a = diagnostics.get("view_scale")
    estimated_tau = diagnostics.get("extra_noise")
    estimated_bias = diagnostics.get("view_bias")
    metrics = missing()
    if estimated_a is not None:
        a = np.asarray(estimated_a, dtype=np.float64)
        if a.shape == true_a.shape:
            a = a.copy()
            if np.isfinite(a[0]) and abs(a[0]) > 1e-12:
                a = a / a[0]
            a[0] = 1.0
            diff = a - true_a
            metrics["loading_rmse"] = float(np.sqrt(np.mean(diff * diff)))
            metrics["log_loading_rmse"] = float(
                np.sqrt(np.mean(np.square(np.log(np.maximum(a, 1e-12)) - np.log(np.maximum(true_a, 1e-12)))))
            )
            metrics["loading_mae"] = float(np.mean(np.abs(diff)))
            metrics["loading_corr"] = (
                float(np.corrcoef(true_a, a)[0, 1]) if np.std(true_a) > 0.0 and np.std(a) > 0.0 else np.nan
            )
    if estimated_tau is not None:
        tau = np.asarray(estimated_tau, dtype=np.float64)
        if tau.shape == true_tau.shape:
            diff = tau - true_tau
            metrics["tau_rmse"] = float(np.sqrt(np.mean(diff * diff)))
            metrics["tau_mae"] = float(np.mean(np.abs(diff)))
    if estimated_bias is not None:
        bias = np.asarray(estimated_bias, dtype=np.float64)
        if bias.shape == true_bias.shape:
            metrics["bias_rmse"] = float(np.sqrt(np.mean(np.square(bias - true_bias))))
    prior_scale = diagnostics.get("prior_scale", diagnostics.get("sigma_theta"))
    if prior_scale is not None and "prior_scale_error" in metrics:
        try:
            metrics["prior_scale_error"] = float(prior_scale) - float(np.nanstd(true_a))
        except (TypeError, ValueError):
            pass
    return metrics


def compute_metrics(
    *,
    data: IndexedData,
    truth: pd.DataFrame,
    posterior: pd.DataFrame,
    diagnostics: dict[str, object],
    size: str,
    scenario: str,
    method_name: str,
    runtime_seconds: float,
    true_view_parameters: pd.DataFrame | None = None,
    true_view_scale: np.ndarray | None = None,
    true_extra_noise: np.ndarray | None = None,
) -> dict[str, float | str | bool | int]:
    theta = truth["theta"].to_numpy(dtype=np.float64)
    mean = posterior["posterior_mean"].to_numpy(dtype=np.float64)
    var = np.maximum(posterior["posterior_var"].to_numpy(dtype=np.float64), 1e-18)
    sd = np.sqrt(var)
    error = mean - theta
    z = error / sd
    top_k = max(1, int(0.1 * theta.size))
    truth_top = set(np.argsort(np.abs(theta))[-top_k:].tolist())
    estimate_top = set(np.argsort(np.abs(mean))[-top_k:].tolist())
    corr = float(np.corrcoef(theta, mean)[0, 1]) if np.std(mean) > 0.0 and np.std(theta) > 0.0 else np.nan
    mse = float(np.mean(error * error))
    point = evaluate_point_metrics(posterior, truth)
    interval = evaluate_interval_metrics(posterior, truth, level=0.90)
    ranking = evaluate_ranking_metrics(
        posterior,
        truth,
        k_values=(1, 5, 10, max(1, int(0.1 * theta.size))),
        n_samples=1000,
        random_state=17,
    )
    decision = evaluate_decision_metrics(posterior, truth, threshold=0.95, n_samples=1000, random_state=23)
    decision_multi = evaluate_decision_metrics_multi(posterior, truth, n_samples=1000, random_state=23)
    pairwise_scores = evaluate_pairwise_scores(posterior, truth, max_pairs=10000, random_state=29)
    parameter_recovery = synthetic_parameter_recovery_metrics(
        diagnostics,
        true_view_parameters=true_view_parameters,
        true_view_scale=true_view_scale,
        true_extra_noise=true_extra_noise,
    )
    graph = diagnostics.get("graph_diagnostic", {})
    row = {
        "size": size,
        "scenario": scenario,
        "method_name": method_name,
        "n_items": int(theta.size),
        "n_rows": int(data.n_rows),
        "rmse": float(math.sqrt(mse)),
        "correlation": corr,
        "sign_accuracy": float(np.mean(np.sign(theta) == np.sign(mean))),
        "coverage_90": float(
            np.mean((theta >= mean - 1.6448536269514722 * sd) & (theta <= mean + 1.6448536269514722 * sd))
        ),
        "posterior_z_mean": float(np.mean(z)),
        "posterior_z_sd": float(np.std(z)),
        "variance_ratio": float(np.mean(var) / max(mse, 1e-18)),
        "top_k_overlap": float(len(truth_top & estimate_top) / top_k),
        "predictive_log_likelihood": _predictive_log_likelihood(data, posterior, diagnostics),
        "runtime_seconds": float(runtime_seconds),
        "optimizer_success": bool(diagnostics.get("success", True)),
    }
    row.update(point)
    row.update(interval)
    row.update(ranking)
    row.update(decision)
    row.update(decision_multi)
    row.update(pairwise_scores)
    row.update(parameter_recovery)
    if isinstance(graph, dict):
        row.update(
            {
                "graph_n_views": graph.get("n_views", data.n_views),
                "graph_n_edges": graph.get("n_edges", np.nan),
                "graph_n_components": graph.get("n_components", np.nan),
                "component_count": graph.get("component_count", graph.get("n_components", np.nan)),
                "graph_anchor_component_is_non_bipartite": graph.get("anchor_component_is_non_bipartite", False),
                "anchor_component_bipartite": graph.get("anchor_component_bipartite", np.nan),
                "anchor_component_has_odd_cycle": graph.get("anchor_component_has_odd_cycle", np.nan),
                "graph_rank_deficiency_delta": graph.get("rank_deficiency_delta", np.nan),
                "delta_G": graph.get("delta_G", graph.get("rank_deficiency_delta", np.nan)),
                "graph_signless_incidence_rank": graph.get("signless_incidence_rank", np.nan),
                "graph_design_rank": graph.get("graph_design_rank", graph.get("signless_incidence_rank", np.nan)),
                "graph_design_num_columns": graph.get("graph_design_num_columns", np.nan),
                "smallest_full_singular_value": graph.get(
                    "smallest_full_singular_value",
                    graph.get("smallest_singular_value", np.nan),
                ),
                "smallest_singular_value": graph.get("smallest_full_singular_value", graph.get("smallest_singular_value", np.nan)),
                "largest_singular_value": graph.get("largest_singular_value", np.nan),
                "condition_number": graph.get("condition_number", np.nan),
                "normalized_smallest_full_singular_value": graph.get("normalized_smallest_full_singular_value", np.nan),
                "normalized_condition_number": graph.get("normalized_condition_number", np.nan),
            }
        )
    return row


def skipped_metric_row(
    *,
    data: IndexedData,
    truth: pd.DataFrame,
    diagnostics: dict[str, object],
    size: str,
    scenario: str,
    method_name: str,
    runtime_seconds: float,
) -> dict[str, float | str | bool | int]:
    return {
        "size": size,
        "scenario": scenario,
        "method_name": method_name,
        "n_items": int(truth.shape[0]),
        "n_rows": int(data.n_rows),
        "rmse": np.nan,
        "mae": np.nan,
        "bias": np.nan,
        "correlation": np.nan,
        "spearman_correlation": np.nan,
        "kendall_tau": np.nan,
        "sign_accuracy": np.nan,
        "coverage_90": np.nan,
        "posterior_z_mean": np.nan,
        "posterior_z_sd": np.nan,
        "z_score_sd_abs_error": np.nan,
        "variance_ratio": np.nan,
        "top_k_overlap": np.nan,
        "posterior_rank_entropy": np.nan,
        "false_best_rate": np.nan,
        "false_best_rate_075": np.nan,
        "false_best_rate_090": np.nan,
        "false_best_rate_095": np.nan,
        "abstention_rate_075": np.nan,
        "abstention_rate_090": np.nan,
        "abstention_rate_095": np.nan,
        "selective_accuracy_075": np.nan,
        "selective_accuracy_090": np.nan,
        "selective_accuracy_095": np.nan,
        "probability_select_true_best": np.nan,
        "true_best_in_credible_top_1": np.nan,
        "true_best_in_credible_top_3": np.nan,
        "true_best_in_credible_top_5": np.nan,
        "average_interval_width_90": np.nan,
        "median_interval_width_90": np.nan,
        "coverage_width_score": np.nan,
        "loading_rmse": np.nan,
        "log_loading_rmse": np.nan,
        "loading_mae": np.nan,
        "loading_corr": np.nan,
        "tau_rmse": np.nan,
        "tau_mae": np.nan,
        "bias_rmse": np.nan,
        "prior_scale_error": np.nan,
        "pairwise_brier_score": np.nan,
        "pairwise_log_score": np.nan,
        "all_pair_ece": np.nan,
        "close_pair_ece": np.nan,
        "close_pair_brier_score": np.nan,
        "close_pair_log_score": np.nan,
        "top_decile_pair_ece": np.nan,
        "predictive_log_likelihood": np.nan,
        "runtime_seconds": float(runtime_seconds),
        "optimizer_success": False,
        "skipped": bool(diagnostics.get("skipped", False)),
        "failed": bool(diagnostics.get("failed", False)),
        "skip_reason": str(diagnostics.get("skip_reason", "")),
        "failure_message": str(diagnostics.get("failure_message", "")),
        "mash_available": bool(diagnostics.get("mash_available", False)),
    }


def _item_view_counts_and_components(data: IndexedData, diagnostics: dict[str, object]) -> tuple[np.ndarray, np.ndarray | None]:
    view_count = np.bincount(data.item_idx, minlength=data.n_items)
    graph = diagnostics.get("graph_diagnostic", {})
    if not isinstance(graph, dict) or "component_id_by_view" not in graph:
        return view_count, None
    component_by_view = np.asarray(graph["component_id_by_view"])
    component = np.full(data.n_items, -1, dtype=np.int64)
    for item in range(data.n_items):
        rows = data.item_idx == item
        if np.any(rows):
            components = component_by_view[data.view_idx[rows]]
            values, counts = np.unique(components, return_counts=True)
            component[item] = int(values[np.argmax(counts)])
    return view_count, component


def run_methods(
    data: IndexedData,
    *,
    max_iter: int,
    truth: pd.DataFrame | None = None,
    include_bootstraps: bool = False,
    bootstrap_B: int = 5,
    size: str | None = None,
    scenario: str | None = None,
) -> list[tuple[str, pd.DataFrame, dict[str, object], float]]:
    outputs = []
    methods = dict(METHODS)
    if include_bootstraps:
        methods.update(BOOTSTRAP_METHODS)
    iter_methods = {
        "normal_eb_no_features",
        "feature_normal_eb_no_view_scale",
        "joint_eb_full",
        "anchored_calibrated_eb",
        "adaptive_shrinkage_like",
    }
    for method_name, method in methods.items():
        prefix = f" size={size} scenario={scenario}" if size is not None and scenario is not None else ""
        print(f"[synthetic] start{prefix} method={method_name}", flush=True)
        start = time.perf_counter()
        if method_name == "posthoc_inverse_variance_all_views":
            output = method(data, truth=truth)
        elif method_name == "bootstrap_aceb_variance_correction":
            output = method(data, B=bootstrap_B, max_iter=max(20, max_iter // 2))
        elif method_name in BOOTSTRAP_METHODS:
            output = method(data, B=bootstrap_B)
        elif method_name in iter_methods:
            output = method(data, max_iter=max_iter)
        else:
            output = method(data)
        runtime_seconds = time.perf_counter() - start
        print(f"[synthetic] finished{prefix} method={method_name} runtime={runtime_seconds:.2f}s", flush=True)
        outputs.append((method_name, output.posterior, output.diagnostics, runtime_seconds))
    return outputs


def _svg_bar_plot(frame: pd.DataFrame, value: str, title: str, path: Path) -> None:
    rows = frame.sort_values(["scenario", "method_name"])
    labels = [f"{row.scenario[:12]}|{row.method_name[:12]}" for row in rows.itertuples()]
    values = rows[value].to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    max_value = float(np.max(finite)) if finite.size else 1.0
    max_value = max(max_value, 1e-9)
    width = max(900, 18 * len(values) + 180)
    height = 420
    left = 120
    bottom = 90
    plot_height = height - bottom - 50
    bar_width = max(4, int((width - left - 40) / max(len(values), 1)) - 2)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="{left}" y="28" font-size="18" font-family="sans-serif">{title}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-20}" y2="{height-bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="45" x2="{left}" y2="{height-bottom}" stroke="black"/>',
    ]
    for idx, val in enumerate(values):
        bar_height = 0.0 if not np.isfinite(val) else plot_height * max(val, 0.0) / max_value
        x = left + idx * (bar_width + 2)
        y = height - bottom - bar_height
        parts.append(f'<rect x="{x}" y="{y:.2f}" width="{bar_width}" height="{bar_height:.2f}" fill="#4C78A8"/>')
        if idx % max(1, len(values) // 24) == 0:
            parts.append(
                f'<text x="{x}" y="{height-bottom+14}" font-size="9" font-family="sans-serif" transform="rotate(55 {x},{height-bottom+14})">{labels[idx]}</text>'
            )
    parts.append(f'<text x="12" y="55" font-size="11" font-family="sans-serif">max={max_value:.3g}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_plots(metrics: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    first_size = metrics["size"].iloc[0]
    frame = metrics.loc[metrics["size"] == first_size].copy()
    _svg_bar_plot(frame, "rmse", "RMSE by scenario and method", plots_dir / "rmse.svg")
    _svg_bar_plot(frame, "coverage_90", "90% coverage by scenario and method", plots_dir / "coverage_90.svg")
    _svg_bar_plot(frame, "posterior_z_sd", "Posterior z-score SD by scenario and method", plots_dir / "posterior_z_sd.svg")


def _best_method_table(metrics: pd.DataFrame, metric: str, ascending: bool) -> pd.DataFrame:
    valid = metrics.loc[np.isfinite(metrics[metric].to_numpy(dtype=np.float64))].copy()
    if valid.empty:
        return metrics.iloc[0:0].loc[:, ["size", "scenario", "method_name", metric]].copy()
    grouped = valid.groupby(["size", "scenario"])[metric]
    idx = grouped.idxmin() if ascending else grouped.idxmax()
    return valid.loc[idx, ["size", "scenario", "method_name", metric]].sort_values(["size", "scenario"])


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4g}")
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_summary(metrics: pd.DataFrame, path: Path, *, sizes_run: list[str], scenarios_run: list[str]) -> None:
    def metric_value(scenario: str, method: str, metric: str) -> float:
        subset = metrics.loc[(metrics["scenario"] == scenario) & (metrics["method_name"] == method), metric]
        return float(subset.mean()) if not subset.empty else float("nan")

    scale_joint_z = metric_value("scale_confounding", "joint_eb_full", "posterior_z_sd")
    scale_anchored_z = metric_value("scale_confounding", "anchored_calibrated_eb", "posterior_z_sd")
    scale_joint_cov = metric_value("scale_confounding", "joint_eb_full", "coverage_90")
    scale_anchored_cov = metric_value("scale_confounding", "anchored_calibrated_eb", "coverage_90")
    anchored_cov = metrics.loc[metrics["method_name"] == "anchored_calibrated_eb", "coverage_90"].mean()
    naive_cov = metrics.loc[metrics["method_name"] == "inverse_variance_all_views", "coverage_90"].mean()
    anchored_rmse_rank = (
        metrics.assign(rmse_rank=metrics.groupby(["size", "scenario"])["rmse"].rank(method="min"))
        .loc[lambda frame: frame["method_name"] == "anchored_calibrated_eb", "rmse_rank"]
        .mean()
    )
    failures = (
        metrics.loc[metrics["method_name"] == "anchored_calibrated_eb"]
        .assign(coverage_gap=lambda frame: np.abs(frame["coverage_90"] - 0.90))
        .sort_values(["coverage_gap", "rmse"], ascending=False)
        .head(3)
    )
    best_rmse = _best_method_table(metrics, "rmse", ascending=True)
    best_coverage = metrics.assign(coverage_gap=np.abs(metrics["coverage_90"] - 0.90))
    best_coverage = _best_method_table(best_coverage, "coverage_gap", ascending=True)

    lines = [
        "# Synthetic Benchmark Summary",
        "",
        f"Sizes run: {', '.join(sizes_run)}",
        f"Scenarios run: {', '.join(scenarios_run)}",
        "",
        "## Questions",
        "",
        "1. Does joint EB become miscalibrated under scale/noise confounding?",
        "",
        (
            f"Under `scale_confounding`, joint EB had mean 90% coverage {scale_joint_cov:.3f} "
            f"and z-score SD {scale_joint_z:.3f}; anchored calibrated EB had coverage "
            f"{scale_anchored_cov:.3f} and z-score SD {scale_anchored_z:.3f}."
        ),
        "",
        "2. Does anchored calibrated EB improve coverage?",
        "",
        (
            f"Across the run, anchored calibrated EB mean 90% coverage was {anchored_cov:.3f}, "
            f"versus {naive_cov:.3f} for naive inverse-variance pooling."
        ),
        "",
        "3. Does anchored calibrated EB retain competitive RMSE?",
        "",
        f"Anchored calibrated EB had average RMSE rank {anchored_rmse_rank:.2f} across scenario-size cells.",
        "",
        "4. Where does the method fail?",
        "",
    ]
    for row in failures.itertuples(index=False):
        lines.append(
            f"- `{row.size}/{row.scenario}`: coverage={row.coverage_90:.3f}, rmse={row.rmse:.3f}, "
            f"z_sd={row.posterior_z_sd:.3f}, variance_ratio={row.variance_ratio:.3f}"
        )
    lines.extend(
        [
            "",
            "## Best RMSE",
            "",
            _markdown_table(best_rmse),
            "",
            "## Closest Coverage To 90%",
            "",
            _markdown_table(best_coverage),
            "",
            "## Mean Metrics By Method",
            "",
            _markdown_table(
                metrics.groupby("method_name")[
                    ["rmse", "correlation", "coverage_90", "posterior_z_sd", "variance_ratio", "predictive_log_likelihood"]
                ]
                .mean()
                .reset_index()
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_synthetic_benchmark(
    *,
    output_dir: Path,
    sizes: list[str],
    scenarios: list[str],
    seed: int = 1,
    max_iter: int = 80,
    include_bootstraps: bool = False,
    bootstrap_B: int = 5,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    pairwise_rows = []
    stratified_rows = []
    for size_idx, size in enumerate(sizes):
        if size not in SIZES:
            raise ValueError(f"Unknown size: {size}")
        params = SIZES[size]
        for scenario_idx, scenario in enumerate(scenarios):
            if scenario not in SCENARIOS:
                raise ValueError(f"Unknown scenario: {scenario}")
            scenario_seed = seed + 1000 * size_idx + 17 * scenario_idx
            print(f"[synthetic] simulate size={size} scenario={scenario} seed={scenario_seed}", flush=True)
            dataset = simulate_scenario(scenario=scenario, seed=scenario_seed, **params)
            for method_name, posterior, diagnostics, runtime_seconds in run_methods(
                dataset.data,
                max_iter=max_iter,
                truth=dataset.truth,
                include_bootstraps=include_bootstraps,
                bootstrap_B=bootstrap_B,
                size=size,
                scenario=scenario,
            ):
                if diagnostics.get("skipped", False):
                    metrics_rows.append(
                        skipped_metric_row(
                            data=dataset.data,
                            truth=dataset.truth,
                            diagnostics=diagnostics,
                            size=size,
                            scenario=scenario,
                            method_name=method_name,
                            runtime_seconds=runtime_seconds,
                        )
                    )
                    continue
                pairwise_curve, pairwise_ece = evaluate_pairwise_calibration(
                    posterior,
                    dataset.truth,
                    bins=10,
                    max_pairs=20000,
                    random_state=scenario_seed,
                    pair_subset="all_pairs",
                )
                pairwise_curve.insert(0, "method_name", method_name)
                pairwise_curve.insert(0, "scenario", scenario)
                pairwise_curve.insert(0, "size", size)
                pairwise_curve["pairwise_ece"] = pairwise_ece
                pairwise_rows.append(pairwise_curve)
                close_curve, close_ece = evaluate_pairwise_calibration(
                    posterior,
                    dataset.truth,
                    bins=10,
                    max_pairs=20000,
                    random_state=scenario_seed,
                    pair_subset="close_pairs",
                )
                close_curve.insert(0, "method_name", method_name)
                close_curve.insert(0, "scenario", scenario)
                close_curve.insert(0, "size", size)
                close_curve["pairwise_ece"] = close_ece
                pairwise_rows.append(close_curve)
                view_count, component = _item_view_counts_and_components(dataset.data, diagnostics)
                stratified = evaluate_stratified_coverage(
                    posterior,
                    dataset.truth,
                    view_count=view_count,
                    graph_component=component,
                    level=0.90,
                )
                stratified.insert(0, "method_name", method_name)
                stratified.insert(0, "scenario", scenario)
                stratified.insert(0, "size", size)
                stratified_rows.append(stratified)
                metric_row = compute_metrics(
                    data=dataset.data,
                    truth=dataset.truth,
                    posterior=posterior,
                    diagnostics=diagnostics,
                    size=size,
                    scenario=scenario,
                    method_name=method_name,
                    runtime_seconds=runtime_seconds,
                    true_view_scale=dataset.true_view_scale,
                    true_extra_noise=dataset.true_extra_noise,
                )
                metric_row["pairwise_ece"] = pairwise_ece
                metrics_rows.append(
                    metric_row
                )
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    if pairwise_rows:
        pd.concat(pairwise_rows, ignore_index=True).to_csv(output_dir / "pairwise_calibration.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "pairwise_calibration.csv", index=False)
    if stratified_rows:
        pd.concat(stratified_rows, ignore_index=True).to_csv(output_dir / "stratified_coverage.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "stratified_coverage.csv", index=False)
    write_plots(metrics, output_dir / "plots")
    write_summary(metrics, output_dir / "summary.md", sizes_run=sizes, scenarios_run=scenarios)
    return metrics


def run_synthetic(seed: int = 1):
    dataset = simulate_scenario(scenario="correct_specification", seed=seed, **SIZES["small"])
    return anchored_calibrated_eb(dataset.data), dataset.truth


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run synthetic calibrated multi-view EB benchmark.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/synthetic"))
    parser.add_argument("--sizes", nargs="+", default=["small"], choices=sorted(SIZES))
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS, choices=SCENARIOS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--include-bootstraps", action="store_true")
    parser.add_argument("--bootstrap-B", type=int, default=5)
    args = parser.parse_args(argv)
    metrics = run_synthetic_benchmark(
        output_dir=args.output_dir,
        sizes=args.sizes,
        scenarios=args.scenarios,
        seed=args.seed,
        max_iter=args.max_iter,
        include_bootstraps=args.include_bootstraps,
        bootstrap_B=args.bootstrap_B,
    )
    print(f"Wrote {len(metrics)} metric rows to {args.output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
