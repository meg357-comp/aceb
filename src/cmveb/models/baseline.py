from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm

from cmveb.config import ModelConfig
from cmveb.graph_diagnostics import build_coobservation_graph
from cmveb.models.anchored_eb import fit_staged_calibrated_eb
from cmveb.posterior import (
    compute_posterior,
    compute_posterior_from_prior,
    inverse_softplus_numpy,
    marginal_log_likelihood,
    prior_from_features,
)
from cmveb.schemas import IndexedData, PosteriorState


@dataclass(slots=True)
class MethodOutput:
    posterior: pd.DataFrame
    diagnostics: dict[str, object]


BaselineCallable = Callable[[IndexedData], MethodOutput]


def _with_graph_diagnostics(data: IndexedData, diagnostics: dict[str, object]) -> dict[str, object]:
    enriched = dict(diagnostics)
    enriched["graph_diagnostic"] = build_coobservation_graph(data, min_shared_items=1).to_summary_dict()
    return enriched


def posterior_to_frame(
    data: IndexedData,
    posterior: PosteriorState,
    method_name: str,
    *,
    lfsr: np.ndarray | None = None,
) -> pd.DataFrame:
    posterior_var = np.maximum(posterior.variance, 0.0)
    posterior_sd = np.sqrt(posterior_var)
    if lfsr is None:
        z = np.divide(
            posterior.mean,
            posterior_sd,
            out=np.zeros_like(posterior.mean, dtype=np.float64),
            where=posterior_sd > 0.0,
        )
        lfsr = np.minimum(norm.cdf(z), norm.cdf(-z))
    return pd.DataFrame(
        {
            "item_id": data.item_ids,
            "posterior_mean": posterior.mean,
            "posterior_var": posterior_var,
            "posterior_sd": posterior_sd,
            "posterior_second_moment": posterior.mean * posterior.mean + posterior_var,
            "lfsr": lfsr,
            "method_name": method_name,
        }
    )


def _posterior_state(mean: np.ndarray, variance: np.ndarray) -> PosteriorState:
    variance = np.maximum(np.asarray(variance, dtype=np.float64), 1e-18)
    mean = np.asarray(mean, dtype=np.float64)
    return PosteriorState(
        mean=mean,
        variance=variance,
        second_moment=mean * mean + variance,
        prior_mean=np.zeros_like(mean),
        prior_variance=np.full_like(mean, np.inf),
    )


def _subset_items(data: IndexedData, item_indices: np.ndarray) -> IndexedData:
    item_indices = np.asarray(item_indices, dtype=np.int64)
    old_to_new = np.full(data.n_items, -1, dtype=np.int64)
    old_to_new[item_indices] = np.arange(item_indices.size, dtype=np.int64)
    row_mask = old_to_new[data.item_idx] >= 0
    item_idx = old_to_new[data.item_idx[row_mask]]
    order = np.lexsort((data.view_idx[row_mask], item_idx))
    item_idx = np.ascontiguousarray(item_idx[order])
    view_idx = np.ascontiguousarray(data.view_idx[row_mask][order])
    estimate = np.ascontiguousarray(data.estimate[row_mask][order])
    standard_error = np.ascontiguousarray(data.standard_error[row_mask][order])
    counts = np.bincount(item_idx, minlength=item_indices.size)
    item_offsets = np.zeros(item_indices.size + 1, dtype=np.int64)
    item_offsets[1:] = np.cumsum(counts)
    return IndexedData(
        item_ids=np.ascontiguousarray(data.item_ids[item_indices]),
        view_ids=data.view_ids.copy(),
        item_features=np.ascontiguousarray(data.item_features[item_indices], dtype=np.float64),
        feature_columns=list(data.feature_columns),
        scaler=dict(data.scaler),
        item_idx=item_idx,
        view_idx=view_idx,
        estimate=estimate,
        standard_error=standard_error,
        is_reference_view=data.is_reference_view.copy(),
        item_offsets=item_offsets,
        row_order=np.arange(estimate.size, dtype=np.int64),
    )


def _perturb_estimates(data: IndexedData, rng: np.random.Generator) -> IndexedData:
    estimate = rng.normal(data.estimate, data.standard_error)
    return IndexedData(
        item_ids=data.item_ids.copy(),
        view_ids=data.view_ids.copy(),
        item_features=data.item_features.copy(),
        feature_columns=list(data.feature_columns),
        scaler=dict(data.scaler),
        item_idx=data.item_idx.copy(),
        view_idx=data.view_idx.copy(),
        estimate=np.ascontiguousarray(estimate, dtype=np.float64),
        standard_error=data.standard_error.copy(),
        is_reference_view=data.is_reference_view.copy(),
        item_offsets=data.item_offsets.copy(),
        row_order=data.row_order.copy(),
    )


def _truth_by_item(data: IndexedData, truth: pd.DataFrame | None) -> np.ndarray | None:
    if truth is None:
        return None
    column = "theta" if "theta" in truth.columns else "truth" if "truth" in truth.columns else None
    if column is None:
        raise ValueError("truth must contain a theta or truth column.")
    values = truth.set_index("item_id").loc[data.item_ids, column].to_numpy(dtype=np.float64)
    return values


def _heldout_view_inflation(data: IndexedData, posterior: pd.DataFrame) -> float:
    mean = posterior["posterior_mean"].to_numpy(dtype=np.float64)
    variance = np.maximum(posterior["posterior_var"].to_numpy(dtype=np.float64), 1e-18)
    row_var = np.square(data.standard_error) + variance[data.item_idx]
    z2 = np.square(data.estimate - mean[data.item_idx]) / np.maximum(row_var, 1e-18)
    return float(max(1.0, np.mean(z2)))


def _inflate_output(output: MethodOutput, inflation: float, method_name: str) -> MethodOutput:
    posterior = output.posterior.copy()
    posterior["posterior_var"] = np.maximum(posterior["posterior_var"].to_numpy(dtype=np.float64) * inflation, 1e-18)
    posterior["posterior_sd"] = np.sqrt(posterior["posterior_var"].to_numpy(dtype=np.float64))
    posterior["posterior_second_moment"] = (
        np.square(posterior["posterior_mean"].to_numpy(dtype=np.float64))
        + posterior["posterior_var"].to_numpy(dtype=np.float64)
    )
    posterior["method_name"] = method_name
    diagnostics = dict(output.diagnostics)
    diagnostics.update({"posthoc_variance_inflation_c2": float(inflation)})
    return MethodOutput(posterior, diagnostics)


def raw_single_reference(data: IndexedData) -> MethodOutput:
    mean = np.zeros(data.n_items, dtype=np.float64)
    variance = np.full(data.n_items, np.inf, dtype=np.float64)
    reference_rows = data.is_reference_view[data.view_idx]
    for item_idx in range(data.n_items):
        rows = (data.item_idx == item_idx) & reference_rows
        if not np.any(rows):
            continue
        weights = 1.0 / np.square(data.standard_error[rows])
        weight_sum = float(np.sum(weights))
        mean[item_idx] = float(np.sum(weights * data.estimate[rows]) / weight_sum)
        variance[item_idx] = 1.0 / weight_sum
    missing = ~np.isfinite(variance)
    if np.any(missing):
        fallback = inverse_variance_all_views(data).posterior
        mean[missing] = fallback.loc[missing, "posterior_mean"].to_numpy(dtype=np.float64)
        variance[missing] = fallback.loc[missing, "posterior_var"].to_numpy(dtype=np.float64)
    posterior = _posterior_state(mean, variance)
    return MethodOutput(
        posterior_to_frame(data, posterior, "raw_single_reference"),
        _with_graph_diagnostics(data, {"n_missing_reference_items": int(np.sum(missing))}),
    )


def inverse_variance_all_views(data: IndexedData) -> MethodOutput:
    precision = np.zeros(data.n_items, dtype=np.float64)
    natural = np.zeros(data.n_items, dtype=np.float64)
    row_precision = 1.0 / np.square(data.standard_error)
    np.add.at(precision, data.item_idx, row_precision)
    np.add.at(natural, data.item_idx, data.estimate * row_precision)
    variance = 1.0 / np.maximum(precision, 1e-18)
    mean = variance * natural
    posterior = _posterior_state(mean, variance)
    return MethodOutput(
        posterior_to_frame(data, posterior, "inverse_variance_all_views"),
        _with_graph_diagnostics(data, {}),
    )


def normal_eb_no_features(data: IndexedData, *, eps: float = 1e-8, max_iter: int = 200) -> MethodOutput:
    collapsed = inverse_variance_all_views(data).posterior
    mu_init = float(collapsed["posterior_mean"].mean())
    var_init = max(float(collapsed["posterior_mean"].var(ddof=0)), eps)
    x0 = np.array([mu_init, np.log(var_init)], dtype=np.float64)
    view_scale = np.ones(data.n_views, dtype=np.float64)
    extra_noise = np.zeros(data.n_views, dtype=np.float64)

    def objective(params: np.ndarray) -> float:
        mu0 = params[0]
        v0 = np.exp(params[1]) + eps
        prior_mean = np.full(data.n_items, mu0, dtype=np.float64)
        prior_variance = np.full(data.n_items, v0, dtype=np.float64)
        return -marginal_log_likelihood(data, prior_mean, prior_variance, view_scale, extra_noise)

    result = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": max_iter})
    mu0 = float(result.x[0])
    v0 = float(np.exp(result.x[1]) + eps)
    posterior = compute_posterior_from_prior(
        data,
        np.full(data.n_items, mu0, dtype=np.float64),
        np.full(data.n_items, v0, dtype=np.float64),
        view_scale,
        extra_noise,
    )
    return MethodOutput(
        posterior_to_frame(data, posterior, "normal_eb_no_features"),
        _with_graph_diagnostics(
            data,
            {"mu0": mu0, "v0": v0, "objective": float(result.fun), "success": bool(result.success)},
        ),
    )


def feature_normal_eb_no_view_scale(data: IndexedData, *, eps: float = 1e-8, max_iter: int = 200) -> MethodOutput:
    n_features = data.item_features.shape[1]
    collapsed = inverse_variance_all_views(data).posterior["posterior_mean"].to_numpy(dtype=np.float64)
    w_mu0, *_ = np.linalg.lstsq(data.item_features, collapsed, rcond=None)
    residual = collapsed - data.item_features @ w_mu0
    variance_guess = max(float(np.var(residual)), eps)
    w_v0 = np.zeros(n_features, dtype=np.float64)
    w_v0[0] = inverse_softplus_numpy(np.array([variance_guess], dtype=np.float64))[0]
    x0 = np.concatenate([w_mu0, w_v0])
    view_scale = np.ones(data.n_views, dtype=np.float64)
    extra_noise = np.zeros(data.n_views, dtype=np.float64)

    def objective(params: np.ndarray) -> float:
        w_mu = params[:n_features]
        w_v = params[n_features:]
        prior_mean, prior_variance = prior_from_features(data.item_features, w_mu, w_v, eps=eps)
        return -marginal_log_likelihood(data, prior_mean, prior_variance, view_scale, extra_noise)

    result = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": max_iter})
    w_mu = result.x[:n_features]
    w_v = result.x[n_features:]
    posterior = compute_posterior(data, w_mu, w_v, view_scale, extra_noise, eps=eps)
    return MethodOutput(
        posterior_to_frame(data, posterior, "feature_normal_eb_no_view_scale"),
        _with_graph_diagnostics(
            data,
            {"w_mu": w_mu, "w_v": w_v, "objective": float(result.fun), "success": bool(result.success)},
        ),
    )


def joint_eb_full(
    data: IndexedData,
    *,
    eps: float = 1e-8,
    max_iter: int = 200,
    extra_noise_upper: float = 10.0,
) -> MethodOutput:
    n_features = data.item_features.shape[1]
    reference = data.is_reference_view
    free_views = np.flatnonzero(~reference)
    base = feature_normal_eb_no_view_scale(data, eps=eps, max_iter=max(25, max_iter // 4))
    w_mu0 = np.linalg.lstsq(
        data.item_features,
        base.posterior["posterior_mean"].to_numpy(dtype=np.float64),
        rcond=None,
    )[0]
    w_v0 = np.zeros(n_features, dtype=np.float64)
    mean_posterior_var = max(float(base.posterior["posterior_var"].mean()), eps)
    w_v0[0] = inverse_softplus_numpy(np.array([mean_posterior_var], dtype=np.float64))[0]
    x0 = np.concatenate([w_mu0, w_v0, np.zeros(free_views.size), np.full(data.n_views, np.log(1e-4))])

    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pos = 0
        w_mu = params[pos : pos + n_features]
        pos += n_features
        w_v = params[pos : pos + n_features]
        pos += n_features
        view_scale = np.ones(data.n_views, dtype=np.float64)
        view_scale[free_views] = np.exp(params[pos : pos + free_views.size])
        pos += free_views.size
        extra_noise = np.minimum(np.exp(params[pos : pos + data.n_views]), extra_noise_upper)
        return w_mu, w_v, view_scale, extra_noise

    def objective(params: np.ndarray) -> float:
        w_mu, w_v, view_scale, extra_noise = unpack(params)
        prior_mean, prior_variance = prior_from_features(data.item_features, w_mu, w_v, eps=eps)
        penalty = 1e-4 * float(np.sum(params[: 2 * n_features] ** 2))
        return -marginal_log_likelihood(data, prior_mean, prior_variance, view_scale, extra_noise) + penalty

    result = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": max_iter})
    w_mu, w_v, view_scale, extra_noise = unpack(result.x)
    posterior = compute_posterior(data, w_mu, w_v, view_scale, extra_noise, eps=eps)
    diagnostics = {
        "objective": float(result.fun),
        "success": bool(result.success),
        "view_scale": view_scale,
        "extra_noise": extra_noise,
        "n_iterations": int(result.nit),
    }
    return MethodOutput(posterior_to_frame(data, posterior, "joint_eb_full"), _with_graph_diagnostics(data, diagnostics))


def anchored_calibrated_eb(
    data: IndexedData,
    config: ModelConfig | None = None,
    *,
    max_iter: int = 200,
    residual_calibration: str | None = None,
    n_crossfit_folds: int | None = None,
    random_state: int | None = None,
    fit_view_bias: bool | None = None,
    center_reference: bool | None = None,
) -> MethodOutput:
    result = fit_staged_calibrated_eb(
        data,
        config,
        max_iter=max_iter,
        residual_calibration=residual_calibration,
        n_crossfit_folds=n_crossfit_folds,
        random_state=random_state,
        fit_view_bias=fit_view_bias,
        center_reference=center_reference,
    )
    diagnostics = {
        "view_scale": result.view_scale,
        "extra_noise": result.extra_noise,
        "view_bias": result.view_bias if result.view_bias is not None else np.zeros(data.n_views, dtype=np.float64),
        "history": result.history,
    }
    if result.diagnostics is not None:
        diagnostics.update(result.diagnostics)
    return MethodOutput(
        posterior_to_frame(data, result.posterior, "anchored_calibrated_eb"),
        _with_graph_diagnostics(data, diagnostics),
    )


def random_effects_meta_analysis(data: IndexedData) -> MethodOutput:
    mean = np.zeros(data.n_items, dtype=np.float64)
    variance = np.zeros(data.n_items, dtype=np.float64)
    random_effect_variance = np.zeros(data.n_items, dtype=np.float64)
    for item_idx in range(data.n_items):
        rows = data.item_idx == item_idx
        y = data.estimate[rows]
        se2 = np.square(data.standard_error[rows])
        fixed_weights = 1.0 / se2
        fixed_mean = float(np.sum(fixed_weights * y) / np.sum(fixed_weights))
        if y.size <= 1:
            random_effect_variance_i = 0.0
        else:
            q = float(np.sum(fixed_weights * np.square(y - fixed_mean)))
            c = float(np.sum(fixed_weights) - np.sum(fixed_weights * fixed_weights) / np.sum(fixed_weights))
            random_effect_variance_i = max(0.0, (q - (y.size - 1)) / c) if c > 0.0 else 0.0
        weights = 1.0 / (se2 + random_effect_variance_i)
        mean[item_idx] = float(np.sum(weights * y) / np.sum(weights))
        variance[item_idx] = 1.0 / float(np.sum(weights))
        random_effect_variance[item_idx] = random_effect_variance_i
    posterior = _posterior_state(mean, variance)
    return MethodOutput(
        posterior_to_frame(data, posterior, "random_effects_meta_analysis"),
        _with_graph_diagnostics(
            data,
            {
                "mean_random_effect_variance": float(np.mean(random_effect_variance)),
                "random_effect_variance": random_effect_variance,
            },
        ),
    )


def _collapse_all_views(data: IndexedData) -> tuple[np.ndarray, np.ndarray]:
    collapsed = inverse_variance_all_views(data).posterior
    return (
        collapsed["posterior_mean"].to_numpy(dtype=np.float64),
        collapsed["posterior_var"].to_numpy(dtype=np.float64),
    )


def adaptive_shrinkage_like(
    data: IndexedData,
    *,
    sigma_grid: np.ndarray | None = None,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> MethodOutput:
    estimate, variance = _collapse_all_views(data)
    if sigma_grid is None:
        max_abs = max(float(np.max(np.abs(estimate))), float(np.sqrt(np.max(variance))), 1e-3)
        sigma_grid = np.concatenate([[0.0], np.geomspace(max_abs / 20.0, max_abs * 2.0, 12)])
    sigma2_grid = np.square(np.asarray(sigma_grid, dtype=np.float64))
    weights = np.full(sigma2_grid.size, 1.0 / sigma2_grid.size, dtype=np.float64)
    log_likelihood = -np.inf

    for iteration in range(max_iter):
        component_var = variance[:, None] + sigma2_grid[None, :]
        log_prob = np.log(weights[None, :] + 1e-300) + norm.logpdf(estimate[:, None], 0.0, np.sqrt(component_var))
        normalizer = logsumexp(log_prob, axis=1)
        responsibility = np.exp(log_prob - normalizer[:, None])
        weights = responsibility.mean(axis=0)
        new_log_likelihood = float(np.sum(normalizer))
        if abs(new_log_likelihood - log_likelihood) < tol:
            break
        log_likelihood = new_log_likelihood

    component_var = variance[:, None] + sigma2_grid[None, :]
    log_prob = np.log(weights[None, :] + 1e-300) + norm.logpdf(estimate[:, None], 0.0, np.sqrt(component_var))
    responsibility = np.exp(log_prob - logsumexp(log_prob, axis=1)[:, None])
    shrink = np.divide(sigma2_grid[None, :], component_var, out=np.zeros_like(component_var), where=component_var > 0.0)
    component_mean = shrink * estimate[:, None]
    component_post_var = np.divide(
        sigma2_grid[None, :] * variance[:, None],
        component_var,
        out=np.zeros_like(component_var),
        where=component_var > 0.0,
    )
    mean = np.sum(responsibility * component_mean, axis=1)
    second = np.sum(responsibility * (component_post_var + component_mean * component_mean), axis=1)
    posterior_var = np.maximum(second - mean * mean, 1e-18)
    sign_prob_negative = np.sum(
        responsibility * norm.cdf(0.0, loc=component_mean, scale=np.sqrt(np.maximum(component_post_var, 1e-18))),
        axis=1,
    )
    lfsr = np.minimum(sign_prob_negative, 1.0 - sign_prob_negative)
    posterior = _posterior_state(mean, posterior_var)
    diagnostics = {
        "mixture_weight": weights,
        "sigma_grid": np.sqrt(sigma2_grid),
        "log_likelihood": log_likelihood,
        "n_iterations": int(iteration + 1),
    }
    return MethodOutput(
        posterior_to_frame(data, posterior, "adaptive_shrinkage_like", lfsr=lfsr),
        _with_graph_diagnostics(data, diagnostics),
    )


def posthoc_variance_inflation(
    data: IndexedData,
    base_method: Callable[..., MethodOutput] = inverse_variance_all_views,
    *,
    truth: pd.DataFrame | None = None,
    calibration_fraction: float = 0.3,
    random_state: int = 0,
    method_name: str | None = None,
    max_iter: int | None = None,
) -> MethodOutput:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1.")
    rng = np.random.default_rng(random_state)
    n_calibration = max(1, min(data.n_items - 1, int(round(calibration_fraction * data.n_items))))
    permutation = rng.permutation(data.n_items)
    calibration_items = np.sort(permutation[:n_calibration])
    fit_items = np.sort(permutation[n_calibration:])
    fit_data = _subset_items(data, fit_items)
    calibration_data = _subset_items(data, calibration_items)

    kwargs = {"max_iter": max_iter} if max_iter is not None else {}
    fit_output = base_method(fit_data, **kwargs)
    calibration_output = base_method(calibration_data, **kwargs)
    truth_values = _truth_by_item(calibration_data, truth)
    if truth_values is not None:
        calibration_mean = calibration_output.posterior["posterior_mean"].to_numpy(dtype=np.float64)
        calibration_sd = np.maximum(calibration_output.posterior["posterior_sd"].to_numpy(dtype=np.float64), 1e-9)
        z2 = np.square((calibration_mean - truth_values) / calibration_sd)
        inflation = float(max(1.0, np.mean(z2)))
        calibration_source = "truth"
    else:
        inflation = _heldout_view_inflation(calibration_data, calibration_output.posterior)
        calibration_source = "heldout_views"

    full_output = base_method(data, **kwargs)
    output_name = method_name or f"posthoc_variance_inflation_{full_output.posterior['method_name'].iloc[0]}"
    inflated = _inflate_output(full_output, inflation, output_name)
    inflated.diagnostics.update(
        {
            "base_method_name": str(full_output.posterior["method_name"].iloc[0]),
            "posthoc_calibration_fraction": float(calibration_fraction),
            "posthoc_calibration_source": calibration_source,
            "posthoc_n_fit_items": int(fit_data.n_items),
            "posthoc_n_calibration_items": int(calibration_data.n_items),
            "posthoc_fit_success": bool(fit_output.diagnostics.get("success", True)),
            "random_state": int(random_state),
        }
    )
    return inflated


def posthoc_inverse_variance_all_views(
    data: IndexedData,
    *,
    truth: pd.DataFrame | None = None,
    calibration_fraction: float = 0.3,
    random_state: int = 0,
) -> MethodOutput:
    return posthoc_variance_inflation(
        data,
        inverse_variance_all_views,
        truth=truth,
        calibration_fraction=calibration_fraction,
        random_state=random_state,
        method_name="posthoc_inverse_variance_all_views",
    )


def bootstrap_method(
    data: IndexedData,
    base_method: Callable[..., MethodOutput],
    *,
    B: int = 20,
    random_state: int = 0,
    method_name: str,
    max_iter: int | None = None,
) -> MethodOutput:
    if B < 1:
        raise ValueError("B must be positive.")
    start = time.perf_counter()
    kwargs = {"max_iter": max_iter} if max_iter is not None else {}
    base_output = base_method(data, **kwargs)
    base_posterior = base_output.posterior.copy()
    rng = np.random.default_rng(random_state)
    means = []
    failures = 0
    for _ in range(B):
        bootstrap_data = _perturb_estimates(data, rng)
        try:
            output = base_method(bootstrap_data, **kwargs)
            means.append(output.posterior["posterior_mean"].to_numpy(dtype=np.float64))
        except Exception:
            failures += 1
    if means:
        boot_means = np.vstack(means)
        boot_var = np.var(boot_means, axis=0, ddof=1) if boot_means.shape[0] > 1 else np.zeros(data.n_items)
    else:
        boot_var = np.zeros(data.n_items, dtype=np.float64)
    posterior = base_posterior.copy()
    base_var = np.maximum(posterior["posterior_var"].to_numpy(dtype=np.float64), 1e-18)
    posterior["posterior_var"] = np.maximum(base_var + boot_var, 1e-18)
    posterior["posterior_sd"] = np.sqrt(posterior["posterior_var"].to_numpy(dtype=np.float64))
    posterior["posterior_second_moment"] = (
        np.square(posterior["posterior_mean"].to_numpy(dtype=np.float64))
        + posterior["posterior_var"].to_numpy(dtype=np.float64)
    )
    posterior["method_name"] = method_name
    diagnostics = dict(base_output.diagnostics)
    diagnostics.update(
        {
            "bootstrap_B": int(B),
            "bootstrap_successful_fits": int(len(means)),
            "bootstrap_failures": int(failures),
            "bootstrap_random_state": int(random_state),
            "bootstrap_runtime_seconds": float(time.perf_counter() - start),
            "base_method_name": str(base_output.posterior["method_name"].iloc[0]),
        }
    )
    return MethodOutput(posterior, diagnostics)


def bootstrap_inverse_variance_all_views(
    data: IndexedData,
    *,
    B: int = 5,
    random_state: int = 0,
) -> MethodOutput:
    return bootstrap_method(
        data,
        inverse_variance_all_views,
        B=B,
        random_state=random_state,
        method_name="bootstrap_inverse_variance_all_views",
    )


def bootstrap_reference_view(
    data: IndexedData,
    *,
    B: int = 5,
    random_state: int = 0,
) -> MethodOutput:
    return bootstrap_method(
        data,
        raw_single_reference,
        B=B,
        random_state=random_state,
        method_name="bootstrap_reference_view",
    )


def bootstrap_aceb_variance_correction(
    data: IndexedData,
    *,
    B: int = 3,
    random_state: int = 0,
    max_iter: int = 40,
) -> MethodOutput:
    return bootstrap_method(
        data,
        anchored_calibrated_eb,
        B=B,
        random_state=random_state,
        method_name="bootstrap_aceb_variance_correction",
        max_iter=max_iter,
    )


def no_pooling_baseline(data: IndexedData, reference_view_scale: float = 1.0) -> PosteriorState:
    prior_mean = np.zeros(data.n_items, dtype=np.float64)
    prior_variance = np.full(data.n_items, 1e12, dtype=np.float64)
    view_scale = np.ones(data.n_views, dtype=np.float64)
    view_scale[data.is_reference_view] = reference_view_scale
    extra_noise = np.zeros(data.n_views, dtype=np.float64)
    return compute_posterior_from_prior(data, prior_mean, prior_variance, view_scale, extra_noise)


BASELINE_METHODS = {
    "raw_single_reference": raw_single_reference,
    "inverse_variance_all_views": inverse_variance_all_views,
    "normal_eb_no_features": normal_eb_no_features,
    "feature_normal_eb_no_view_scale": feature_normal_eb_no_view_scale,
    "joint_eb_full": joint_eb_full,
    "anchored_calibrated_eb": anchored_calibrated_eb,
    "random_effects_meta_analysis": random_effects_meta_analysis,
    "adaptive_shrinkage_like": adaptive_shrinkage_like,
    "posthoc_inverse_variance_all_views": posthoc_inverse_variance_all_views,
    "bootstrap_inverse_variance_all_views": bootstrap_inverse_variance_all_views,
    "bootstrap_reference_view": bootstrap_reference_view,
    "bootstrap_aceb_variance_correction": bootstrap_aceb_variance_correction,
}
