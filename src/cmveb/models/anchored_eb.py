from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from cmveb.calibration import estimate_tau2_leave_view, summarize_tau2_by_view, compute_leave_view_posterior_by_row
from cmveb.config import ModelConfig
from cmveb.posterior import (
    compute_posterior,
    inverse_softplus_numpy,
    marginal_log_likelihood,
    prior_from_features,
    residual_moment_extra_noise,
)
from cmveb.schemas import FitResult, IndexedData


def initialize_view_scale(
    data: IndexedData,
    *,
    reference_scale: float = 1.0,
    non_reference_scale: float = 0.5,
) -> np.ndarray:
    view_scale = np.full(data.n_views, float(non_reference_scale), dtype=np.float64)
    view_scale[data.is_reference_view] = float(reference_scale)
    return view_scale


def initialize_extra_noise(data: IndexedData, value: float = 1e-4) -> np.ndarray:
    return np.full(data.n_views, float(value), dtype=np.float64)


def _initial_prior_parameters(data: IndexedData, eps: float) -> tuple[np.ndarray, np.ndarray]:
    n_features = data.item_features.shape[1]
    w_mu = np.zeros(n_features, dtype=np.float64)
    centered = data.estimate - float(np.mean(data.estimate))
    variance_guess = max(float(np.var(centered)), eps)
    w_v = np.zeros(n_features, dtype=np.float64)
    w_v[0] = inverse_softplus_numpy(np.array([variance_guess], dtype=np.float64))[0]
    return w_mu, w_v


def fit_prior_hyperparameters(
    data: IndexedData,
    view_scale: np.ndarray,
    extra_noise: np.ndarray,
    *,
    eps: float = 1e-8,
    l2_prior: float = 1e-4,
    max_iter: int = 200,
) -> tuple[np.ndarray, np.ndarray, float]:
    w_mu0, w_v0 = _initial_prior_parameters(data, eps)
    n_features = data.item_features.shape[1]
    x0 = np.concatenate([w_mu0, w_v0])

    def objective(params: np.ndarray) -> float:
        w_mu = params[:n_features]
        w_v = params[n_features:]
        prior_mean, prior_variance = prior_from_features(data.item_features, w_mu, w_v, eps=eps)
        value = -marginal_log_likelihood(data, prior_mean, prior_variance, view_scale, extra_noise)
        penalty = l2_prior * float(np.sum(w_mu * w_mu) + np.sum(w_v * w_v))
        return value + penalty

    result = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": max_iter})
    params = result.x
    return params[:n_features], params[n_features:], float(result.fun)


def fit_non_reference_view_scales(
    data: IndexedData,
    w_mu: np.ndarray,
    w_v: np.ndarray,
    view_scale: np.ndarray,
    extra_noise: np.ndarray,
    *,
    eps: float = 1e-8,
    l2_view_scale: float = 1e-4,
    max_iter: int = 200,
) -> tuple[np.ndarray, float]:
    reference = data.is_reference_view
    free_indices = np.flatnonzero(~reference)
    if free_indices.size == 0:
        return view_scale.copy(), 0.0

    prior_mean, prior_variance = prior_from_features(data.item_features, w_mu, w_v, eps=eps)
    x0 = np.log(np.maximum(view_scale[free_indices], eps))

    def objective(log_scale: np.ndarray) -> float:
        candidate = view_scale.copy()
        candidate[reference] = 1.0
        candidate[free_indices] = np.exp(log_scale)
        value = -marginal_log_likelihood(data, prior_mean, prior_variance, candidate, extra_noise)
        penalty = l2_view_scale * float(np.sum((candidate[free_indices] - 1.0) ** 2))
        return value + penalty

    result = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": max_iter})
    fitted = view_scale.copy()
    fitted[reference] = 1.0
    fitted[free_indices] = np.exp(result.x)
    return fitted, float(result.fun)


def _subset_items(data: IndexedData, selected_items: np.ndarray) -> IndexedData:
    selected_items = np.asarray(selected_items, dtype=np.int64)
    remap = {int(old): new for new, old in enumerate(selected_items.tolist())}
    row_mask = np.isin(data.item_idx, selected_items)
    old_row_items = data.item_idx[row_mask]
    new_item_idx = np.array([remap[int(item)] for item in old_row_items], dtype=np.int64)
    counts = np.bincount(new_item_idx, minlength=selected_items.size)
    item_offsets = np.zeros(selected_items.size + 1, dtype=np.int64)
    item_offsets[1:] = np.cumsum(counts)
    order = np.lexsort((data.view_idx[row_mask], new_item_idx))
    row_indices = np.flatnonzero(row_mask)[order]
    sorted_item_idx = new_item_idx[order]
    counts = np.bincount(sorted_item_idx, minlength=selected_items.size)
    item_offsets[1:] = np.cumsum(counts)
    return IndexedData(
        item_ids=data.item_ids[selected_items],
        view_ids=data.view_ids,
        item_features=np.ascontiguousarray(data.item_features[selected_items], dtype=np.float64),
        feature_columns=data.feature_columns,
        scaler=data.scaler,
        item_idx=np.ascontiguousarray(sorted_item_idx),
        view_idx=np.ascontiguousarray(data.view_idx[row_indices], dtype=np.int64),
        estimate=np.ascontiguousarray(data.estimate[row_indices], dtype=np.float64),
        standard_error=np.ascontiguousarray(data.standard_error[row_indices], dtype=np.float64),
        is_reference_view=data.is_reference_view,
        item_offsets=item_offsets,
        row_order=np.ascontiguousarray(data.row_order[row_indices], dtype=np.int64),
    )


def _estimate_tau_in_sample(
    data: IndexedData,
    posterior,
    view_scale: np.ndarray,
    *,
    view_bias: np.ndarray,
    lower: float,
    upper: float,
):
    scale_by_row = view_scale[data.view_idx]
    bias_by_row = view_bias[data.view_idx]
    raw_by_row = (
        (data.estimate - bias_by_row - scale_by_row * posterior.mean[data.item_idx]) ** 2
        + scale_by_row * scale_by_row * posterior.variance[data.item_idx]
        - data.standard_error * data.standard_error
    )
    return summarize_tau2_by_view(
        raw_by_row,
        data.view_idx,
        data.n_views,
        lower=lower,
        upper=upper,
        method="in_sample",
        crossfit_n_folds=0,
    )


def _estimate_tau_unit_crossfit(
    data: IndexedData,
    cfg: ModelConfig,
    *,
    n_crossfit_folds: int,
    random_state: int,
    max_iter: int,
    fit_view_bias: bool,
    center_reference: bool,
):
    n_folds = max(2, min(int(n_crossfit_folds), data.n_items))
    rng = np.random.default_rng(random_state)
    item_order = np.arange(data.n_items, dtype=np.int64)
    rng.shuffle(item_order)
    folds = np.array_split(item_order, n_folds)
    raw_values = []
    raw_views = []
    for heldout_items in folds:
        train_items = np.setdiff1d(np.arange(data.n_items, dtype=np.int64), heldout_items, assume_unique=False)
        train_data = _subset_items(data, train_items)
        heldout_data = _subset_items(data, heldout_items)
        fold_view_scale = initialize_view_scale(
            train_data,
            reference_scale=cfg.reference_view_scale,
            non_reference_scale=cfg.initial_non_reference_view_scale,
        )
        fold_extra_noise = initialize_extra_noise(train_data, cfg.initial_extra_noise)
        fold_w_mu, fold_w_v, _ = fit_prior_hyperparameters(
            train_data,
            fold_view_scale,
            fold_extra_noise,
            eps=cfg.eps,
            l2_prior=cfg.l2_prior,
            max_iter=max_iter,
        )
        fold_view_scale, _ = fit_non_reference_view_scales(
            train_data,
            fold_w_mu,
            fold_w_v,
            fold_view_scale,
            fold_extra_noise,
            eps=cfg.eps,
            l2_view_scale=cfg.l2_view_scale,
            max_iter=max_iter,
        )
        prior_mean, prior_variance = prior_from_features(heldout_data.item_features, fold_w_mu, fold_w_v, eps=cfg.eps)
        fold_view_bias = np.zeros(data.n_views, dtype=np.float64)
        if fit_view_bias:
            train_interim = compute_posterior(
                train_data,
                fold_w_mu,
                fold_w_v,
                fold_view_scale,
                fold_extra_noise,
                eps=cfg.eps,
            )
            fold_view_bias = _estimate_view_bias(
                train_data,
                train_interim,
                fold_view_scale,
                center_reference=center_reference,
            )
        leave = compute_leave_view_posterior_by_row(
            heldout_data,
            prior_mean,
            prior_variance,
            fold_view_scale,
            fold_extra_noise,
            bias=fold_view_bias,
        )
        scale_by_row = fold_view_scale[heldout_data.view_idx]
        bias_by_row = fold_view_bias[heldout_data.view_idx]
        raw_by_row = (
            (heldout_data.estimate - bias_by_row - scale_by_row * leave.mean) ** 2
            - scale_by_row * scale_by_row * leave.variance
            - heldout_data.standard_error * heldout_data.standard_error
        )
        raw_values.append(raw_by_row)
        raw_views.append(heldout_data.view_idx)
    return summarize_tau2_by_view(
        np.concatenate(raw_values) if raw_values else np.zeros(0, dtype=np.float64),
        np.concatenate(raw_views) if raw_views else np.zeros(0, dtype=np.int64),
        data.n_views,
        lower=cfg.extra_noise_lower,
        upper=cfg.extra_noise_upper,
        method="unit_crossfit",
        crossfit_n_folds=n_folds,
    )


def _estimate_view_bias(
    data: IndexedData,
    posterior,
    view_scale: np.ndarray,
    *,
    center_reference: bool,
) -> np.ndarray:
    view_bias = np.zeros(data.n_views, dtype=np.float64)
    weights = 1.0 / np.maximum(data.standard_error * data.standard_error, 1e-12)
    residual = data.estimate - view_scale[data.view_idx] * posterior.mean[data.item_idx]
    for view in range(data.n_views):
        rows = data.view_idx == view
        if np.any(rows):
            view_bias[view] = float(np.sum(weights[rows] * residual[rows]) / np.sum(weights[rows]))
    if not center_reference:
        view_bias[data.is_reference_view] = 0.0
    return view_bias


def fit_staged_calibrated_eb(
    data: IndexedData,
    config: ModelConfig | None = None,
    *,
    max_iter: int = 200,
    residual_calibration: str | None = None,
    n_crossfit_folds: int | None = None,
    random_state: int | None = None,
    crossfit_max_iter: int | None = None,
    fit_view_bias: bool | None = None,
    center_reference: bool | None = None,
) -> FitResult:
    cfg = config or ModelConfig()
    residual_calibration = residual_calibration or cfg.residual_calibration
    n_crossfit_folds = cfg.n_crossfit_folds if n_crossfit_folds is None else n_crossfit_folds
    random_state = cfg.random_state if random_state is None else random_state
    crossfit_max_iter = cfg.crossfit_max_iter if crossfit_max_iter is None else crossfit_max_iter
    fit_view_bias = cfg.fit_view_bias if fit_view_bias is None else fit_view_bias
    center_reference = cfg.center_reference if center_reference is None else center_reference
    if residual_calibration not in {"leave_view", "in_sample", "unit_crossfit"}:
        raise ValueError("residual_calibration must be 'leave_view', 'in_sample', or 'unit_crossfit'.")
    view_scale = initialize_view_scale(
        data,
        reference_scale=cfg.reference_view_scale,
        non_reference_scale=cfg.initial_non_reference_view_scale,
    )
    extra_noise = initialize_extra_noise(data, cfg.initial_extra_noise)
    history: list[dict[str, float]] = []

    w_mu, w_v, prior_objective = fit_prior_hyperparameters(
        data,
        view_scale,
        extra_noise,
        eps=cfg.eps,
        l2_prior=cfg.l2_prior,
        max_iter=max_iter,
    )
    history.append({"stage": 1.0, "objective": prior_objective})

    view_scale, scale_objective = fit_non_reference_view_scales(
        data,
        w_mu,
        w_v,
        view_scale,
        extra_noise,
        eps=cfg.eps,
        l2_view_scale=cfg.l2_view_scale,
        max_iter=max_iter,
    )
    history.append({"stage": 2.0, "objective": scale_objective})

    interim = compute_posterior(data, w_mu, w_v, view_scale, extra_noise, eps=cfg.eps)
    view_bias = (
        _estimate_view_bias(data, interim, view_scale, center_reference=center_reference)
        if fit_view_bias
        else np.zeros(data.n_views, dtype=np.float64)
    )
    interim = compute_posterior(data, w_mu, w_v, view_scale, extra_noise, eps=cfg.eps, view_bias=view_bias)
    prior_mean, prior_variance = prior_from_features(data.item_features, w_mu, w_v, eps=cfg.eps)
    if residual_calibration == "leave_view":
        tau_result = estimate_tau2_leave_view(
            data,
            prior_mean,
            prior_variance,
            view_scale,
            extra_noise,
            bias=view_bias,
            lower=cfg.extra_noise_lower,
            upper=cfg.extra_noise_upper,
        )
    elif residual_calibration == "in_sample":
        tau_result = _estimate_tau_in_sample(
            data,
            interim,
            view_scale,
            view_bias=view_bias,
            lower=cfg.extra_noise_lower,
            upper=cfg.extra_noise_upper,
        )
    else:
        tau_result = _estimate_tau_unit_crossfit(
            data,
            cfg,
            n_crossfit_folds=n_crossfit_folds,
            random_state=random_state,
            max_iter=crossfit_max_iter,
            fit_view_bias=bool(fit_view_bias),
            center_reference=bool(center_reference),
        )
    extra_noise = tau_result.tau_by_view
    posterior = compute_posterior(data, w_mu, w_v, view_scale, extra_noise, eps=cfg.eps, view_bias=view_bias)
    history.append({"stage": 3.0, "mean_extra_noise": float(np.mean(extra_noise))})
    diagnostics = tau_result.diagnostics()
    diagnostics["fit_view_bias"] = bool(fit_view_bias)
    diagnostics["center_reference"] = bool(center_reference)
    diagnostics["view_bias"] = view_bias
    return FitResult(posterior, view_scale, extra_noise, w_mu, w_v, history, diagnostics, view_bias)


@dataclass(slots=True)
class AnchoredEBModel:
    config: ModelConfig = field(default_factory=ModelConfig)
    result: FitResult | None = None

    def fit(
        self,
        data: IndexedData,
        *,
        max_iter: int = 200,
        residual_calibration: str | None = None,
        n_crossfit_folds: int | None = None,
        random_state: int | None = None,
        fit_view_bias: bool | None = None,
        center_reference: bool | None = None,
    ) -> FitResult:
        self.result = fit_staged_calibrated_eb(
            data,
            self.config,
            max_iter=max_iter,
            residual_calibration=residual_calibration,
            n_crossfit_folds=n_crossfit_folds,
            random_state=random_state,
            fit_view_bias=fit_view_bias,
            center_reference=center_reference,
        )
        return self.result

    def posterior(self):
        if self.result is None:
            raise RuntimeError("Call fit before requesting posterior.")
        return self.result.posterior
