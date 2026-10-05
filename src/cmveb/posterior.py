from __future__ import annotations

import numpy as np
from scipy.special import expit

from cmveb.schemas import IndexedData, PosteriorState
from cmveb.sufficient_stats import aggregate_sufficient_statistics


def softplus_numpy(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def inverse_softplus_numpy(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x + np.log(-np.expm1(-x))


def prior_from_features(
    item_features: np.ndarray,
    w_mu: np.ndarray,
    w_v: np.ndarray,
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    prior_mean = np.asarray(item_features, dtype=np.float64) @ np.asarray(w_mu, dtype=np.float64)
    prior_variance = softplus_numpy(np.asarray(item_features, dtype=np.float64) @ np.asarray(w_v, dtype=np.float64)) + eps
    return prior_mean, prior_variance


def compute_posterior_from_prior(
    data: IndexedData,
    prior_mean: np.ndarray,
    prior_variance: np.ndarray,
    view_scale: np.ndarray,
    extra_noise: np.ndarray,
    view_bias: np.ndarray | None = None,
) -> PosteriorState:
    prior_mean = np.asarray(prior_mean, dtype=np.float64)
    prior_variance = np.asarray(prior_variance, dtype=np.float64)
    if prior_mean.shape != (data.n_items,) or prior_variance.shape != (data.n_items,):
        raise ValueError("prior_mean and prior_variance must have one value per item.")
    if np.any(prior_variance <= 0.0):
        raise ValueError("prior_variance must be positive.")

    stats = aggregate_sufficient_statistics(data, view_scale, extra_noise, view_bias=view_bias)
    posterior_precision = 1.0 / prior_variance + stats.precision_addend
    variance = 1.0 / posterior_precision
    mean = variance * (prior_mean / prior_variance + stats.natural_addend)
    second_moment = mean * mean + variance
    return PosteriorState(mean, variance, second_moment, prior_mean, prior_variance)


def compute_posterior(
    data: IndexedData,
    w_mu: np.ndarray,
    w_v: np.ndarray,
    view_scale: np.ndarray,
    extra_noise: np.ndarray,
    *,
    eps: float = 1e-8,
    view_bias: np.ndarray | None = None,
) -> PosteriorState:
    prior_mean, prior_variance = prior_from_features(data.item_features, w_mu, w_v, eps=eps)
    return compute_posterior_from_prior(data, prior_mean, prior_variance, view_scale, extra_noise, view_bias=view_bias)


def marginal_log_likelihood(
    data: IndexedData,
    prior_mean: np.ndarray,
    prior_variance: np.ndarray,
    view_scale: np.ndarray,
    extra_noise: np.ndarray,
    view_bias: np.ndarray | None = None,
) -> float:
    if view_bias is None:
        view_bias = np.zeros(data.n_views, dtype=np.float64)
    else:
        view_bias = np.asarray(view_bias, dtype=np.float64)
        if view_bias.shape != (data.n_views,):
            raise ValueError("view_bias must have one value per view.")
    total = 0.0
    for item_idx in range(data.n_items):
        start = data.item_offsets[item_idx]
        stop = data.item_offsets[item_idx + 1]
        rows = slice(start, stop)
        views = data.view_idx[rows]
        scale = view_scale[views]
        bias = view_bias[views]
        diagonal = data.standard_error[rows] ** 2 + extra_noise[views] ** 2
        covariance = np.diag(diagonal) + prior_variance[item_idx] * np.outer(scale, scale)
        residual = data.estimate[rows] - bias - scale * prior_mean[item_idx]
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            return -np.inf
        solved = np.linalg.solve(covariance, residual)
        total += -0.5 * (len(residual) * np.log(2.0 * np.pi) + logdet + residual @ solved)
    return float(total)


def residual_moment_extra_noise(
    data: IndexedData,
    posterior: PosteriorState,
    view_scale: np.ndarray,
    *,
    lower: float = 0.0,
    upper: float = np.inf,
    shrinkage: float = 1.0,
) -> np.ndarray:
    if not (0.0 <= shrinkage <= 1.0):
        raise ValueError("shrinkage must be between 0 and 1.")
    raw = np.zeros(data.n_views, dtype=np.float64)
    for view_idx in range(data.n_views):
        rows = data.view_idx == view_idx
        if not np.any(rows):
            raw[view_idx] = lower
            continue
        item_idx = data.item_idx[rows]
        scale = view_scale[view_idx]
        residual_second_moment = (
            (data.estimate[rows] - scale * posterior.mean[item_idx]) ** 2
            + scale * scale * posterior.variance[item_idx]
        )
        moment = float(np.mean(residual_second_moment - data.standard_error[rows] ** 2))
        raw[view_idx] = max(moment, 0.0)
    bounded_variance = np.clip(shrinkage * raw, lower * lower, upper * upper)
    return np.sqrt(bounded_variance)
