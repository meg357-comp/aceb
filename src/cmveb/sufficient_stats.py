from __future__ import annotations

import numpy as np

from cmveb.schemas import IndexedData, SufficientStatistics


def observation_variance(data: IndexedData, extra_noise: np.ndarray) -> np.ndarray:
    if extra_noise.shape != (data.n_views,):
        raise ValueError("extra_noise must have one value per view.")
    return data.standard_error * data.standard_error + extra_noise[data.view_idx] ** 2


def aggregate_sufficient_statistics(
    data: IndexedData,
    view_scale: np.ndarray,
    extra_noise: np.ndarray,
    view_bias: np.ndarray | None = None,
) -> SufficientStatistics:
    if view_scale.shape != (data.n_views,):
        raise ValueError("view_scale must have one value per view.")
    if view_bias is None:
        view_bias = np.zeros(data.n_views, dtype=np.float64)
    else:
        view_bias = np.asarray(view_bias, dtype=np.float64)
        if view_bias.shape != (data.n_views,):
            raise ValueError("view_bias must have one value per view.")
    variance = observation_variance(data, extra_noise)
    if np.any(variance <= 0.0):
        raise ValueError("observation variances must be positive.")

    size = data.n_items
    precision_addend = np.zeros(size, dtype=np.float64)
    natural_addend = np.zeros(size, dtype=np.float64)
    second_moment_weight = np.zeros(size, dtype=np.float64)

    scale = view_scale[data.view_idx]
    precision = 1.0 / variance
    weighted_scale_sq = scale * scale * precision
    np.add.at(precision_addend, data.item_idx, weighted_scale_sq)
    np.add.at(natural_addend, data.item_idx, scale * (data.estimate - view_bias[data.view_idx]) * precision)
    np.add.at(second_moment_weight, data.item_idx, weighted_scale_sq)

    return SufficientStatistics(precision_addend, natural_addend, second_moment_weight)


def brute_force_sufficient_statistics(
    data: IndexedData,
    view_scale: np.ndarray,
    extra_noise: np.ndarray,
    view_bias: np.ndarray | None = None,
) -> SufficientStatistics:
    if view_bias is None:
        view_bias = np.zeros(data.n_views, dtype=np.float64)
    precision_addend = np.zeros(data.n_items, dtype=np.float64)
    natural_addend = np.zeros(data.n_items, dtype=np.float64)
    variance = observation_variance(data, extra_noise)
    for row_idx in range(data.n_rows):
        item_idx = data.item_idx[row_idx]
        scale = view_scale[data.view_idx[row_idx]]
        precision = 1.0 / variance[row_idx]
        precision_addend[item_idx] += scale * scale * precision
        natural_addend[item_idx] += scale * (data.estimate[row_idx] - view_bias[data.view_idx[row_idx]]) * precision
    return SufficientStatistics(precision_addend, natural_addend, precision_addend.copy())
