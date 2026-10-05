from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cmveb.schemas import IndexedData


@dataclass(slots=True)
class LeaveViewPosteriorByRow:
    mean: np.ndarray
    variance: np.ndarray
    row_precision: np.ndarray
    row_natural: np.ndarray
    total_precision: np.ndarray
    total_natural: np.ndarray


@dataclass(slots=True)
class TauCalibrationResult:
    tau2_by_view: np.ndarray
    tau_by_view: np.ndarray
    n_rows_by_view: np.ndarray
    tau2_raw_before_positive_part: np.ndarray
    tau2_num_negative_raw: int
    residual_calibration_method: str
    crossfit_n_folds: int

    def diagnostics(self) -> dict[str, object]:
        return {
            "residual_calibration_method": self.residual_calibration_method,
            "tau2_by_view": self.tau2_by_view,
            "tau_by_view": self.tau_by_view,
            "n_rows_by_view": self.n_rows_by_view,
            "tau2_raw_before_positive_part": self.tau2_raw_before_positive_part,
            "tau2_num_negative_raw": self.tau2_num_negative_raw,
            "crossfit_n_folds": self.crossfit_n_folds,
        }


def _bias_by_row(data: IndexedData, bias: np.ndarray | None) -> np.ndarray:
    if bias is None:
        return np.zeros(data.n_rows, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    if bias.shape != (data.n_views,):
        raise ValueError("bias must have one value per view.")
    return bias[data.view_idx]


def compute_leave_view_posterior_by_row(
    data: IndexedData,
    prior_mean: np.ndarray,
    prior_variance: np.ndarray,
    view_scale: np.ndarray,
    extra_noise: np.ndarray,
    *,
    bias: np.ndarray | None = None,
    variance_floor: float = 1e-12,
) -> LeaveViewPosteriorByRow:
    prior_mean = np.asarray(prior_mean, dtype=np.float64)
    prior_variance = np.asarray(prior_variance, dtype=np.float64)
    view_scale = np.asarray(view_scale, dtype=np.float64)
    extra_noise = np.asarray(extra_noise, dtype=np.float64)
    if prior_mean.shape != (data.n_items,) or prior_variance.shape != (data.n_items,):
        raise ValueError("prior_mean and prior_variance must have one value per item.")
    if view_scale.shape != (data.n_views,) or extra_noise.shape != (data.n_views,):
        raise ValueError("view_scale and extra_noise must have one value per view.")
    prior_variance = np.maximum(prior_variance, variance_floor)
    sigma2 = np.maximum(data.standard_error * data.standard_error + extra_noise[data.view_idx] ** 2, variance_floor)
    scale_by_row = view_scale[data.view_idx]
    bias_by_row = _bias_by_row(data, bias)
    row_precision = scale_by_row * scale_by_row / sigma2
    row_natural = scale_by_row * (data.estimate - bias_by_row) / sigma2

    total_precision = 1.0 / prior_variance
    total_natural = prior_mean / prior_variance
    np.add.at(total_precision, data.item_idx, row_precision)
    np.add.at(total_natural, data.item_idx, row_natural)

    precision_minus = np.maximum(total_precision[data.item_idx] - row_precision, variance_floor)
    natural_minus = total_natural[data.item_idx] - row_natural
    variance_minus = 1.0 / precision_minus
    mean_minus = variance_minus * natural_minus
    return LeaveViewPosteriorByRow(
        mean=mean_minus,
        variance=variance_minus,
        row_precision=row_precision,
        row_natural=row_natural,
        total_precision=total_precision,
        total_natural=total_natural,
    )


def estimate_tau2_leave_view(
    data: IndexedData,
    prior_mean: np.ndarray,
    prior_variance: np.ndarray,
    view_scale: np.ndarray,
    extra_noise: np.ndarray,
    *,
    bias: np.ndarray | None = None,
    lower: float = 0.0,
    upper: float = np.inf,
    variance_floor: float = 1e-12,
) -> TauCalibrationResult:
    leave = compute_leave_view_posterior_by_row(
        data,
        prior_mean,
        prior_variance,
        view_scale,
        extra_noise,
        bias=bias,
        variance_floor=variance_floor,
    )
    scale_by_row = view_scale[data.view_idx]
    bias_by_row = _bias_by_row(data, bias)
    raw_by_row = (
        (data.estimate - bias_by_row - scale_by_row * leave.mean) ** 2
        - scale_by_row * scale_by_row * leave.variance
        - data.standard_error * data.standard_error
    )
    return summarize_tau2_by_view(
        raw_by_row,
        data.view_idx,
        data.n_views,
        lower=lower,
        upper=upper,
        method="leave_view",
        crossfit_n_folds=0,
    )


def summarize_tau2_by_view(
    raw_by_row: np.ndarray,
    view_idx: np.ndarray,
    n_views: int,
    *,
    lower: float = 0.0,
    upper: float = np.inf,
    method: str,
    crossfit_n_folds: int,
) -> TauCalibrationResult:
    raw_by_row = np.asarray(raw_by_row, dtype=np.float64)
    view_idx = np.asarray(view_idx, dtype=np.int64)
    tau2_raw = np.zeros(n_views, dtype=np.float64)
    n_rows = np.zeros(n_views, dtype=np.int64)
    for view in range(n_views):
        rows = view_idx == view
        n_rows[view] = int(np.sum(rows))
        tau2_raw[view] = float(np.mean(raw_by_row[rows])) if np.any(rows) else 0.0
    lower2 = lower * lower
    upper2 = upper * upper
    tau2 = np.clip(np.maximum(tau2_raw, 0.0), lower2, upper2)
    return TauCalibrationResult(
        tau2_by_view=tau2,
        tau_by_view=np.sqrt(np.maximum(tau2, 0.0)),
        n_rows_by_view=n_rows,
        tau2_raw_before_positive_part=tau2_raw,
        tau2_num_negative_raw=int(np.sum(tau2_raw < 0.0)),
        residual_calibration_method=method,
        crossfit_n_folds=int(crossfit_n_folds),
    )
