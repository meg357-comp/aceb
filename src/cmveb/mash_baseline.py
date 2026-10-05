from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cmveb.models.baseline import MethodOutput
from cmveb.schemas import IndexedData


class MethodSkipped(RuntimeError):
    """Raised when an optional comparator is unavailable."""


@dataclass(slots=True)
class MashMatrixData:
    bhat: pd.DataFrame
    shat: pd.DataFrame
    kept_item_ids: np.ndarray
    kept_view_ids: np.ndarray
    dropped_items: int
    dropped_views: int
    missing_mode: str


def indexed_to_mash_matrices(
    data: IndexedData,
    *,
    missing: str = "complete_case",
    large_se: float = 1e6,
) -> MashMatrixData:
    if missing not in {"complete_case", "large_se"}:
        raise ValueError("missing must be 'complete_case' or 'large_se'.")
    bhat = np.full((data.n_items, data.n_views), np.nan, dtype=np.float64)
    shat = np.full((data.n_items, data.n_views), np.nan, dtype=np.float64)
    bhat[data.item_idx, data.view_idx] = data.estimate
    shat[data.item_idx, data.view_idx] = data.standard_error

    if missing == "complete_case":
        item_mask = np.all(np.isfinite(bhat) & np.isfinite(shat), axis=1)
        view_mask = np.any(np.isfinite(bhat[item_mask]), axis=0) if np.any(item_mask) else np.zeros(data.n_views, dtype=bool)
        bhat = bhat[item_mask][:, view_mask]
        shat = shat[item_mask][:, view_mask]
    else:
        item_mask = np.ones(data.n_items, dtype=bool)
        view_mask = np.any(np.isfinite(bhat), axis=0)
        bhat = bhat[:, view_mask]
        shat = shat[:, view_mask]
        missing_cells = ~np.isfinite(bhat) | ~np.isfinite(shat)
        bhat[missing_cells] = 0.0
        shat[missing_cells] = large_se

    kept_item_ids = data.item_ids[item_mask]
    kept_view_ids = data.view_ids[view_mask]
    bhat_frame = pd.DataFrame(bhat, index=kept_item_ids, columns=kept_view_ids)
    shat_frame = pd.DataFrame(shat, index=kept_item_ids, columns=kept_view_ids)
    return MashMatrixData(
        bhat=bhat_frame,
        shat=shat_frame,
        kept_item_ids=kept_item_ids,
        kept_view_ids=kept_view_ids,
        dropped_items=int(data.n_items - kept_item_ids.size),
        dropped_views=int(data.n_views - kept_view_ids.size),
        missing_mode=missing,
    )


def skipped_mash_output(data: IndexedData, reason: str, *, method_name: str = "mashr_multivariate_eb") -> MethodOutput:
    posterior = pd.DataFrame(
        {
            "item_id": data.item_ids,
            "posterior_mean": np.full(data.n_items, np.nan),
            "posterior_var": np.full(data.n_items, np.nan),
            "posterior_sd": np.full(data.n_items, np.nan),
            "posterior_second_moment": np.full(data.n_items, np.nan),
            "lfsr": np.full(data.n_items, np.nan),
            "method_name": method_name,
        }
    )
    return MethodOutput(posterior, {"skipped": True, "skip_reason": reason, "mash_available": False})


def _write_r_script(path: Path) -> None:
    path.write_text(
        """
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 4) {
  stop("usage: run_mashr.R Bhat.csv Shat.csv posterior_mean.csv posterior_sd.csv")
}
if (!requireNamespace("mashr", quietly=TRUE)) {
  message("mashr is not installed")
  quit(status=66)
}
Bhat <- as.matrix(read.csv(args[[1]], row.names=1, check.names=FALSE))
Shat <- as.matrix(read.csv(args[[2]], row.names=1, check.names=FALSE))
data <- mashr::mash_set_data(Bhat, Shat)
U.c <- mashr::cov_canonical(data)
m <- mashr::mash(data, U.c)
pm <- mashr::get_pm(m)
psd <- tryCatch(mashr::get_psd(m), error=function(e) {
  matrix(NA_real_, nrow=nrow(pm), ncol=ncol(pm), dimnames=dimnames(pm))
})
write.csv(pm, args[[3]], quote=FALSE)
write.csv(psd, args[[4]], quote=FALSE)
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _aggregate_mash_scores(
    pm: pd.DataFrame,
    psd: pd.DataFrame,
    *,
    aggregation: str,
    reference_view_id: object | None,
) -> tuple[np.ndarray, np.ndarray]:
    if aggregation not in {"inverse_variance", "reference"}:
        raise ValueError("aggregation must be 'inverse_variance' or 'reference'.")
    mean_matrix = pm.to_numpy(dtype=np.float64)
    sd_matrix = psd.to_numpy(dtype=np.float64)
    if aggregation == "reference":
        column = reference_view_id if reference_view_id in pm.columns else pm.columns[0]
        mean = pm[column].to_numpy(dtype=np.float64)
        sd = psd[column].to_numpy(dtype=np.float64) if column in psd.columns else np.full(pm.shape[0], np.nan)
        var = np.square(sd)
        if not np.all(np.isfinite(var)):
            var = np.nanvar(mean_matrix, axis=1) + 1e-6
        return mean, np.maximum(var, 1e-18)

    var_matrix = np.square(sd_matrix)
    finite = np.isfinite(var_matrix) & (var_matrix > 0.0)
    weights = np.divide(1.0, var_matrix, out=np.zeros_like(var_matrix), where=finite)
    weight_sum = np.sum(weights, axis=1)
    fallback = weight_sum <= 0.0
    mean = np.divide(
        np.sum(weights * mean_matrix, axis=1),
        weight_sum,
        out=np.nanmean(mean_matrix, axis=1),
        where=weight_sum > 0.0,
    )
    var = np.divide(1.0, weight_sum, out=np.nanvar(mean_matrix, axis=1) + 1e-6, where=weight_sum > 0.0)
    var[fallback] = np.maximum(var[fallback], 1e-6)
    return mean, np.maximum(var, 1e-18)


def mashr_multivariate_eb(
    data: IndexedData,
    *,
    missing: str = "complete_case",
    aggregation: str = "inverse_variance",
    large_se: float = 1e6,
    rscript_path: str | None = None,
    raise_on_skip: bool = False,
) -> MethodOutput:
    rscript = rscript_path or shutil.which("Rscript")
    if rscript is None:
        reason = "Rscript not found on PATH."
        if raise_on_skip:
            raise MethodSkipped(reason)
        return skipped_mash_output(data, reason)

    matrix_data = indexed_to_mash_matrices(data, missing=missing, large_se=large_se)
    if matrix_data.bhat.empty or matrix_data.bhat.shape[1] == 0:
        reason = "No complete MASH matrix after missing-data filtering."
        if raise_on_skip:
            raise MethodSkipped(reason)
        return skipped_mash_output(data, reason)

    reference_view_id = data.view_ids[np.flatnonzero(data.is_reference_view)[0]] if np.any(data.is_reference_view) else None
    with tempfile.TemporaryDirectory(prefix="cmveb_mashr_") as tmp:
        tmpdir = Path(tmp)
        bhat_path = tmpdir / "Bhat.csv"
        shat_path = tmpdir / "Shat.csv"
        pm_path = tmpdir / "posterior_mean.csv"
        psd_path = tmpdir / "posterior_sd.csv"
        script_path = tmpdir / "run_mashr.R"
        matrix_data.bhat.to_csv(bhat_path)
        matrix_data.shat.to_csv(shat_path)
        _write_r_script(script_path)
        result = subprocess.run(
            [rscript, str(script_path), str(bhat_path), str(shat_path), str(pm_path), str(psd_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            reason = result.stderr.strip() or result.stdout.strip() or f"Rscript exited with {result.returncode}."
            if raise_on_skip:
                raise MethodSkipped(reason)
            return skipped_mash_output(data, reason)
        pm = pd.read_csv(pm_path, index_col=0)
        psd = pd.read_csv(psd_path, index_col=0)

    kept_mean, kept_var = _aggregate_mash_scores(
        pm,
        psd,
        aggregation=aggregation,
        reference_view_id=reference_view_id,
    )
    mean = np.full(data.n_items, np.nan, dtype=np.float64)
    var = np.full(data.n_items, np.nan, dtype=np.float64)
    item_lookup = {item_id: idx for idx, item_id in enumerate(data.item_ids.tolist())}
    kept_indices = np.array([item_lookup[item_id] for item_id in matrix_data.kept_item_ids], dtype=np.int64)
    mean[kept_indices] = kept_mean
    var[kept_indices] = kept_var
    posterior = pd.DataFrame(
        {
            "item_id": data.item_ids,
            "posterior_mean": mean,
            "posterior_var": var,
            "posterior_sd": np.sqrt(var),
            "posterior_second_moment": mean * mean + var,
            "lfsr": np.nan,
            "method_name": "mashr_multivariate_eb",
        }
    )
    diagnostics = {
        "mash_available": True,
        "missing_mode": missing,
        "aggregation": aggregation,
        "dropped_items": matrix_data.dropped_items,
        "dropped_views": matrix_data.dropped_views,
        "n_mash_items": int(matrix_data.bhat.shape[0]),
        "n_mash_views": int(matrix_data.bhat.shape[1]),
        "skipped": False,
    }
    return MethodOutput(posterior, diagnostics)
