from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cmveb.models.baseline import MethodOutput
from cmveb.schemas import IndexedData


class FullBayesSkipped(RuntimeError):
    """Raised when the optional full Bayesian backend is unavailable."""


def build_stan_data(
    data: IndexedData,
    *,
    fit_bias: bool = False,
    bias_prior_scale: float = 1.0,
) -> dict[str, Any]:
    reference = np.flatnonzero(data.is_reference_view)
    ref_view = int(reference[0]) + 1 if reference.size else 1
    return {
        "N": int(data.n_items),
        "R": int(data.n_rows),
        "J": int(data.n_views),
        "item_idx": (data.item_idx + 1).astype(int).tolist(),
        "view_idx": (data.view_idx + 1).astype(int).tolist(),
        "y": data.estimate.astype(float).tolist(),
        "se": data.standard_error.astype(float).tolist(),
        "ref_view": ref_view,
        "fit_bias": int(fit_bias),
        "bias_prior_scale": float(bias_prior_scale),
    }


def skipped_full_bayes_output(
    data: IndexedData,
    reason: str,
    *,
    method_name: str = "full_bayes_hierarchical",
) -> MethodOutput:
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
    return MethodOutput(
        posterior,
        {
            "skipped": True,
            "skip_reason": reason,
            "full_bayes_available": False,
            "success": False,
        },
    )


def _stan_file() -> Path:
    return Path(__file__).resolve().parents[2] / "stan" / "aceb_hierarchical.stan"


def _extract_theta_draws(fit: Any, n_items: int) -> np.ndarray:
    try:
        draws = fit.stan_variable("theta")
        return np.asarray(draws, dtype=np.float64)
    except Exception:
        frame = fit.draws_pd(vars=[f"theta[{idx}]" for idx in range(1, n_items + 1)])
        return frame.to_numpy(dtype=np.float64)


def _diagnostics(fit: Any) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    try:
        summary = fit.summary()
        diagnostics["max_rhat"] = float(summary["R_hat"].max()) if "R_hat" in summary else np.nan
        diagnostics["min_ess_bulk"] = float(summary["ESS_bulk"].min()) if "ESS_bulk" in summary else np.nan
        diagnostics["min_ess_tail"] = float(summary["ESS_tail"].min()) if "ESS_tail" in summary else np.nan
    except Exception as exc:
        diagnostics["summary_error"] = str(exc)
    try:
        sampler_diagnostics = fit.diagnose()
        diagnostics["cmdstan_diagnose"] = sampler_diagnostics
        diagnostics["divergent_transitions"] = int(sampler_diagnostics.lower().count("divergent"))
    except Exception:
        diagnostics["divergent_transitions"] = np.nan
    return diagnostics


def full_bayes_hierarchical(
    data: IndexedData,
    *,
    chains: int = 2,
    iter_warmup: int = 250,
    iter_sampling: int = 250,
    seed: int = 0,
    fit_bias: bool = False,
    bias_prior_scale: float = 1.0,
    stan_file: str | Path | None = None,
    raise_on_skip: bool = False,
) -> MethodOutput:
    start = time.perf_counter()
    try:
        from cmdstanpy import CmdStanModel, cmdstan_path
    except ImportError:
        reason = "cmdstanpy is not installed. Install optional support with: pip install cmdstanpy"
        if raise_on_skip:
            raise FullBayesSkipped(reason)
        return skipped_full_bayes_output(data, reason)

    try:
        cmdstan_path()
    except Exception as exc:
        reason = f"CmdStan is unavailable: {exc}"
        if raise_on_skip:
            raise FullBayesSkipped(reason)
        return skipped_full_bayes_output(data, reason)

    stan_path = Path(stan_file) if stan_file is not None else _stan_file()
    try:
        model = CmdStanModel(stan_file=str(stan_path))
        fit = model.sample(
            data=build_stan_data(data, fit_bias=fit_bias, bias_prior_scale=bias_prior_scale),
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            show_progress=False,
        )
    except Exception as exc:
        reason = f"CmdStan sampling failed: {type(exc).__name__}: {exc}"
        if raise_on_skip:
            raise
        return skipped_full_bayes_output(data, reason)

    theta_draws = _extract_theta_draws(fit, data.n_items)
    mean = np.mean(theta_draws, axis=0)
    variance = np.maximum(np.var(theta_draws, axis=0, ddof=1), 1e-18)
    sd = np.sqrt(variance)
    z = np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0.0)
    lfsr = np.minimum(norm.cdf(z), norm.cdf(-z))
    posterior = pd.DataFrame(
        {
            "item_id": data.item_ids,
            "posterior_mean": mean,
            "posterior_var": variance,
            "posterior_sd": sd,
            "posterior_second_moment": mean * mean + variance,
            "lfsr": lfsr,
            "method_name": "full_bayes_hierarchical",
        }
    )
    diagnostics = {
        "full_bayes_available": True,
        "skipped": False,
        "success": True,
        "chains": int(chains),
        "iter_warmup": int(iter_warmup),
        "iter_sampling": int(iter_sampling),
        "fit_bias": bool(fit_bias),
        "runtime_seconds": float(time.perf_counter() - start),
    }
    diagnostics.update(_diagnostics(fit))
    try:
        diagnostics["posterior_covariance"] = np.cov(theta_draws, rowvar=False).tolist()
    except Exception:
        pass
    return MethodOutput(posterior, diagnostics)


__all__ = [
    "FullBayesSkipped",
    "build_stan_data",
    "full_bayes_hierarchical",
    "skipped_full_bayes_output",
]
