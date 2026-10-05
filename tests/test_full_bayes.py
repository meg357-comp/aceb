from __future__ import annotations

import sys

import numpy as np
import pytest

from cmveb.full_bayes import build_stan_data, full_bayes_hierarchical
from tests.test_baselines import make_tiny_data


def test_missing_cmdstanpy_skips_gracefully(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "cmdstanpy", None)
    data = make_tiny_data()
    output = full_bayes_hierarchical(data)
    assert output.diagnostics["skipped"] is True
    assert output.diagnostics["full_bayes_available"] is False
    assert output.posterior.shape[0] == data.n_items


def test_stan_data_builder_creates_correct_dimensions() -> None:
    data = make_tiny_data()
    stan_data = build_stan_data(data, fit_bias=True, bias_prior_scale=0.5)
    assert stan_data["N"] == data.n_items
    assert stan_data["R"] == data.n_rows
    assert stan_data["J"] == data.n_views
    assert len(stan_data["item_idx"]) == data.n_rows
    assert len(stan_data["view_idx"]) == data.n_rows
    assert min(stan_data["item_idx"]) == 1
    assert max(stan_data["item_idx"]) == data.n_items
    assert stan_data["ref_view"] == 1
    assert stan_data["fit_bias"] == 1
    assert stan_data["bias_prior_scale"] == 0.5


def test_tiny_cmdstan_model_runs_if_available() -> None:
    try:
        from cmdstanpy import cmdstan_path  # noqa: F401

        cmdstan_path()
    except Exception as exc:
        pytest.skip(f"cmdstanpy/CmdStan unavailable: {exc}")
    data = make_tiny_data()
    output = full_bayes_hierarchical(data, chains=1, iter_warmup=5, iter_sampling=5, seed=123)
    if output.diagnostics.get("skipped", False):
        pytest.skip(str(output.diagnostics.get("skip_reason", "full Bayes skipped")))
    assert output.posterior.shape[0] == data.n_items
    assert np.all(output.posterior["posterior_var"].to_numpy(dtype=float) > 0.0)
