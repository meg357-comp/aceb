from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from cmveb import experiment
from cmveb.benchmarks import graph_synthetic
from cmveb.models.baseline import MethodOutput


EXPECTED_RUN_FILES = [
    "config.yaml",
    "resolved_config.yaml",
    "git_commit.txt",
    "package_versions.txt",
    "random_seeds.json",
    "metrics.csv",
    "pairwise_calibration.csv",
    "stratified_coverage.csv",
    "method_diagnostics.json",
    "graph_diagnostics.json",
    "summary.md",
]


def write_tiny_graph_config(path: Path, *, run_id: str = "tiny_graph") -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "run_id": run_id,
                "random_state": 123,
                "overwrite": True,
                "strict": False,
                "graph_designs": ["star"],
                "n_items_values": [24],
                "n_views_values": [5],
                "tau_ratios": [0.2],
                "loading_spreads": [0.2],
                "missingness_modes": ["none"],
                "theta_distributions": ["normal"],
                "max_iter": 2,
                "include_mash": False,
                "include_bootstrap": False,
            }
        ),
        encoding="utf-8",
    )


def test_tiny_synthetic_run_writes_all_expected_files(tmp_path) -> None:
    config_path = tmp_path / "synthetic_graph.yaml"
    write_tiny_graph_config(config_path)
    run_dir = experiment.run_experiment(
        experiment_type="synthetic_graph",
        config_path=config_path,
        output_root=tmp_path / "runs",
    )
    for name in EXPECTED_RUN_FILES:
        assert (run_dir / name).exists(), name
    assert any((run_dir / "plots").glob("*.svg"))


def test_failed_optional_method_is_recorded_but_does_not_crash(monkeypatch, tmp_path) -> None:
    def broken_method(data, *args, **kwargs):
        raise RuntimeError("intentional test failure")

    monkeypatch.setattr(graph_synthetic, "BASE_METHODS", {"broken_optional": broken_method})
    config_path = tmp_path / "synthetic_graph.yaml"
    write_tiny_graph_config(config_path, run_id="failure_graph")
    run_dir = experiment.run_experiment(
        experiment_type="synthetic_graph",
        config_path=config_path,
        output_root=tmp_path / "runs",
    )
    metrics = pd.read_csv(run_dir / "metrics.csv")
    failed = metrics.loc[metrics["method_name"] == "broken_optional"]
    assert not failed.empty
    assert bool(failed["failed"].iloc[0]) is True
    assert "intentional test failure" in str(failed["failure_message"].iloc[0])


def test_configs_validate() -> None:
    for config_path, experiment_type in [
        (Path("configs/synthetic_graph_small.yaml"), "synthetic_graph"),
        (Path("configs/synthetic_graph_medium.yaml"), "synthetic_graph"),
        (Path("configs/openml_small.yaml"), "openml"),
        (Path("configs/openml_local_real.yaml"), "openml"),
        (Path("configs/helm_small.yaml"), "helm"),
        (Path("configs/helm_main_paper.yaml"), "helm"),
    ]:
        config = experiment.load_yaml_config(config_path)
        experiment.validate_config(config, experiment_type=experiment_type)


def test_summary_includes_key_metrics(tmp_path) -> None:
    config_path = tmp_path / "synthetic_graph.yaml"
    write_tiny_graph_config(config_path, run_id="summary_graph")
    run_dir = experiment.run_experiment(
        experiment_type="synthetic_graph",
        config_path=config_path,
        output_root=tmp_path / "runs",
    )
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "Best RMSE" in summary
    assert "Closest 90% Coverage" in summary
