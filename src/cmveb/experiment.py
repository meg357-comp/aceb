from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cmveb.benchmarks.graph_synthetic import run_graph_synthetic_benchmark
from cmveb.benchmarks.helm_runner import run_helm_benchmark
from cmveb.benchmarks.openml_runner import run_openml_benchmark
from cmveb.reporting import write_run_summary


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Experiment config must be a mapping.")
    return loaded


def generate_run_id(prefix: str, seed: int | None = None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_seed{seed}" if seed is not None else ""
    return f"{prefix}_{stamp}{suffix}"


def resolve_config(config: dict[str, Any], *, experiment_type: str) -> dict[str, Any]:
    resolved = dict(config)
    seed = int(resolved.get("random_state", resolved.get("seed", 0)))
    resolved["random_state"] = seed
    resolved.setdefault("seed", seed)
    resolved.setdefault("experiment_type", experiment_type)
    resolved.setdefault("run_id", generate_run_id(experiment_type, seed))
    resolved.setdefault("strict", False)
    resolved.setdefault("overwrite", False)
    return resolved


def prepare_run_dir(output_root: str | Path, resolved_config: dict[str, Any]) -> Path:
    output_root = Path(output_root)
    run_dir = output_root / str(resolved_config["run_id"])
    if run_dir.exists():
        if not bool(resolved_config.get("overwrite", False)):
            raise FileExistsError(f"Run directory exists: {run_dir}. Set overwrite: true to replace metadata/results.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_reproducibility_files(
    run_dir: str | Path,
    *,
    original_config: dict[str, Any],
    resolved_config: dict[str, Any],
    config_path: str | Path | None = None,
) -> None:
    run_dir = Path(run_dir)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(original_config, sort_keys=True), encoding="utf-8")
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved_config, sort_keys=True), encoding="utf-8")
    (run_dir / "git_commit.txt").write_text(_git_commit(), encoding="utf-8")
    (run_dir / "package_versions.txt").write_text(_package_versions(), encoding="utf-8")
    seeds = {
        "random_state": int(resolved_config.get("random_state", 0)),
        "numpy_bit_generator": "default_rng",
        "config_path": str(config_path) if config_path is not None else None,
    }
    (run_dir / "random_seeds.json").write_text(json.dumps(seeds, indent=2), encoding="utf-8")


def _git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    except Exception as exc:
        return f"unavailable: {exc}\n"
    return (result.stdout.strip() or result.stderr.strip() or "unavailable") + "\n"


def _package_versions() -> str:
    packages = ["numpy", "scipy", "pandas", "pyarrow", "pyyaml", "pytest", "calibrated-multiview-eb"]
    lines = [f"python=={sys.version.split()[0]}"]
    for package in packages:
        try:
            lines.append(f"{package}=={version(package)}")
        except PackageNotFoundError:
            lines.append(f"{package}==not-installed")
    return "\n".join(lines) + "\n"


def validate_config(config: dict[str, Any], *, experiment_type: str) -> None:
    if experiment_type not in {"synthetic_graph", "openml", "helm"}:
        raise ValueError("experiment_type must be synthetic_graph, openml, or helm.")
    if "random_state" in config and not isinstance(config["random_state"], int):
        raise TypeError("random_state must be an integer.")
    if experiment_type == "helm" and "input_path" not in config:
        raise ValueError("HELM config requires input_path.")
    if experiment_type == "helm":
        for key in ["require_higher_is_better", "allow_view_mean_truth"]:
            if key in config and not isinstance(config[key], bool):
                raise TypeError(f"{key} must be boolean.")
        mode = config.get("experiment_mode", "model_score_mode")
        if mode not in {"model_score_mode", "pairwise_gap_mode"}:
            raise ValueError("HELM experiment_mode must be model_score_mode or pairwise_gap_mode.")
    if experiment_type == "openml":
        mode = config.get("mode", "toy")
        if mode not in {"toy", "openml", "local_real", "sklearn_real"}:
            raise ValueError("OpenML mode must be toy, openml, local_real, or sklearn_real.")


def run_experiment(
    *,
    experiment_type: str,
    config_path: str | Path,
    output_root: str | Path,
    overrides: dict[str, Any] | None = None,
) -> Path:
    original = load_yaml_config(config_path)
    config = dict(original)
    if overrides:
        config.update(overrides)
    validate_config(config, experiment_type=experiment_type)
    resolved = resolve_config(config, experiment_type=experiment_type)
    np.random.seed(int(resolved["random_state"]))
    run_dir = prepare_run_dir(output_root, resolved)
    write_reproducibility_files(run_dir, original_config=original, resolved_config=resolved, config_path=config_path)
    strict = bool(resolved.get("strict", False))
    if experiment_type == "synthetic_graph":
        run_graph_synthetic_benchmark(
            output_dir=run_dir,
            graph_designs=list(resolved.get("graph_designs", ["star", "star_plus_edge_closes_triangle"])),
            n_items_values=list(resolved.get("n_items_values", [40])),
            n_views_values=list(resolved.get("n_views_values", [5])),
            tau_ratios=list(resolved.get("tau_ratios", [0.2])),
            loading_spreads=list(resolved.get("loading_spreads", [0.2])),
            missingness_modes=list(resolved.get("missingness_modes", ["none"])),
            theta_distributions=list(resolved.get("theta_distributions", ["normal"])),
            seed=int(resolved["random_state"]),
            seeds=list(resolved.get("seeds", [int(resolved["random_state"])])),
            max_iter=int(resolved.get("max_iter", 20)),
            include_bootstrap=bool(resolved.get("include_bootstrap", False)),
            include_mash=bool(resolved.get("include_mash", False)),
            strict=strict,
            method_names=list(resolved["methods"]) if "methods" in resolved else None,
            include_oracles=bool(resolved.get("include_oracles", True)),
        )
    elif experiment_type == "openml":
        run_openml_benchmark(output_dir=run_dir, config={**resolved, "strict": strict})
    elif experiment_type == "helm":
        run_helm_benchmark(output_dir=run_dir, config={**resolved, "strict": strict})
    _ensure_standard_artifacts(run_dir)
    write_run_summary(run_dir)
    return run_dir


def _ensure_standard_artifacts(run_dir: Path) -> None:
    for name in ["pairwise_calibration.csv", "stratified_coverage.csv", "metrics.csv"]:
        path = run_dir / name
        if not path.exists():
            path.write_text("\n", encoding="utf-8")
    for name in ["method_diagnostics.json", "graph_diagnostics.json"]:
        path = run_dir / name
        if not path.exists():
            path.write_text("{}\n", encoding="utf-8")
    (run_dir / "plots").mkdir(exist_ok=True)


__all__ = [
    "generate_run_id",
    "load_yaml_config",
    "prepare_run_dir",
    "resolve_config",
    "run_experiment",
    "validate_config",
    "write_reproducibility_files",
]
