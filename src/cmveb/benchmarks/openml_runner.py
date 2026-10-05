from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from cmveb.benchmarks.graph_synthetic import write_graph_plots
from cmveb.benchmarks.synthetic_runner import (
    _item_view_counts_and_components,
    compute_metrics,
    evaluate_pairwise_calibration,
    evaluate_stratified_coverage,
    skipped_metric_row,
)
from cmveb.datasets.openml_adapter import (
    DEFAULT_ALGORITHMS,
    OpenMLBuildConfig,
    build_openml_tables,
    load_openml_views,
    openml_available,
    openml_install_message,
)
from cmveb.evaluate import evaluate_against_noisy_target
from cmveb.graph_diagnostics import build_coobservation_graph
from cmveb.full_bayes import full_bayes_hierarchical
from cmveb.mash_baseline import mashr_multivariate_eb
from cmveb.models.baseline import (
    MethodOutput,
    adaptive_shrinkage_like,
    anchored_calibrated_eb,
    bootstrap_inverse_variance_all_views,
    feature_normal_eb_no_view_scale,
    inverse_variance_all_views,
    normal_eb_no_features,
    posthoc_inverse_variance_all_views,
    random_effects_meta_analysis,
    raw_single_reference,
)
from cmveb.models.joint_eb import joint_eb_full
from cmveb.schemas import IndexedData


def anchored_calibrated_eb_in_sample(data: IndexedData, *, max_iter: int = 200) -> MethodOutput:
    output = anchored_calibrated_eb(data, max_iter=max_iter, residual_calibration="in_sample")
    output.posterior["method_name"] = "anchored_calibrated_eb_in_sample"
    output.diagnostics["residual_calibration_method"] = "in_sample"
    return output


OPENML_METHODS: dict[str, Callable[..., MethodOutput]] = {
    "raw_reference": raw_single_reference,
    "inverse_variance_all_views": inverse_variance_all_views,
    "normal_eb_no_features": normal_eb_no_features,
    "feature_normal_eb_no_view_scale": feature_normal_eb_no_view_scale,
    "joint_eb_full": joint_eb_full,
    "anchored_calibrated_eb": anchored_calibrated_eb,
    "anchored_calibrated_eb_in_sample": anchored_calibrated_eb_in_sample,
    "random_effects_meta_analysis": random_effects_meta_analysis,
    "adaptive_shrinkage_like": adaptive_shrinkage_like,
    "posthoc_inverse_variance_all_views": posthoc_inverse_variance_all_views,
    "mashr_multivariate_eb": mashr_multivariate_eb,
    "full_bayes_hierarchical": full_bayes_hierarchical,
}

ITER_METHODS = {
    "normal_eb_no_features",
    "feature_normal_eb_no_view_scale",
    "joint_eb_full",
    "anchored_calibrated_eb",
    "anchored_calibrated_eb_in_sample",
    "adaptive_shrinkage_like",
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="list")
    return value


def truth_for_evaluation(truth: pd.DataFrame) -> pd.DataFrame:
    if "target_mean" in truth.columns:
        target_column = "target_mean"
    elif "target_estimate" in truth.columns:
        target_column = "target_estimate"
    else:
        raise ValueError("truth must contain target_mean or target_estimate.")
    return truth.loc[:, ["item_id", target_column]].rename(columns={target_column: "theta"})


def _truth_with_target_uncertainty(truth: pd.DataFrame) -> pd.DataFrame:
    if "target_estimate" in truth.columns:
        target_column = "target_estimate"
    elif "target_mean" in truth.columns:
        target_column = "target_mean"
    else:
        raise ValueError("truth must contain target_estimate or target_mean.")
    columns = ["item_id", target_column]
    rename = {target_column: "target_estimate"}
    if "target_standard_error" in truth.columns:
        columns.append("target_standard_error")
    elif "target_se" in truth.columns:
        columns.append("target_se")
        rename["target_se"] = "target_standard_error"
    out = truth.loc[:, columns].rename(columns=rename)
    if "target_standard_error" not in out.columns:
        out["target_standard_error"] = 0.0
    return out


def _add_target_aware_metrics(row: dict[str, Any], posterior: pd.DataFrame, truth: pd.DataFrame, *, enabled: bool) -> None:
    if not enabled:
        row.update(
            {
                "coverage_90_target_aware": np.nan,
                "z_score_sd_target_aware": np.nan,
                "interval_score_90_target_aware": np.nan,
                "target_se_mean": np.nan,
                "target_se_median": np.nan,
                "target_aware_metrics_used": False,
            }
        )
        return
    if not ({"target_standard_error", "target_se"} & set(truth.columns)):
        row.update(
            {
                "coverage_90_target_aware": np.nan,
                "z_score_sd_target_aware": np.nan,
                "interval_score_90_target_aware": np.nan,
                "target_se_mean": np.nan,
                "target_se_median": np.nan,
                "target_aware_metrics_used": False,
            }
        )
        return
    target_truth = _truth_with_target_uncertainty(truth)
    metrics = evaluate_against_noisy_target(posterior, target_truth, level=0.90)
    row["coverage_90_target_aware"] = metrics["coverage_90_target_aware"]
    row["z_score_sd"] = metrics["z_score_sd"]
    row["z_score_sd_target_aware"] = metrics["z_score_sd_target_aware"]
    row["interval_score_90"] = metrics["interval_score_90_naive"]
    row["interval_score_90_target_aware"] = metrics["interval_score_90_target_aware"]
    row["rmse_to_target_estimate"] = metrics["rmse_to_target_estimate"]
    row["normalized_rmse_target_aware"] = metrics["normalized_rmse_target_aware"]
    row["target_se_mean"] = metrics["target_se_mean"]
    row["target_se_median"] = metrics["target_se_median"]
    row["target_aware_metrics_used"] = metrics["target_aware_metrics_used"]


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("OpenML config must be a mapping.")
    return loaded


def build_or_load_tables(config: dict[str, Any], output_dir: Path) -> Path:
    data_dir = Path(config.get("data_dir", output_dir / "data"))
    if config.get("reuse_cached", True) and (data_dir / "observations.parquet").exists():
        return data_dir
    build_cfg = OpenMLBuildConfig(
        mode=str(config.get("mode", "toy")),
        graph_design=str(config.get("graph_design", "star_plus_edge")),
        n_datasets=int(config.get("n_datasets", 6)),
        n_views=int(config.get("n_views", 5)),
        metric=str(config.get("metric", "accuracy")),
        metrics=tuple(config.get("metrics", [config.get("metric", "accuracy")])),
        transform=str(config.get("transform", "raw")),
        random_state=int(config.get("random_state", 0)),
        min_standard_error=float(config.get("min_standard_error", 0.015)),
        n_test_min=int(config.get("n_test_min", 80)),
        n_test_max=int(config.get("n_test_max", 240)),
        er_density=float(config.get("er_density", 0.35)),
        seeds=tuple(int(seed) for seed in config.get("seeds", [0, 1, 2, 3, 4])),
        n_folds=int(config.get("n_folds", config.get("folds", 5))),
        datasets=tuple(config.get("datasets", ["breast_cancer", "wine", "digits"])),
        digits_max_samples=int(config.get("digits_max_samples", 600)),
        strict_algorithm_failures=bool(config.get("strict_algorithm_failures", config.get("strict", False))),
    )
    algorithms = list(config.get("algorithms", DEFAULT_ALGORITHMS))
    try:
        build_openml_tables(data_dir, config=build_cfg, algorithms=algorithms)
    except RuntimeError as exc:
        print(str(exc))
        print(openml_install_message())
        raise
    return data_dir


def run_methods(
    data: IndexedData,
    truth_eval: pd.DataFrame,
    *,
    method_names: list[str],
    max_iter: int,
    include_bootstrap: bool,
    bootstrap_B: int,
    strict: bool = False,
) -> list[tuple[str, pd.DataFrame, dict[str, object], float]]:
    methods = dict(OPENML_METHODS)
    if include_bootstrap:
        methods["bootstrap_inverse_variance_all_views"] = bootstrap_inverse_variance_all_views
    outputs = []
    for name in method_names:
        if name not in methods:
            raise ValueError(f"Unknown OpenML method: {name}")
        method = methods[name]
        start = time.perf_counter()
        try:
            if name == "posthoc_inverse_variance_all_views":
                output = method(data, truth=truth_eval)
            elif name == "full_bayes_hierarchical":
                output = method(data, iter_warmup=max(25, max_iter * 5), iter_sampling=max(25, max_iter * 5))
            elif name == "bootstrap_inverse_variance_all_views":
                output = method(data, B=bootstrap_B)
            elif name in ITER_METHODS:
                output = method(data, max_iter=max_iter)
            else:
                output = method(data)
        except Exception as exc:
            if strict:
                raise
            output = _failed_output(data, name, exc)
        outputs.append((name, output.posterior, output.diagnostics, time.perf_counter() - start))
    return outputs


def _copy_prepared_tables(data_dir: Path, output_dir: Path) -> None:
    for name in ["observations.parquet", "truth.parquet", "view_metadata.parquet", "item_features.parquet"]:
        src = data_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def _graph_edges_frame(graph: object, data: IndexedData) -> pd.DataFrame:
    rows = []
    for edge in graph.edge_moments:
        rows.append(
            {
                "view_j": int(edge.view_j),
                "view_k": int(edge.view_k),
                "view_id_j": str(data.view_ids[int(edge.view_j)]),
                "view_id_k": str(data.view_ids[int(edge.view_k)]),
                "n_shared": int(edge.n_shared),
                "edge_moment": float(edge.edge_moment),
                "is_positive": bool(edge.is_positive),
            }
        )
    return pd.DataFrame(rows)


def _item_metadata(item_ids: np.ndarray) -> pd.DataFrame:
    rows = []
    for item_id in item_ids.astype(str):
        parts = item_id.split("__")
        rows.append(
            {
                "item_id": item_id,
                "algorithm_id": parts[0] if len(parts) > 0 else "",
                "dataset_id": parts[1] if len(parts) > 1 else "",
                "metric_name": parts[2] if len(parts) > 2 else "",
            }
        )
    return pd.DataFrame(rows)


def _naive_item_estimates(data: IndexedData) -> pd.DataFrame:
    rows = []
    for item_idx, item_id in enumerate(data.item_ids):
        mask = data.item_idx == item_idx
        rows.append(
            {
                "item_id": str(item_id),
                "naive_estimate": float(np.mean(data.estimate[mask])) if np.any(mask) else np.nan,
                "n_observed_views": int(np.sum(mask)),
            }
        )
    return pd.DataFrame(rows)


def leaderboard_outputs(posterior_by_method: dict[str, pd.DataFrame], data: IndexedData) -> pd.DataFrame:
    rows = []
    meta = _item_metadata(data.item_ids)
    naive = _naive_item_estimates(data)
    for method_name, posterior in posterior_by_method.items():
        frame = posterior.merge(meta, on="item_id", how="left").merge(naive, on="item_id", how="left")
        sd = np.maximum(frame["posterior_sd"].to_numpy(dtype=np.float64), 1e-12)
        frame["lower_90"] = frame["posterior_mean"] - 1.6448536269514722 * sd
        frame["upper_90"] = frame["posterior_mean"] + 1.6448536269514722 * sd
        frame["method_name"] = method_name
        frame["rank"] = frame.groupby(["dataset_id", "metric_name"])["posterior_mean"].rank(ascending=False, method="min")
        rows.append(
            frame.loc[
                :,
                [
                    "method_name",
                    "dataset_id",
                    "metric_name",
                    "algorithm_id",
                    "item_id",
                    "rank",
                    "naive_estimate",
                    "posterior_mean",
                    "lower_90",
                    "upper_90",
                    "n_observed_views",
                ],
            ]
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _failed_output(data: IndexedData, method_name: str, exc: Exception) -> MethodOutput:
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
    return MethodOutput(posterior, {"failed": True, "failure_message": f"{type(exc).__name__}: {exc}", "success": False})


def run_openml_benchmark(
    *,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    config = config or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = build_or_load_tables(config, output_dir)
    data, truth, view_metadata = load_openml_views(data_dir)
    truth_eval = truth_for_evaluation(truth)
    graph = build_coobservation_graph(data, min_shared_items=int(config.get("min_shared_items", 1)))
    graph_summary = graph.to_summary_dict()
    method_names = list(config.get("methods", OPENML_METHODS.keys()))
    if not bool(config.get("include_mash", True)) and "mashr_multivariate_eb" in method_names:
        method_names.remove("mashr_multivariate_eb")
    include_bootstrap = bool(config.get("include_bootstrap", False))
    metrics_rows: list[dict[str, Any]] = []
    pairwise_rows: list[pd.DataFrame] = []
    stratified_rows: list[pd.DataFrame] = []
    method_diagnostics: dict[str, Any] = {}
    posterior_by_method: dict[str, pd.DataFrame] = {}
    for method_name, posterior, diagnostics, runtime in run_methods(
        data,
        truth_eval,
        method_names=method_names,
        max_iter=int(config.get("max_iter", 60)),
        include_bootstrap=include_bootstrap,
        bootstrap_B=int(config.get("bootstrap_B", 5)),
        strict=bool(config.get("strict", False)),
    ):
        method_diagnostics[method_name] = diagnostics
        common = {
            "run_id": str(config.get("run_id", output_dir.name)),
            "graph_design": str(config.get("graph_design", "unknown")),
            "delta_G": graph_summary["rank_deficiency_delta"],
            "has_odd_cycle": graph_summary["anchor_component_is_non_bipartite"],
            "n_graph_edges": graph_summary["n_edges"],
            "openml_available": openml_available(),
        }
        if diagnostics.get("skipped", False) or diagnostics.get("failed", False):
            row = skipped_metric_row(
                data=data,
                truth=truth_eval,
                diagnostics=diagnostics,
                size=str(data.n_items),
                scenario=str(config.get("graph_design", "openml")),
                method_name=method_name,
                runtime_seconds=runtime,
            )
            row.update(common)
            metrics_rows.append(row)
            continue
        posterior_by_method[method_name] = posterior.copy()
        pairwise_curve, pairwise_ece = evaluate_pairwise_calibration(
            posterior,
            truth_eval,
            bins=int(config.get("pairwise_bins", 10)),
            max_pairs=int(config.get("max_pairs", 20000)),
            random_state=int(config.get("random_state", 0)),
        )
        pairwise_curve.insert(0, "method_name", method_name)
        pairwise_curve.insert(0, "run_id", common["run_id"])
        pairwise_curve.insert(0, "graph_design", common["graph_design"])
        pairwise_curve["pairwise_ece"] = pairwise_ece
        pairwise_rows.append(pairwise_curve)
        view_count, component = _item_view_counts_and_components(data, diagnostics)
        stratified = evaluate_stratified_coverage(
            posterior,
            truth_eval,
            view_count=view_count,
            graph_component=component,
        )
        stratified.insert(0, "method_name", method_name)
        stratified.insert(0, "run_id", common["run_id"])
        stratified.insert(0, "graph_design", common["graph_design"])
        stratified_rows.append(stratified)
        row = compute_metrics(
            data=data,
            truth=truth_eval,
            posterior=posterior,
            diagnostics=diagnostics,
            size=str(data.n_items),
            scenario=str(config.get("graph_design", "openml")),
            method_name=method_name,
            runtime_seconds=runtime,
        )
        row.update(common)
        row["pairwise_ece"] = pairwise_ece
        _add_target_aware_metrics(
            row,
            posterior,
            truth,
            enabled=bool(config.get("real_data_target_uncertainty", True)),
        )
        metrics_rows.append(row)

    metrics = pd.DataFrame(metrics_rows)
    pairwise = pd.concat(pairwise_rows, ignore_index=True) if pairwise_rows else pd.DataFrame()
    stratified = pd.concat(stratified_rows, ignore_index=True) if stratified_rows else pd.DataFrame()
    _copy_prepared_tables(data_dir, output_dir)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pairwise.to_csv(output_dir / "pairwise_calibration.csv", index=False)
    stratified.to_csv(output_dir / "stratified_coverage.csv", index=False)
    leaderboard = leaderboard_outputs(posterior_by_method, data)
    leaderboard.to_csv(output_dir / "leaderboard_outputs.csv", index=False)
    _graph_edges_frame(graph, data).to_csv(output_dir / "graph_edges.csv", index=False)
    (output_dir / "graph_diagnostics.json").write_text(json.dumps(_json_ready(graph_summary), indent=2), encoding="utf-8")
    diagnostics_payload = {
        "method_diagnostics": method_diagnostics,
        "view_metadata": view_metadata.to_dict(orient="records"),
        "truth_columns": truth.columns.tolist(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "openml_available": openml_available(),
        },
        "config": config,
    }
    (output_dir / "method_diagnostics.json").write_text(
        json.dumps(_json_ready(diagnostics_payload), indent=2),
        encoding="utf-8",
    )
    write_graph_plots(metrics, pairwise, output_dir)
    summary_lines = [
        "# OpenML-Style Results Summary",
        "",
        f"Run ID: `{config.get('run_id', output_dir.name)}`",
        f"Mode: `{config.get('mode', 'toy')}`",
        f"Items: {data.n_items}",
        f"Views: {data.n_views}",
        f"Rows: {data.n_rows}",
        f"Methods: {', '.join(metrics['method_name'].astype(str).tolist()) if not metrics.empty else 'none'}",
        f"Graph edges: {graph_summary.get('n_edges', 0)}",
        f"Rank deficiency delta_G: {graph_summary.get('rank_deficiency_delta', np.nan)}",
        "",
    ]
    run_metadata_path = data_dir / "run_metadata.json"
    if run_metadata_path.exists():
        try:
            metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
            failures = metadata.get("algorithm_failures", [])
            summary_lines.append(f"Algorithm failures recorded: {len(failures)}")
        except json.JSONDecodeError:
            pass
    (output_dir / "OPENML_RESULTS_SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return metrics


def run_openml(*args, **kwargs):
    return run_openml_benchmark(*args, **kwargs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run OpenML-style ACEB benchmark.")
    parser.add_argument("--config", type=Path, default=Path("configs/openml_small.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/openml/local_toy"))
    args = parser.parse_args(argv)
    config = load_config(args.config) if args.config.exists() else {}
    metrics = run_openml_benchmark(output_dir=args.output_dir, config=config)
    print(f"Wrote {len(metrics)} OpenML-style metric rows to {args.output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
