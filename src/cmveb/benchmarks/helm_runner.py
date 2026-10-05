from __future__ import annotations

import argparse
import json
import shutil
import signal
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from cmveb.benchmarks.graph_synthetic import write_graph_plots
from cmveb.benchmarks.synthetic_runner import (
    _item_view_counts_and_components,
    compute_metrics,
    evaluate_pairwise_calibration,
    evaluate_stratified_coverage,
    skipped_metric_row,
)
from cmveb.datasets.helm_adapter import HelmAdapterConfig, build_helm_tables, load_helm_views
from cmveb.datasets.helm_adapter import GRAPH_DESIGN_ALIASES, read_helm_normalized
from cmveb.evaluate import evaluate_against_noisy_target, evaluate_selective_sign_decisions
from cmveb.graph_diagnostics import build_coobservation_graph
from cmveb.full_bayes import full_bayes_hierarchical
from cmveb.mash_baseline import mashr_multivariate_eb
from cmveb.models.baseline import (
    MethodOutput,
    adaptive_shrinkage_like,
    anchored_calibrated_eb,
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


HELM_METHODS: dict[str, Callable[..., MethodOutput]] = {
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


def _target_decidability_summary(truth: pd.DataFrame) -> pd.DataFrame:
    if "target_estimate" in truth.columns:
        target = truth["target_estimate"].to_numpy(dtype=np.float64)
    elif "target_mean" in truth.columns:
        target = truth["target_mean"].to_numpy(dtype=np.float64)
    else:
        raise ValueError("truth must contain target_estimate or target_mean.")
    if "target_standard_error" in truth.columns:
        target_se = truth["target_standard_error"].fillna(0.0).to_numpy(dtype=np.float64)
    elif "target_se" in truth.columns:
        target_se = truth["target_se"].fillna(0.0).to_numpy(dtype=np.float64)
    else:
        target_se = np.zeros(len(truth), dtype=np.float64)
    target_se = np.maximum(target_se, 0.0)
    target_z = np.divide(
        np.abs(target),
        target_se,
        out=np.where(np.abs(target) > 0.0, np.inf, 0.0),
        where=target_se > 0.0,
    )
    subsets = {
        "all_pairs": np.ones(len(target_z), dtype=bool),
        "decidable_pairs_90": target_z >= 1.645,
        "decidable_pairs_95": target_z >= 1.96,
        "ambiguous_pairs_90": target_z < 1.645,
    }
    rows: list[dict[str, Any]] = []
    total = len(target_z)
    for subset, mask in subsets.items():
        z_sub = target_z[mask]
        rows.append(
            {
                "subset": subset,
                "n_items": int(mask.sum()),
                "fraction": float(mask.mean()) if total else np.nan,
                "target_z_median": float(np.median(z_sub)) if len(z_sub) else np.nan,
                "target_z_min": float(np.min(z_sub)) if len(z_sub) else np.nan,
                "target_z_max": float(np.max(z_sub)) if len(z_sub) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _add_target_aware_metrics(row: dict[str, Any], posterior: pd.DataFrame, truth: pd.DataFrame, *, enabled: bool) -> None:
    if not enabled or not ({"target_standard_error", "target_se"} & set(truth.columns)):
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


def _write_real_quality_gate(
    *,
    output_dir: Path,
    config: dict[str, Any],
    data: IndexedData,
    truth: pd.DataFrame,
    graph_summary: dict[str, Any],
) -> dict[str, Any]:
    input_path = config.get("input_path")
    normalized = read_helm_normalized(input_path) if input_path is not None and Path(input_path).exists() else pd.DataFrame()
    min_pairwise_items = int(config.get("quality_gate_min_pairwise_items", 100))
    min_views_per_item = float(config.get("min_views_per_item", 3))
    min_items_per_group = int(config.get("min_items_per_scenario_metric", 10))
    max_proxy_fraction = float(config.get("quality_gate_max_proxy_target_fraction", 0.50))
    max_floor_fraction = float(config.get("quality_gate_max_floor_se_fraction", 0.30))
    floor = float(config.get("standard_error_floor", 0.02))

    item_view_counts = pd.Series(data.item_idx).value_counts()
    avg_views = float(item_view_counts.mean()) if not item_view_counts.empty else 0.0
    target_source_counts = truth.get("target_source", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
    target_source_text = truth.get("target_source", pd.Series(dtype=str)).astype(str)
    aggregate_proxy_fraction = float(target_source_text.str.contains("aggregate_proxy", na=False).mean()) if len(target_source_text) else 0.0
    any_proxy_fraction = float(target_source_text.str.contains("proxy|view_mean", case=False, regex=True, na=False).mean()) if len(target_source_text) else 0.0
    if not normalized.empty and "standard_error" in normalized.columns:
        se = pd.to_numeric(normalized["standard_error"], errors="coerce")
        floor_limited_fraction = float((se <= floor + 1e-12).mean())
    else:
        floor_limited_fraction = np.nan
    if not normalized.empty and {"scenario_id", "metric_name", "model_id"}.issubset(normalized.columns):
        per_group_models = normalized.groupby(["scenario_id", "metric_name"])["model_id"].nunique()
        per_group_pairwise_items = (per_group_models * (per_group_models - 1) // 2).astype(int)
        min_group_items = int(per_group_pairwise_items.min()) if not per_group_pairwise_items.empty else 0
        scenario_metric_group_count = int(per_group_pairwise_items.shape[0])
    else:
        min_group_items = 0
        scenario_metric_group_count = 0

    checks = [
        {
            "check": "pairwise item count",
            "status": "pass" if data.n_items >= min_pairwise_items else "fail",
            "detail": f"{data.n_items} items; threshold {min_pairwise_items}",
        },
        {
            "check": "average observed views per item",
            "status": "pass" if avg_views >= min_views_per_item else "warn",
            "detail": f"{avg_views:.3f}; threshold {min_views_per_item}",
        },
        {
            "check": "minimum pairwise items per scenario/metric",
            "status": "pass" if min_group_items >= min_items_per_group else "warn",
            "detail": f"{min_group_items}; threshold {min_items_per_group}",
        },
        {
            "check": "aggregate proxy target fraction",
            "status": "pass" if aggregate_proxy_fraction <= max_proxy_fraction else "fail",
            "detail": f"{aggregate_proxy_fraction:.3f}; threshold {max_proxy_fraction}",
        },
        {
            "check": "floor-limited standard-error fraction",
            "status": "pass" if not np.isfinite(floor_limited_fraction) or floor_limited_fraction <= max_floor_fraction else "warn",
            "detail": f"{floor_limited_fraction:.3f}; threshold {max_floor_fraction}",
        },
        {
            "check": "useful graph edges",
            "status": "pass" if int(graph_summary.get("n_edges", 0)) > 0 else "fail",
            "detail": f"{graph_summary.get('n_edges', 0)} edges",
        },
        {
            "check": "all methods evaluated against proxy truth",
            "status": "pass" if any_proxy_fraction < 1.0 else "fail",
            "detail": f"{any_proxy_fraction:.3f} proxy/view-mean truth fraction",
        },
        {
            "check": "target-aware metrics enabled",
            "status": "pass" if bool(config.get("real_data_target_uncertainty", True)) else "warn",
            "detail": f"real_data_target_uncertainty={bool(config.get('real_data_target_uncertainty', True))}",
        },
    ]
    status_order = {"fail": 2, "warn": 1, "pass": 0}
    overall = "PASS"
    if any(row["status"] == "fail" for row in checks):
        overall = "FAIL"
    elif any(row["status"] == "warn" for row in checks):
        overall = "PASS_WITH_WARNINGS"
    payload = {
        "overall_status": overall,
        "n_items": int(data.n_items),
        "n_views": int(data.n_views),
        "n_rows": int(data.n_rows),
        "average_views_per_item": avg_views,
        "scenario_metric_group_count": scenario_metric_group_count,
        "min_pairwise_items_per_scenario_metric": min_group_items,
        "target_source_counts": target_source_counts,
        "aggregate_proxy_fraction": aggregate_proxy_fraction,
        "any_proxy_fraction": any_proxy_fraction,
        "floor_limited_standard_error_fraction": floor_limited_fraction,
        "graph_edges": int(graph_summary.get("n_edges", 0)),
        "delta_G": graph_summary.get("rank_deficiency_delta", np.nan),
        "checks": checks,
    }
    lines = [
        "# HELM Real Quality Gate",
        "",
        f"Overall status: **{overall}**",
        "",
        "| check | status | detail |",
        "| --- | --- | --- |",
    ]
    for row in checks:
        lines.append(f"| {row['check']} | `{row['status']}` | {row['detail']} |")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Pairwise-gap items: {data.n_items}",
            f"- Views: {data.n_views}",
            f"- Rows: {data.n_rows}",
            f"- Average observed views per item: {avg_views:.3f}",
            f"- Graph edges: {graph_summary.get('n_edges', 0)}",
            f"- Rank deficiency delta_G: {graph_summary.get('rank_deficiency_delta', np.nan)}",
            f"- Target source counts: `{target_source_counts}`",
            f"- Floor-limited SE fraction: {floor_limited_fraction:.3f}" if np.isfinite(floor_limited_fraction) else "- Floor-limited SE fraction: unavailable",
            "",
        ]
    )
    if overall == "FAIL":
        lines.append("These results should not be treated as paper evidence until failed checks are resolved.")
    elif overall == "PASS_WITH_WARNINGS":
        lines.append("These results are usable with caveats; warning checks should be disclosed or sensitivity-checked.")
    else:
        lines.append("The real-data quality gate passed.")
    (output_dir / "HELM_REAL_QUALITY_GATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "helm_real_quality_gate.json").write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    payload["has_failure"] = max(status_order[row["status"]] for row in checks) >= status_order["fail"]
    return payload


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("HELM config must be a mapping.")
    return loaded


def build_or_load_tables(config: dict[str, Any], output_dir: Path) -> Path:
    data_dir = Path(config.get("data_dir", output_dir / "data"))
    if config.get("reuse_cached", True) and (data_dir / "observations.parquet").exists():
        return data_dir
    input_path = config.get("input_path")
    if input_path is None:
        raise ValueError("HELM runner requires input_path pointing to normalized CSV/Parquet.")
    graph_design = config.get("graph_design")
    if graph_design in {"all_views", "observed"}:
        graph_design = None
    adapter_config = HelmAdapterConfig(
        experiment_mode=str(config.get("experiment_mode", "model_score_mode")),
        graph_design=graph_design,
        standard_error_floor=float(config.get("standard_error_floor", 0.02)),
        truth_source=str(config.get("truth_source", "target_or_view_mean")),
        er_density=float(config.get("er_density", 0.35)),
        random_state=int(config.get("random_state", 0)),
        require_higher_is_better=bool(config.get("require_higher_is_better", True)),
        allow_view_mean_truth=bool(config.get("allow_view_mean_truth", False))
        or not bool(config.get("require_target_estimate", True)),
        max_items=int(config["max_items"]) if config.get("max_items") is not None else None,
    )
    build_helm_tables(input_path, data_dir, config=adapter_config)
    return data_dir


def run_methods(
    data: IndexedData,
    truth_eval: pd.DataFrame,
    *,
    method_names: list[str],
    max_iter: int,
    method_timeout_seconds: float | None = None,
    method_timeout_seconds_by_method: dict[str, float] | None = None,
    skip_methods: set[str] | None = None,
    strict: bool = False,
) -> list[tuple[str, pd.DataFrame, dict[str, object], float]]:
    outputs = []
    skip_methods = skip_methods or set()
    method_timeout_seconds_by_method = method_timeout_seconds_by_method or {}
    for name in method_names:
        if name not in HELM_METHODS:
            raise ValueError(f"Unknown HELM method: {name}")
        method = HELM_METHODS[name]
        start = time.perf_counter()
        timeout = method_timeout_seconds_by_method.get(name, method_timeout_seconds)
        try:
            if name in skip_methods:
                raise RuntimeError("method skipped by configuration")
            if timeout is not None and timeout > 0:
                def _timeout_handler(signum, frame):  # noqa: ARG001
                    raise TimeoutError(f"method exceeded timeout of {timeout:g} seconds")

                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, float(timeout))
            if name == "posthoc_inverse_variance_all_views":
                output = method(data, truth=truth_eval)
            elif name == "full_bayes_hierarchical":
                output = method(data, iter_warmup=max(25, max_iter * 5), iter_sampling=max(25, max_iter * 5))
            elif name in ITER_METHODS:
                output = method(data, max_iter=max_iter)
            else:
                output = method(data)
        except Exception as exc:
            if strict:
                raise
            output = _failed_output(data, name, exc)
        finally:
            if timeout is not None and timeout > 0:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, old_handler)
        outputs.append((name, output.posterior, output.diagnostics, time.perf_counter() - start))
    return outputs


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
    diagnostics = {"failed": True, "failure_message": f"{type(exc).__name__}: {exc}", "success": False}
    if isinstance(exc, TimeoutError):
        diagnostics["status"] = "timeout"
        diagnostics["timeout"] = True
    return MethodOutput(posterior, diagnostics)


def _short_model_name(model_full: str) -> str:
    model = str(model_full).split("__deployment=", 1)[0]
    return model.rsplit("/", 1)[-1]


def parse_pairwise_item_id(item_id: str) -> dict[str, str]:
    item_id = str(item_id)
    if "__minus__" not in item_id:
        parts = item_id.split("__")
        return {
            "item_id": item_id,
            "model_a_full": parts[0] if parts else "",
            "model_b_full": "",
            "model_a_short": _short_model_name(parts[0]) if parts else "",
            "model_b_short": "",
            "scenario_metric": "__".join(parts[1:]) if len(parts) > 1 else "",
            "scenario_id": parts[1] if len(parts) > 1 else "",
            "metric_name": parts[2] if len(parts) > 2 else "",
        }
    model_a_full, right = item_id.split("__minus__", 1)
    right_parts = right.split("__")
    if len(right_parts) >= 2 and right_parts[1].startswith("deployment="):
        model_b_full = "__".join(right_parts[:2])
        suffix_parts = right_parts[2:]
    else:
        model_b_full = right_parts[0] if right_parts else ""
        suffix_parts = right_parts[1:]
    scenario_metric = "__".join(suffix_parts)
    metric_name = suffix_parts[-1] if suffix_parts else ""
    scenario_id = "__".join(suffix_parts[:-1]) if len(suffix_parts) > 1 else ""
    return {
        "item_id": item_id,
        "model_a_full": model_a_full,
        "model_b_full": model_b_full,
        "model_a_short": _short_model_name(model_a_full),
        "model_b_short": _short_model_name(model_b_full),
        "scenario_metric": scenario_metric,
        "scenario_id": scenario_id,
        "metric_name": metric_name,
    }


def _pairwise_metadata_frame(item_ids: pd.Series | np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([parse_pairwise_item_id(str(item_id)) for item_id in item_ids])


def _add_pairwise_leaderboard_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    parse_columns = [
        "model_a",
        "model_b",
        "model_a_full",
        "model_b_full",
        "model_a_short",
        "model_b_short",
        "scenario_metric",
        "scenario_id",
        "metric_name",
        "posterior_prob_model_a_better",
        "probability_positive_gap",
        "confidence",
        "predicted_winner",
        "predicted_winner_full",
        "predicted_winner_short",
    ]
    frame = frame.drop(columns=[column for column in parse_columns if column in frame.columns], errors="ignore")
    metadata = _pairwise_metadata_frame(frame["item_id"])
    frame = frame.merge(metadata, on="item_id", how="left")
    sd = np.maximum(frame["posterior_sd"].to_numpy(dtype=np.float64), 1e-12)
    mean = frame["posterior_mean"].to_numpy(dtype=np.float64)
    p_model_a = norm.cdf(mean / sd)
    frame["posterior_prob_model_a_better"] = p_model_a
    frame["probability_positive_gap"] = p_model_a
    frame["confidence"] = np.maximum(p_model_a, 1.0 - p_model_a)
    frame["predicted_winner_full"] = np.where(
        mean > 0.0,
        frame["model_a_full"],
        np.where(mean < 0.0, frame["model_b_full"], "tie"),
    )
    frame["predicted_winner_short"] = np.where(
        mean > 0.0,
        frame["model_a_short"],
        np.where(mean < 0.0, frame["model_b_short"], "tie"),
    )
    frame["model_a"] = frame["model_a_full"]
    frame["model_b"] = frame["model_b_full"]
    frame["predicted_winner"] = frame["predicted_winner_full"]
    return frame


def _validate_leaderboard_outputs(leaderboard: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    if leaderboard.empty:
        payload = {"n_rows": 0, "invalid_minus_rows": 0, "invalid_winner_rows": 0, "status": "empty"}
    elif "model_b_full" not in leaderboard.columns:
        payload = {"n_rows": int(len(leaderboard)), "invalid_minus_rows": np.nan, "invalid_winner_rows": np.nan, "status": "not_pairwise"}
    else:
        invalid_minus = (
            leaderboard["model_a_full"].astype(str).eq("minus")
            | leaderboard["model_b_full"].astype(str).eq("minus")
            | leaderboard.get("model_a", pd.Series("", index=leaderboard.index)).astype(str).eq("minus")
            | leaderboard.get("model_b", pd.Series("", index=leaderboard.index)).astype(str).eq("minus")
        )
        valid_winner = (
            leaderboard["predicted_winner_full"].astype(str).eq("tie")
            | leaderboard["predicted_winner_full"].astype(str).eq(leaderboard["model_a_full"].astype(str))
            | leaderboard["predicted_winner_full"].astype(str).eq(leaderboard["model_b_full"].astype(str))
        )
        payload = {
            "n_rows": int(len(leaderboard)),
            "invalid_minus_rows": int(invalid_minus.sum()),
            "invalid_winner_rows": int((~valid_winner).sum()),
            "status": "pass" if int(invalid_minus.sum()) == 0 and int((~valid_winner).sum()) == 0 else "fail",
        }
    lines = [
        "# Leaderboard Output Validation",
        "",
        f"Status: **{payload['status']}**",
        "",
        "| check | value |",
        "| --- | ---: |",
        f"| rows | {payload['n_rows']} |",
        f"| rows with model_a/model_b equal to `minus` | {payload['invalid_minus_rows']} |",
        f"| rows with invalid predicted winner | {payload['invalid_winner_rows']} |",
        "",
    ]
    if payload["status"] == "pass":
        lines.append("No invalid pairwise parsing rows were detected.")
    else:
        lines.append("One or more leaderboard rows failed pairwise parsing validation.")
    (output_dir / "LEADERBOARD_OUTPUT_VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "leaderboard_output_validation.json").write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    return payload


def leaderboard_outputs(
    posterior: pd.DataFrame,
    *,
    method_name: str,
    experiment_mode: str,
) -> pd.DataFrame:
    frame = posterior.loc[:, ["item_id", "posterior_mean", "posterior_sd", "lfsr"]].copy()
    frame.insert(0, "method_name", method_name)
    frame["rank"] = frame["posterior_mean"].rank(ascending=False, method="min")
    frame["experiment_mode"] = experiment_mode
    if experiment_mode == "pairwise_gap_mode":
        frame = _add_pairwise_leaderboard_columns(frame)
    return frame


def _copy_prepared_tables(data_dir: Path, output_dir: Path) -> None:
    for name in ["observations.parquet", "truth.parquet", "view_metadata.parquet", "item_features.parquet"]:
        src = data_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
            if name in {"observations.parquet", "truth.parquet", "view_metadata.parquet"}:
                pd.read_parquet(src).to_csv(output_dir / name.replace(".parquet", ".csv"), index=False)


def _graph_edges_frame(graph: Any, data: IndexedData) -> pd.DataFrame:
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


def _item_metadata(item_ids: np.ndarray, experiment_mode: str) -> pd.DataFrame:
    if experiment_mode == "pairwise_gap_mode":
        metadata = _pairwise_metadata_frame(item_ids.astype(str))
        metadata["model_a"] = metadata["model_a_full"]
        metadata["model_b"] = metadata["model_b_full"]
        return metadata
    records = []
    for item_id in item_ids.astype(str):
        parts = item_id.split("__")
        record = {"item_id": item_id}
        if experiment_mode != "pairwise_gap_mode":
            record["model_id"] = parts[0] if len(parts) > 0 else ""
            record["scenario_id"] = parts[1] if len(parts) > 1 else ""
            record["metric_name"] = parts[2] if len(parts) > 2 else ""
        records.append(record)
    return pd.DataFrame(records)


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


def _sign_calibration_ece(probability: np.ndarray, observed: np.ndarray, *, bins: int = 10) -> float:
    probability = np.asarray(probability, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    total = max(1, probability.size)
    for idx in range(bins):
        left, right = edges[idx], edges[idx + 1]
        mask = (probability >= left) & (probability <= right if idx == bins - 1 else probability < right)
        if not np.any(mask):
            continue
        ece += float(np.sum(mask) / total) * abs(float(np.mean(probability[mask])) - float(np.mean(observed[mask])))
    return float(ece)


def _pairwise_gap_report(
    posterior_by_method: dict[str, pd.DataFrame],
    truth_eval: pd.DataFrame,
    *,
    experiment_mode: str,
) -> pd.DataFrame:
    if experiment_mode != "pairwise_gap_mode":
        return pd.DataFrame()
    truth = truth_eval.set_index("item_id")["theta"]
    metadata = _item_metadata(truth.index.to_numpy(), experiment_mode)
    group_count = int(metadata[["scenario_id", "metric_name"]].drop_duplicates().shape[0]) if not metadata.empty else 0
    model_pair_count = int(metadata[["model_a", "model_b"]].drop_duplicates().shape[0]) if {"model_a", "model_b"}.issubset(metadata.columns) else 0
    rows = []
    for method_name, posterior in posterior_by_method.items():
        aligned = posterior.set_index("item_id").loc[truth.index]
        mean = aligned["posterior_mean"].to_numpy(dtype=np.float64)
        sd = np.maximum(aligned["posterior_sd"].to_numpy(dtype=np.float64), 1e-12)
        theta = truth.to_numpy(dtype=np.float64)
        prob_positive = norm.cdf(mean / sd)
        observed_positive = (theta > 0.0).astype(float)
        pred_positive = mean > 0.0
        rows.append(
            {
                "method_name": method_name,
                "false_sign_rate": float(np.mean(pred_positive != observed_positive.astype(bool))),
                "sign_accuracy": float(np.mean(pred_positive == observed_positive.astype(bool))),
                "sign_calibration_ece": _sign_calibration_ece(prob_positive, observed_positive),
                "pairwise_gap_brier_score": float(np.mean(np.square(prob_positive - observed_positive))),
                "model_pair_count": model_pair_count,
                "scenario_metric_group_count": group_count,
            }
        )
    return pd.DataFrame(rows)


def _grouped_leaderboard_outputs(
    posterior_by_method: dict[str, pd.DataFrame],
    data: IndexedData,
    *,
    experiment_mode: str,
    threshold: float = 0.95,
) -> pd.DataFrame:
    if not posterior_by_method:
        return pd.DataFrame()
    metadata = _item_metadata(data.item_ids, experiment_mode)
    naive = _naive_item_estimates(data)
    rows = []
    for method_name, posterior in posterior_by_method.items():
        frame = posterior.merge(metadata, on="item_id", how="left").merge(naive, on="item_id", how="left")
        sd = np.maximum(frame["posterior_sd"].to_numpy(dtype=np.float64), 1e-12)
        frame["lower_90"] = frame["posterior_mean"] - 1.6448536269514722 * sd
        frame["upper_90"] = frame["posterior_mean"] + 1.6448536269514722 * sd
        frame["method_name"] = method_name
        if experiment_mode == "pairwise_gap_mode":
            frame = _add_pairwise_leaderboard_columns(frame)
            frame["abstain"] = frame["confidence"] < threshold
            keep = [
                "method_name",
                "scenario_id",
                "metric_name",
                "scenario_metric",
                "model_a",
                "model_b",
                "model_a_full",
                "model_b_full",
                "model_a_short",
                "model_b_short",
                "item_id",
                "naive_estimate",
                "posterior_mean",
                "lower_90",
                "upper_90",
                "posterior_prob_model_a_better",
                "probability_positive_gap",
                "predicted_winner",
                "predicted_winner_full",
                "predicted_winner_short",
                "confidence",
                "abstain",
                "n_observed_views",
            ]
        else:
            frame["rank"] = frame.groupby(["scenario_id", "metric_name"])["posterior_mean"].rank(ascending=False, method="min")
            keep = [
                "method_name",
                "scenario_id",
                "metric_name",
                "model_id",
                "item_id",
                "rank",
                "naive_estimate",
                "posterior_mean",
                "lower_90",
                "upper_90",
                "n_observed_views",
            ]
        rows.append(frame.loc[:, [column for column in keep if column in frame.columns]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_helm_benchmark(
    *,
    output_dir: str | Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = build_or_load_tables(config, output_dir)
    data, truth, view_metadata = load_helm_views(data_dir)
    truth_eval = truth_for_evaluation(truth)
    graph = build_coobservation_graph(data, min_shared_items=int(config.get("min_shared_items", 1)))
    graph_summary = graph.to_summary_dict()
    quality_gate_payload: dict[str, Any] | None = None
    if bool(config.get("enable_real_quality_gate", False)):
        quality_gate_payload = _write_real_quality_gate(
            output_dir=output_dir,
            config=config,
            data=data,
            truth=truth,
            graph_summary=graph_summary,
        )
        if bool(config.get("quality_gate_fail_on_error", True)) and bool(quality_gate_payload.get("has_failure", False)):
            raise RuntimeError(f"HELM real quality gate failed; see {output_dir / 'HELM_REAL_QUALITY_GATE.md'}")
    experiment_mode = str(config.get("experiment_mode", "model_score_mode"))
    method_names = list(config.get("methods", HELM_METHODS.keys()))
    if not bool(config.get("include_mash", True)) and "mashr_multivariate_eb" in method_names:
        method_names.remove("mashr_multivariate_eb")
    metrics_rows: list[dict[str, Any]] = []
    pairwise_rows: list[pd.DataFrame] = []
    selective_rows: list[pd.DataFrame] = []
    stratified_rows: list[pd.DataFrame] = []
    leaderboard_rows: list[pd.DataFrame] = []
    method_diagnostics: dict[str, Any] = {}
    posterior_by_method: dict[str, pd.DataFrame] = {}
    for method_name, posterior, diagnostics, runtime in run_methods(
        data,
        truth_eval,
        method_names=method_names,
        max_iter=int(config.get("max_iter", 60)),
        method_timeout_seconds=config.get("method_timeout_seconds"),
        method_timeout_seconds_by_method={
            **({str(k): float(v) for k, v in dict(config.get("method_timeout_seconds_by_method", {})).items()}),
            **({"joint_eb_full": float(config["joint_eb_full_timeout_seconds"])} if config.get("joint_eb_full_timeout_seconds") is not None else {}),
        },
        skip_methods=set(config.get("skip_methods", [])),
        strict=bool(config.get("strict", False)),
    ):
        method_diagnostics[method_name] = diagnostics
        common = {
            "run_id": str(config.get("run_id", output_dir.name)),
            "graph_design": str(config.get("graph_design", "observed")),
            "experiment_mode": experiment_mode,
            "delta_G": graph_summary["rank_deficiency_delta"],
            "has_odd_cycle": graph_summary["anchor_component_is_non_bipartite"],
            "n_graph_edges": graph_summary["n_edges"],
        }
        if diagnostics.get("skipped", False) or diagnostics.get("failed", False):
            row = skipped_metric_row(
                data=data,
                truth=truth_eval,
                diagnostics=diagnostics,
                size=str(data.n_items),
                scenario=common["graph_design"],
                method_name=method_name,
                runtime_seconds=runtime,
            )
            row.update(common)
            if diagnostics.get("timeout", False) or str(diagnostics.get("failure_message", "")).startswith("TimeoutError:"):
                row["status"] = "timeout"
            elif diagnostics.get("skipped", False):
                row["status"] = "skipped"
            else:
                row["status"] = "failed"
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
        leaderboard_rows.append(leaderboard_outputs(posterior, method_name=method_name, experiment_mode=experiment_mode))
        row = compute_metrics(
            data=data,
            truth=truth_eval,
            posterior=posterior,
            diagnostics=diagnostics,
            size=str(data.n_items),
            scenario=common["graph_design"],
            method_name=method_name,
            runtime_seconds=runtime,
        )
        row.update(common)
        row["status"] = "success"
        row["pairwise_ece"] = pairwise_ece
        if experiment_mode == "pairwise_gap_mode":
            selective_curve, selective_summary = evaluate_selective_sign_decisions(posterior, truth)
            selective_curve.insert(0, "method_name", method_name)
            selective_curve.insert(0, "run_id", common["run_id"])
            selective_curve.insert(0, "graph_design", common["graph_design"])
            selective_rows.append(selective_curve)
            for key, value in selective_summary.iloc[0].to_dict().items():
                if key != "subset":
                    row[key] = value
            all_pairs_50 = selective_curve[
                (selective_curve["subset"] == "all_pairs")
                & np.isclose(selective_curve["threshold"].to_numpy(dtype=np.float64), 0.50)
            ]
            if all_pairs_50.empty:
                row["false_sign_rate"] = np.nan
                row["sign_accuracy"] = np.nan
            else:
                row["false_sign_rate"] = float(all_pairs_50["false_sign_rate_declared"].iloc[0])
                row["sign_accuracy"] = float(all_pairs_50["selective_accuracy"].iloc[0])
        _add_target_aware_metrics(
            row,
            posterior,
            truth,
            enabled=bool(config.get("real_data_target_uncertainty", True)),
        )
        metrics_rows.append(row)
    metrics = pd.DataFrame(metrics_rows)
    pairwise = pd.concat(pairwise_rows, ignore_index=True) if pairwise_rows else pd.DataFrame()
    selective = pd.concat(selective_rows, ignore_index=True) if selective_rows else pd.DataFrame()
    stratified = pd.concat(stratified_rows, ignore_index=True) if stratified_rows else pd.DataFrame()
    leaderboard = pd.concat(leaderboard_rows, ignore_index=True) if leaderboard_rows else pd.DataFrame()
    decidability = _target_decidability_summary(truth)
    decidability.insert(0, "run_id", str(config.get("run_id", output_dir.name)))
    decidability.insert(0, "graph_design", str(config.get("graph_design", "observed")))
    _copy_prepared_tables(data_dir, output_dir)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pairwise.to_csv(output_dir / "pairwise_calibration.csv", index=False)
    selective.to_csv(output_dir / "selective_decision_curves.csv", index=False)
    decidability.to_csv(output_dir / "target_decidability_summary.csv", index=False)
    stratified.to_csv(output_dir / "stratified_coverage.csv", index=False)
    leaderboard.to_csv(output_dir / "leaderboard_outputs.csv", index=False)
    leaderboard_validation = _validate_leaderboard_outputs(leaderboard, output_dir)
    _graph_edges_frame(graph, data).to_csv(output_dir / "graph_edges.csv", index=False)
    pairwise_gap_report = _pairwise_gap_report(posterior_by_method, truth_eval, experiment_mode=experiment_mode)
    pairwise_gap_report.to_csv(output_dir / "helm_pairwise_gap_report.csv", index=False)
    grouped = _grouped_leaderboard_outputs(
        posterior_by_method,
        data,
        experiment_mode=experiment_mode,
        threshold=float(config.get("decision_threshold", 0.95)),
    )
    grouped.to_csv(output_dir / "grouped_leaderboard_outputs.csv", index=False)
    (output_dir / "graph_diagnostics.json").write_text(json.dumps(_json_ready(graph_summary), indent=2), encoding="utf-8")
    diagnostics_payload = {
        "method_diagnostics": method_diagnostics,
        "view_metadata": view_metadata.to_dict(orient="records"),
        "truth_columns": truth.columns.tolist(),
        "config": config,
        "quality_gate": quality_gate_payload,
        "leaderboard_validation": leaderboard_validation,
    }
    (output_dir / "method_diagnostics.json").write_text(
        json.dumps(_json_ready(diagnostics_payload), indent=2),
        encoding="utf-8",
    )
    write_graph_plots(metrics, pairwise, output_dir)
    summary_lines = [
        "# HELM Results Summary",
        "",
        f"Run ID: `{config.get('run_id', output_dir.name)}`",
        f"Experiment mode: `{experiment_mode}`",
        f"Items: {data.n_items}",
        f"Views: {data.n_views}",
        f"Rows: {data.n_rows}",
        f"Methods: {', '.join(metrics['method_name'].astype(str).tolist()) if not metrics.empty else 'none'}",
        f"Graph edges: {graph_summary.get('n_edges', 0)}",
        f"Rank deficiency delta_G: {graph_summary.get('rank_deficiency_delta', np.nan)}",
        "",
    ]
    if config.get("allow_view_mean_truth", False):
        run_metadata_path = data_dir / "run_metadata.json"
        used_proxy = False
        warning = "Target is an all-views proxy, not an independent high-budget target."
        if run_metadata_path.exists():
            try:
                metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
                used_proxy = bool(metadata.get("used_proxy_truth", False))
                warning = str(metadata.get("truth_warning") or warning)
            except json.JSONDecodeError:
                used_proxy = False
        if used_proxy:
            summary_lines.extend(["## Warning", "", warning, ""])
    (output_dir / "HELM_RESULTS_SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")
    if not pairwise_gap_report.empty:
        best = pairwise_gap_report.loc[pairwise_gap_report["method_name"] == "anchored_calibrated_eb"]
        body = pairwise_gap_report.to_string(index=False)
        lines = [
            "# HELM Pairwise Gap Summary",
            "",
            "Pairwise-gap metrics are computed after orienting all input metrics so larger values are better.",
            "",
            body,
            "",
        ]
        if not best.empty:
            lines.append(
                f"ACEB leave-view sign accuracy: {float(best['sign_accuracy'].iloc[0]):.3f}; "
                f"false-sign rate: {float(best['false_sign_rate'].iloc[0]):.3f}."
            )
        (output_dir / "HELM_PAIRWISE_GAP_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        (output_dir / "HELM_PAIRWISE_GAP_SUMMARY.md").write_text(
            "# HELM Pairwise Gap Summary\n\nNot applicable because experiment_mode is not pairwise_gap_mode.\n",
            encoding="utf-8",
        )
    return metrics


def run_helm(*args, **kwargs):
    return run_helm_benchmark(*args, **kwargs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run HELM-style ACEB benchmark.")
    parser.add_argument("--config", type=Path, default=Path("configs/helm_small.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/helm/local_fixture"))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    metrics = run_helm_benchmark(output_dir=args.output_dir, config=config)
    print(f"Wrote {len(metrics)} HELM-style metric rows to {args.output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
