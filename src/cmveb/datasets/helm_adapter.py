from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cmveb.benchmarks.graph_synthetic import graph_design_edges
from cmveb.graph_diagnostics import build_coobservation_graph
from cmveb.indexing import build_indexed_data
from cmveb.io import load_prepared_data


REQUIRED_NORMALIZED_COLUMNS = ["model_id", "scenario_id", "metric_name", "view_id", "estimate"]
OPTIONAL_NORMALIZED_COLUMNS = [
    "standard_error",
    "n_examples",
    "split_id",
    "prompt_subset",
    "subscenario",
    "perturbation",
    "judge_id",
    "rubric_id",
    "seed",
    "raw_count_correct",
    "raw_count_total",
    "higher_is_better",
    "target_estimate",
    "target_standard_error",
    "target_mean",
    "target_se",
    "model_a",
    "model_b",
]


GRAPH_DESIGN_ALIASES = {
    "all_views": "all_views",
    "observed": "all_views",
    "star_views": "star",
    "chain_views": "chain",
    "odd_cycle_views": "odd_cycle",
    "star_plus_edge_views": "star_plus_edge_closes_triangle",
    "prompt_subsets_star": "star",
    "scenario_slices_chain": "chain",
    "judge_metric_odd_cycle": "odd_cycle",
    "random_sparse_views": "random_erdos_renyi",
    "full_views_high_budget": "complete",
    "star_plus_edge": "star_plus_edge_closes_triangle",
}


@dataclass(slots=True)
class HelmAdapterConfig:
    experiment_mode: str = "model_score_mode"
    graph_design: str | None = None
    standard_error_floor: float = 0.02
    truth_source: str = "target_or_view_mean"
    item_feature_mode: str = "metadata"
    er_density: float = 0.35
    random_state: int = 0
    require_higher_is_better: bool = True
    allow_view_mean_truth: bool = False
    max_items: int | None = None


@dataclass(slots=True)
class HelmTablePaths:
    root: Path
    normalized_input: Path
    observations: Path
    item_features: Path
    view_metadata: Path
    truth: Path
    run_metadata: Path


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def read_helm_normalized(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_NORMALIZED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"HELM normalized input missing required columns: {missing}")
    return frame


def _standard_error(frame: pd.DataFrame, floor: float) -> pd.Series:
    if "standard_error" in frame.columns:
        se = pd.to_numeric(frame["standard_error"], errors="coerce")
    else:
        se = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    if {"raw_count_correct", "raw_count_total"}.issubset(frame.columns):
        total = pd.to_numeric(frame["raw_count_total"], errors="coerce")
        correct = pd.to_numeric(frame["raw_count_correct"], errors="coerce")
        p = correct / total
        binomial = np.sqrt(np.maximum(p * (1.0 - p) / total, 0.0))
        se = se.fillna(binomial)
    if "n_examples" in frame.columns:
        n = pd.to_numeric(frame["n_examples"], errors="coerce")
        estimate = pd.to_numeric(frame["estimate"], errors="coerce")
        bounded = estimate.between(0.0, 1.0, inclusive="both")
        p = estimate.clip(lower=0.0, upper=1.0)
        binomial_style = np.sqrt(np.maximum(p * (1.0 - p) / n, 0.0))
        se = se.mask(se.isna() & bounded & binomial_style.notna(), binomial_style)
    missing = se.isna()
    if missing.any():
        warnings.warn(
            f"{int(missing.sum())} HELM rows missing standard_error/counts; using conservative floor {floor}.",
            RuntimeWarning,
            stacklevel=2,
        )
        se = se.fillna(floor)
    return se.clip(lower=floor).astype(float)


def _as_bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        return values.astype(float).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "higher", "high"}
    falsey = {"false", "0", "no", "n", "lower", "low"}
    parsed = normalized.map(lambda value: True if value in truthy else False if value in falsey else np.nan)
    if parsed.isna().any():
        bad = sorted(values.loc[parsed.isna()].astype(str).unique().tolist())
        raise ValueError(f"higher_is_better contains unrecognized values: {bad}")
    return parsed.astype(bool)


def _add_truth_aliases(truth: pd.DataFrame, floor: float) -> pd.DataFrame:
    truth = truth.copy()
    if "target_mean" not in truth.columns and "target_estimate" in truth.columns:
        truth["target_mean"] = truth["target_estimate"]
    if "target_estimate" not in truth.columns and "target_mean" in truth.columns:
        truth["target_estimate"] = truth["target_mean"]
    if "target_se" not in truth.columns and "target_standard_error" in truth.columns:
        truth["target_se"] = truth["target_standard_error"]
    if "target_standard_error" not in truth.columns and "target_se" in truth.columns:
        truth["target_standard_error"] = truth["target_se"]
    for column in ["target_se", "target_standard_error"]:
        if column not in truth.columns:
            truth[column] = floor
        truth[column] = pd.to_numeric(truth[column], errors="coerce").fillna(floor).clip(lower=floor)
    return truth


def _orient_frame(frame: pd.DataFrame, config: HelmAdapterConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = frame.copy()
    if "target_estimate" not in frame.columns and "target_mean" in frame.columns:
        frame["target_estimate"] = frame["target_mean"]
    if "target_standard_error" not in frame.columns and "target_se" in frame.columns:
        frame["target_standard_error"] = frame["target_se"]
    if config.require_higher_is_better and "higher_is_better" not in frame.columns:
        raise ValueError(
            "HELM paper-mode input must include higher_is_better so scores can be oriented consistently. "
            "Set require_higher_is_better=False only for legacy/debug inputs."
        )
    if "higher_is_better" in frame.columns:
        higher = _as_bool_series(frame["higher_is_better"])
    else:
        higher = pd.Series(True, index=frame.index, dtype=bool)
    sign = np.where(higher.to_numpy(dtype=bool), 1.0, -1.0)
    frame["raw_estimate"] = pd.to_numeric(frame["estimate"], errors="raise").astype(float)
    frame["estimate"] = frame["raw_estimate"].to_numpy(dtype=np.float64) * sign
    if "target_estimate" in frame.columns:
        frame["raw_target_estimate"] = pd.to_numeric(frame["target_estimate"], errors="coerce")
        frame["target_estimate"] = frame["raw_target_estimate"].to_numpy(dtype=np.float64) * sign
    frame["higher_is_better"] = higher.astype(bool)
    frame["oriented"] = (~higher).astype(bool)
    metadata = {
        "orientation_applied": bool((~higher).any()),
        "n_lower_is_better_rows": int((~higher).sum()),
        "require_higher_is_better": bool(config.require_higher_is_better),
    }
    return frame, metadata


def _require_or_allow_truth(frame: pd.DataFrame, config: HelmAdapterConfig) -> dict[str, Any]:
    has_target = "target_estimate" in frame.columns and frame["target_estimate"].notna().any()
    if has_target:
        return {"used_proxy_truth": False, "truth_warning": None}
    if not config.allow_view_mean_truth:
        raise ValueError(
            "HELM input is missing target_estimate. Paper-mode runs require an independent high-budget target. "
            "Set allow_view_mean_truth=True only for smoke/debug runs that intentionally use an all-views proxy."
        )
    warning = "Target is an all-views proxy, not an independent high-budget target."
    warnings.warn(warning, RuntimeWarning, stacklevel=2)
    return {"used_proxy_truth": True, "truth_warning": warning}


def _model_score_tables(frame: pd.DataFrame, config: HelmAdapterConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    observations = frame.copy()
    observations["item_id"] = (
        observations["model_id"].astype(str)
        + "__"
        + observations["scenario_id"].astype(str)
        + "__"
        + observations["metric_name"].astype(str)
    )
    observations["standard_error"] = _standard_error(observations, config.standard_error_floor)
    keep = ["item_id", "view_id", "estimate", "standard_error", "raw_estimate", "higher_is_better", "oriented", "metric_name"]
    observations = observations.loc[:, [column for column in keep if column in observations.columns]]
    if "target_estimate" in frame.columns and frame["target_estimate"].notna().any():
        target_aggs: dict[str, tuple[str, str]] = {
            "target_mean": ("target_estimate", "mean"),
            "target_se": ("target_standard_error", "mean"),
            "raw_target_estimate": ("raw_target_estimate", "mean"),
        }
        if "higher_is_better" in frame.columns:
            target_aggs["higher_is_better"] = ("higher_is_better", "first")
        if "oriented" in frame.columns:
            target_aggs["oriented"] = ("oriented", "first")
        truth = (
            frame.assign(item_id=observations["item_id"])
            .groupby("item_id", as_index=False)
            .agg(**{key: value for key, value in target_aggs.items() if value[0] in frame.columns})
        )
        truth["target_source"] = "target_estimate"
    else:
        truth = observations.groupby("item_id", as_index=False).agg(
            target_mean=("estimate", "mean"),
            target_se=("estimate", lambda values: float(np.std(values, ddof=1) / math.sqrt(max(len(values), 1))) if len(values) > 1 else config.standard_error_floor),
        )
        truth["target_source"] = "view_mean"
    truth = _add_truth_aliases(truth, config.standard_error_floor)
    return observations, truth


def _pairwise_gap_tables(frame: pd.DataFrame, config: HelmAdapterConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if {"model_a", "model_b"}.issubset(frame.columns):
        pairs = frame.copy()
    else:
        frame = frame.copy()
        frame["_row_standard_error"] = _standard_error(frame, config.standard_error_floor)
        rows: list[dict[str, Any]] = []
        keys = ["scenario_id", "metric_name", "view_id"]
        extra_keys = [column for column in ["prompt_subset", "subscenario", "perturbation", "judge_id", "rubric_id", "seed"] if column in frame.columns]
        for _, group in frame.groupby(keys + extra_keys, dropna=False):
            records = group.to_dict(orient="records")
            for idx, left in enumerate(records):
                for right in records[idx + 1 :]:
                    rows.append(
                        {
                            **{key: left.get(key) for key in keys + extra_keys},
                            "model_a": left["model_id"],
                            "model_b": right["model_id"],
                            "estimate": float(left["estimate"]) - float(right["estimate"]),
                            "standard_error": math.sqrt(
                                float(left["_row_standard_error"]) ** 2
                                + float(right["_row_standard_error"]) ** 2
                            ),
                            "target_estimate": (
                                float(left.get("target_estimate")) - float(right.get("target_estimate"))
                                if pd.notna(left.get("target_estimate")) and pd.notna(right.get("target_estimate"))
                                else np.nan
                            ),
                            "target_standard_error": (
                                math.sqrt(float(left.get("target_standard_error")) ** 2 + float(right.get("target_standard_error")) ** 2)
                                if pd.notna(left.get("target_standard_error")) and pd.notna(right.get("target_standard_error"))
                                else np.nan
                            ),
                            "raw_target_estimate": (
                                float(left.get("raw_target_estimate")) - float(right.get("raw_target_estimate"))
                                if pd.notna(left.get("raw_target_estimate")) and pd.notna(right.get("raw_target_estimate"))
                                else np.nan
                            ),
                            "target_source": (
                                str(left.get("target_source"))
                                if str(left.get("target_source")) == str(right.get("target_source"))
                                else f"{left.get('target_source')};{right.get('target_source')}"
                            ),
                            "higher_is_better": bool(left.get("higher_is_better", True)),
                            "oriented": bool(left.get("oriented", False)) or bool(right.get("oriented", False)),
                        }
                    )
        pairs = pd.DataFrame(rows)
    pairs["item_id"] = (
        pairs["model_a"].astype(str)
        + "__minus__"
        + pairs["model_b"].astype(str)
        + "__"
        + pairs["scenario_id"].astype(str)
        + "__"
        + pairs["metric_name"].astype(str)
    )
    pairs["standard_error"] = _standard_error(pairs, config.standard_error_floor)
    keep = ["item_id", "view_id", "estimate", "standard_error", "raw_estimate", "higher_is_better", "oriented", "metric_name"]
    observations = pairs.loc[:, [column for column in keep if column in pairs.columns]].copy()
    if "target_estimate" in pairs.columns and pairs["target_estimate"].notna().any():
        aggs: dict[str, tuple[str, Any]] = {
            "target_mean": ("target_estimate", "mean"),
            "target_se": (
                "target_standard_error",
                lambda values: float(np.mean(pd.to_numeric(values, errors="coerce").dropna()))
                if len(pd.to_numeric(values, errors="coerce").dropna()) > 0
                else config.standard_error_floor,
            ),
        }
        if "raw_target_estimate" in pairs.columns:
            aggs["raw_target_estimate"] = ("raw_target_estimate", "mean")
        if "higher_is_better" in pairs.columns:
            aggs["higher_is_better"] = ("higher_is_better", "first")
        if "oriented" in pairs.columns:
            aggs["oriented"] = ("oriented", "first")
        if "target_source" in pairs.columns:
            aggs["target_source"] = (
                "target_source",
                lambda values: ";".join(sorted({str(value) for value in values.dropna()})),
            )
        truth = pairs.groupby("item_id", as_index=False).agg(**aggs)
        if "target_source" not in truth.columns:
            truth["target_source"] = "target_gap"
    else:
        truth = observations.groupby("item_id", as_index=False).agg(
            target_mean=("estimate", "mean"),
            target_se=("estimate", lambda values: float(np.std(values, ddof=1) / math.sqrt(max(len(values), 1))) if len(values) > 1 else config.standard_error_floor),
        )
        truth["target_source"] = "view_gap_mean"
    truth = _add_truth_aliases(truth, config.standard_error_floor)
    return observations, truth


def _item_features(frame: pd.DataFrame, item_ids: pd.Series) -> pd.DataFrame:
    unique = pd.DataFrame({"item_id": sorted(item_ids.unique().tolist())})
    parts = unique["item_id"].str.split("__", expand=True)
    unique["item_hash"] = unique["item_id"].map(lambda value: float(abs(hash(value)) % 100000) / 100000.0)
    unique["is_pairwise_gap"] = unique["item_id"].str.contains("__minus__").astype(float)
    if parts.shape[1] > 0:
        unique["model_hash"] = parts[0].map(lambda value: float(abs(hash(value)) % 100000) / 100000.0)
    if "metric_name" in frame.columns:
        metric_from_item = unique["item_id"].map(lambda value: str(value).split("__")[-1])
        unique["metric_hash"] = metric_from_item.map(lambda value: float(abs(hash(value)) % 100000) / 100000.0)
        if "higher_is_better" in frame.columns:
            higher_by_metric = frame.groupby("metric_name")["higher_is_better"].first().to_dict()
            unique["higher_is_better"] = metric_from_item.map(lambda value: float(bool(higher_by_metric.get(value, True))))
        if "oriented" in frame.columns:
            oriented_by_metric = frame.groupby("metric_name")["oriented"].any().to_dict()
            unique["oriented"] = metric_from_item.map(lambda value: float(bool(oriented_by_metric.get(value, False))))
    scenario_values = frame[["scenario_id"]].drop_duplicates().reset_index(drop=True)
    unique["n_scenarios_total"] = float(max(1, scenario_values.shape[0]))
    return unique


def _view_metadata(frame: pd.DataFrame, observations: pd.DataFrame, graph_design: str | None) -> pd.DataFrame:
    views = pd.DataFrame({"view_id": sorted(observations["view_id"].astype(str).unique().tolist())})
    views["is_reference_view"] = views.index == 0
    views["view_type"] = "helm_slice"
    for column in ["split_id", "prompt_subset", "subscenario", "perturbation", "judge_id", "rubric_id", "seed"]:
        if column in frame.columns:
            mapping = frame.drop_duplicates("view_id").set_index("view_id")[column].to_dict()
            views[column] = views["view_id"].map(mapping)
    for column in ["metric_name", "higher_is_better", "oriented"]:
        if column in frame.columns:
            mapping = frame.groupby("view_id")[column].agg(lambda values: ";".join(sorted({str(value) for value in values.dropna()}))).to_dict()
            views[column] = views["view_id"].map(mapping)
    views["graph_design"] = graph_design or "observed"
    return views


def _subsample_items(
    observations: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    max_items: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if max_items is None or max_items <= 0:
        return observations, truth
    item_ids = np.asarray(sorted(observations["item_id"].astype(str).unique().tolist()), dtype=object)
    if item_ids.size <= max_items:
        return observations, truth
    rng = np.random.default_rng(random_state)
    keep = set(rng.choice(item_ids, size=int(max_items), replace=False).astype(str).tolist())
    observations = observations.loc[observations["item_id"].astype(str).isin(keep)].copy()
    truth = truth.loc[truth["item_id"].astype(str).isin(keep)].copy()
    return observations, truth


def _apply_graph_design(
    observations: pd.DataFrame,
    view_metadata: pd.DataFrame,
    *,
    graph_design: str | None,
    er_density: float,
    random_state: int,
) -> pd.DataFrame:
    if graph_design is None:
        return observations
    canonical = GRAPH_DESIGN_ALIASES.get(graph_design, graph_design)
    if canonical == "all_views":
        return observations
    view_ids = view_metadata["view_id"].tolist()
    edges = graph_design_edges(canonical, len(view_ids), density=er_density, seed=random_state)
    item_ids = sorted(observations["item_id"].unique().tolist())
    edge_counts = np.full(len(edges), len(item_ids) // len(edges), dtype=np.int64)
    edge_counts[: len(item_ids) % len(edges)] += 1
    item_edge = np.repeat(np.arange(len(edges)), edge_counts)[: len(item_ids)]
    rng = np.random.default_rng(random_state)
    rng.shuffle(item_edge)
    allowed: set[tuple[str, str]] = set()
    for item_id, edge_idx in zip(item_ids, item_edge, strict=True):
        for view_idx in edges[int(edge_idx)]:
            allowed.add((item_id, view_ids[view_idx]))
    subset = observations.loc[
        [(item, view) in allowed for item, view in zip(observations["item_id"], observations["view_id"], strict=True)]
    ].copy()
    return subset


def build_helm_tables(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    config: HelmAdapterConfig | None = None,
) -> HelmTablePaths:
    config = config or HelmAdapterConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = read_helm_normalized(input_path)
    frame["view_id"] = frame["view_id"].astype(str)
    frame, orientation_metadata = _orient_frame(frame, config)
    truth_metadata = _require_or_allow_truth(frame, config)
    if config.experiment_mode == "model_score_mode":
        observations, truth = _model_score_tables(frame, config)
    elif config.experiment_mode == "pairwise_gap_mode":
        observations, truth = _pairwise_gap_tables(frame, config)
    else:
        raise ValueError("experiment_mode must be 'model_score_mode' or 'pairwise_gap_mode'.")
    observations, truth = _subsample_items(
        observations,
        truth,
        max_items=config.max_items,
        random_state=config.random_state,
    )
    item_features = _item_features(frame, observations["item_id"])
    view_metadata = _view_metadata(frame, observations, config.graph_design)
    observations = _apply_graph_design(
        observations,
        view_metadata,
        graph_design=config.graph_design,
        er_density=config.er_density,
        random_state=config.random_state,
    )
    paths = HelmTablePaths(
        root=output_dir,
        normalized_input=Path(input_path),
        observations=output_dir / "observations.parquet",
        item_features=output_dir / "item_features.parquet",
        view_metadata=output_dir / "view_metadata.parquet",
        truth=output_dir / "truth.parquet",
        run_metadata=output_dir / "run_metadata.json",
    )
    observations.to_parquet(paths.observations, index=False)
    item_features.to_parquet(paths.item_features, index=False)
    view_metadata.to_parquet(paths.view_metadata, index=False)
    truth.to_parquet(paths.truth, index=False)
    prepared = load_prepared_data(paths.observations, paths.item_features, paths.view_metadata)
    data = build_indexed_data(prepared)
    graph = build_coobservation_graph(data, min_shared_items=1).to_summary_dict()
    payload = {
        "config": asdict(config),
        "required_columns": REQUIRED_NORMALIZED_COLUMNS,
        "optional_columns": OPTIONAL_NORMALIZED_COLUMNS,
        **orientation_metadata,
        **truth_metadata,
        "graph_diagnostic": graph,
    }
    paths.run_metadata.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    return paths


def load_helm_views(
    root: str | Path,
    *,
    standardize_features: bool = True,
) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    prepared = load_prepared_data(
        root / "observations.parquet",
        root / "item_features.parquet",
        root / "view_metadata.parquet",
        standardize_features=standardize_features,
    )
    data = build_indexed_data(prepared)
    truth = pd.read_parquet(root / "truth.parquet")
    view_metadata = pd.read_parquet(root / "view_metadata.parquet")
    return data, truth, view_metadata


__all__ = [
    "GRAPH_DESIGN_ALIASES",
    "HelmAdapterConfig",
    "HelmTablePaths",
    "OPTIONAL_NORMALIZED_COLUMNS",
    "REQUIRED_NORMALIZED_COLUMNS",
    "build_helm_tables",
    "load_helm_views",
    "read_helm_normalized",
]
