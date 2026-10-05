from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cmveb.benchmarks.graph_synthetic import graph_design_edges
from cmveb.graph_diagnostics import build_coobservation_graph
from cmveb.indexing import build_indexed_data
from cmveb.io import load_prepared_data


@dataclass(slots=True)
class OpenMLTablePaths:
    root: Path
    observations: Path
    item_features: Path
    view_metadata: Path
    truth: Path
    run_metadata: Path


@dataclass(slots=True)
class OpenMLBuildConfig:
    mode: str = "toy"
    graph_design: str = "star_plus_edge"
    n_datasets: int = 6
    n_views: int = 5
    metric: str = "accuracy"
    metrics: tuple[str, ...] = ("accuracy",)
    transform: str = "raw"
    random_state: int = 0
    min_standard_error: float = 0.015
    n_test_min: int = 80
    n_test_max: int = 240
    er_density: float = 0.35
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    n_folds: int = 5
    datasets: tuple[str, ...] = ("breast_cancer", "wine", "digits")
    digits_max_samples: int = 600
    strict_algorithm_failures: bool = False


DEFAULT_ALGORITHMS = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "svm",
    "dummy_classifier",
]


GRAPH_DESIGN_ALIASES = {
    "all_views": "all_views",
    "observed": "all_views",
    "star_views": "star",
    "chain_views": "chain",
    "odd_cycle_views": "odd_cycle",
    "star_plus_edge_views": "star_plus_edge_closes_triangle",
    "independent_folds_only": "complete",
    "fold_by_seed_star": "star",
    "chain_overlap": "chain",
    "odd_cycle_overlap": "odd_cycle",
    "random_sparse": "random_erdos_renyi",
    "star_plus_edge": "star_plus_edge_closes_triangle",
}


def openml_available() -> bool:
    try:
        import openml  # noqa: F401
    except ImportError:
        return False
    return True


def openml_install_message() -> str:
    return "Install optional OpenML support with: pip install openml scikit-learn"


def canonical_graph_design(design: str) -> str:
    return GRAPH_DESIGN_ALIASES.get(design, design)


def _metric_transform(value: np.ndarray, se: np.ndarray, transform: str) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(value, dtype=np.float64)
    se = np.asarray(se, dtype=np.float64)
    if transform == "raw":
        return value, se
    if transform == "logit":
        clipped = np.clip(value, 1e-4, 1.0 - 1e-4)
        return np.log(clipped / (1.0 - clipped)), se / np.maximum(clipped * (1.0 - clipped), 1e-8)
    raise ValueError("transform must be 'raw' or 'logit'.")


def _inverse_metric_transform(value: np.ndarray, se: np.ndarray, transform: str) -> tuple[np.ndarray, np.ndarray]:
    if transform == "raw":
        return value, se
    if transform == "logit":
        prob = 1.0 / (1.0 + np.exp(-value))
        return prob, se * prob * (1.0 - prob)
    raise ValueError("transform must be 'raw' or 'logit'.")


def _add_truth_aliases(truth: pd.DataFrame) -> pd.DataFrame:
    truth = truth.copy()
    if "target_mean" not in truth.columns and "target_estimate" in truth.columns:
        truth["target_mean"] = truth["target_estimate"]
    if "target_estimate" not in truth.columns and "target_mean" in truth.columns:
        truth["target_estimate"] = truth["target_mean"]
    if "target_se" not in truth.columns and "target_standard_error" in truth.columns:
        truth["target_se"] = truth["target_standard_error"]
    if "target_standard_error" not in truth.columns and "target_se" in truth.columns:
        truth["target_standard_error"] = truth["target_se"]
    return truth


def _sklearn_missing_message() -> str:
    return "Install optional local-real OpenML support with: pip install scikit-learn"


def sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return False
    return True


def _toy_truth(
    rng: np.random.Generator,
    algorithms: list[str],
    n_datasets: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_difficulty = rng.normal(0.0, 0.8, size=n_datasets)
    dataset_size = rng.integers(200, 2500, size=n_datasets)
    dataset_noise = rng.uniform(0.05, 0.35, size=n_datasets)
    algorithm_skill = {
        "dummy_classifier": -1.2,
        "logistic_regression": 0.25,
        "svm": 0.35,
        "random_forest": 0.55,
        "gradient_boosting": 0.65,
    }
    rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for dataset_idx in range(n_datasets):
        dataset_id = f"dataset_{dataset_idx:03d}"
        for algorithm_idx, algorithm_id in enumerate(algorithms):
            item_id = f"{algorithm_id}__{dataset_id}"
            latent = algorithm_skill.get(algorithm_id, 0.0) - 0.35 * dataset_difficulty[dataset_idx]
            latent += rng.normal(0.0, 0.08)
            target_mean = 0.5 + 0.45 / (1.0 + np.exp(-latent))
            target_mean = float(np.clip(target_mean, 0.05, 0.98))
            rows.append(
                {
                    "item_id": item_id,
                    "target_mean": target_mean,
                    "target_se": float(max(0.004, math.sqrt(target_mean * (1.0 - target_mean) / (20 * dataset_size[dataset_idx])))),
                    "target_source": "toy_high_budget",
                }
            )
            feature = {
                "item_id": item_id,
                "dataset_size": float(dataset_size[dataset_idx]),
                "dataset_noise": float(dataset_noise[dataset_idx]),
                "dataset_difficulty": float(dataset_difficulty[dataset_idx]),
                "algorithm_index": float(algorithm_idx),
            }
            for name in algorithms:
                feature[f"algorithm_is_{name}"] = float(name == algorithm_id)
            feature_rows.append(feature)
    return pd.DataFrame(rows), pd.DataFrame(feature_rows)


def build_toy_openml_tables(
    output_dir: str | Path,
    *,
    config: OpenMLBuildConfig | None = None,
    algorithms: list[str] | None = None,
) -> OpenMLTablePaths:
    config = config or OpenMLBuildConfig()
    algorithms = algorithms or DEFAULT_ALGORITHMS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.random_state)
    truth, item_features = _toy_truth(rng, algorithms, config.n_datasets)
    n_items = truth.shape[0]
    view_ids = np.array([f"view_{idx:02d}" for idx in range(config.n_views)])
    graph_design = canonical_graph_design(config.graph_design)
    edges = graph_design_edges(graph_design, config.n_views, density=config.er_density, seed=config.random_state)
    edge_counts = np.full(len(edges), n_items // len(edges), dtype=np.int64)
    edge_counts[: n_items % len(edges)] += 1
    item_edge = np.repeat(np.arange(len(edges)), edge_counts)[:n_items]
    rng.shuffle(item_edge)
    truth_mean = truth["target_mean"].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    view_meta_rows = []
    view_bias = rng.normal(0.0, 0.01, size=config.n_views)
    view_bias[0] = 0.0
    for view_idx, view_id in enumerate(view_ids):
        view_meta_rows.append(
            {
                "view_id": view_id,
                "is_reference_view": bool(view_idx == 0),
                "view_type": "fold_seed_slice",
                "fold": int(view_idx % max(2, config.n_views // 2)),
                "seed": int(config.random_state + view_idx),
                "slice": f"slice_{view_idx % 3}",
                "graph_design": config.graph_design,
            }
        )
    for item_idx, item_id in enumerate(truth["item_id"].tolist()):
        for view_idx in edges[int(item_edge[item_idx])]:
            n_test = int(rng.integers(config.n_test_min, config.n_test_max + 1))
            p = float(np.clip(truth_mean[item_idx] + view_bias[view_idx], 1e-4, 1.0 - 1e-4))
            se_raw = max(config.min_standard_error, math.sqrt(p * (1.0 - p) / n_test))
            estimate_raw = float(np.clip(rng.normal(p, se_raw), 1e-4, 1.0 - 1e-4))
            estimate, se = _metric_transform(np.array([estimate_raw]), np.array([se_raw]), config.transform)
            rows.append(
                {
                    "item_id": item_id,
                    "view_id": str(view_ids[view_idx]),
                    "estimate": float(estimate[0]),
                    "standard_error": float(max(config.min_standard_error, se[0])),
                }
            )
    observations = pd.DataFrame(rows)
    view_metadata = pd.DataFrame(view_meta_rows)
    if config.transform != "raw":
        transformed_mean, transformed_se = _metric_transform(
            truth["target_mean"].to_numpy(dtype=np.float64),
            truth["target_se"].to_numpy(dtype=np.float64),
            config.transform,
        )
        truth = truth.assign(target_mean=transformed_mean, target_se=transformed_se)
    truth = _add_truth_aliases(truth)

    paths = OpenMLTablePaths(
        root=output_dir,
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
    graph = build_coobservation_graph(
        item_idx=pd.Categorical(observations["item_id"], categories=item_features["item_id"]).codes,
        view_idx=pd.Categorical(observations["view_id"], categories=view_ids).codes,
        estimate=observations["estimate"].to_numpy(dtype=np.float64),
        n_views=config.n_views,
        min_shared_items=1,
    )
    metadata = {
        "config": asdict(config),
        "algorithms": algorithms,
        "graph_design_canonical": graph_design,
        "intended_edges": edges,
        "graph_diagnostic": graph.to_summary_dict(),
        "package_mode": "toy_numpy",
    }
    paths.run_metadata.write_text(json.dumps(_json_ready(metadata), indent=2), encoding="utf-8")
    return paths


def _load_sklearn_dataset(name: str, *, max_digits_samples: int, random_state: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    try:
        from sklearn import datasets as sklearn_datasets
    except ImportError as exc:
        raise RuntimeError(_sklearn_missing_message()) from exc
    if name == "breast_cancer":
        bunch = sklearn_datasets.load_breast_cancer()
    elif name == "wine":
        bunch = sklearn_datasets.load_wine()
    elif name == "digits":
        bunch = sklearn_datasets.load_digits()
    elif name == "iris":
        bunch = sklearn_datasets.load_iris()
    else:
        raise ValueError(f"Unknown local_real dataset: {name}")
    x = np.asarray(bunch.data, dtype=np.float64)
    y = np.asarray(bunch.target)
    if name == "digits" and x.shape[0] > max_digits_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(x.shape[0], size=max_digits_samples, replace=False)
        x = x[idx]
        y = y[idx]
    features = {
        "dataset_n_samples": float(x.shape[0]),
        "dataset_n_features": float(x.shape[1]),
        "dataset_n_classes": float(len(np.unique(y))),
    }
    return x, y, features


def _make_classifier(name: str, random_state: int):
    try:
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(_sklearn_missing_message()) from exc
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, solver="lbfgs", random_state=random_state))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=50, max_depth=8, random_state=random_state, n_jobs=1)
    if name in {"gradient_boosting", "hist_gradient_boosting"}:
        if name == "hist_gradient_boosting":
            return HistGradientBoostingClassifier(max_iter=60, random_state=random_state)
        return GradientBoostingClassifier(n_estimators=50, random_state=random_state)
    if name == "k_neighbors":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    if name == "dummy_classifier":
        return DummyClassifier(strategy="most_frequent")
    raise ValueError(f"Unknown local_real algorithm: {name}")


def _metric_score(name: str, y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None) -> tuple[float, float]:
    try:
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
    except ImportError as exc:
        raise RuntimeError(_sklearn_missing_message()) from exc
    n_test = max(int(len(y_true)), 1)
    if name == "accuracy":
        score = float(accuracy_score(y_true, y_pred))
        se = math.sqrt(max(score * (1.0 - score), 1e-4) / n_test)
        return score, se
    if name == "balanced_accuracy":
        score = float(balanced_accuracy_score(y_true, y_pred))
        se = max(0.03, math.sqrt(max(score * (1.0 - score), 1e-4) / n_test))
        return score, se
    if name == "log_loss":
        if proba is None:
            raise ValueError("log_loss requires predict_proba.")
        losses = -np.log(np.clip(proba[np.arange(n_test), y_true.astype(int)], 1e-12, 1.0))
        score = -float(log_loss(y_true, proba, labels=np.unique(y_true)))
        se = max(0.03, float(np.std(losses, ddof=1) / math.sqrt(n_test)) if n_test > 1 else 0.03)
        return score, se
    raise ValueError(f"Unknown metric: {name}")


def _local_real_full_tables(
    *,
    config: OpenMLBuildConfig,
    algorithms: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    try:
        from sklearn.model_selection import StratifiedKFold
    except ImportError as exc:
        raise RuntimeError(_sklearn_missing_message()) from exc
    metric_names = tuple(config.metrics) if config.metrics else (config.metric,)
    rows: list[dict[str, Any]] = []
    item_feature_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    dataset_features: dict[str, dict[str, float]] = {}
    for dataset_name in config.datasets:
        x, y, features = _load_sklearn_dataset(
            dataset_name,
            max_digits_samples=config.digits_max_samples,
            random_state=config.random_state,
        )
        dataset_features[dataset_name] = features
        for algorithm_idx, algorithm_id in enumerate(algorithms):
            for metric_name in metric_names:
                item_id = f"{algorithm_id}__{dataset_name}__{metric_name}"
                feature = {
                    "item_id": item_id,
                    "algorithm_index": float(algorithm_idx),
                    **features,
                }
                for name in algorithms:
                    feature[f"algorithm_is_{name}"] = float(name == algorithm_id)
                item_feature_rows.append(feature)
            for seed in config.seeds:
                splitter = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=int(seed))
                for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y)):
                    view_id = f"seed_{seed}_fold_{fold}"
                    try:
                        model = _make_classifier(algorithm_id, random_state=int(seed))
                        model.fit(x[train_idx], y[train_idx])
                        pred = model.predict(x[test_idx])
                        proba = model.predict_proba(x[test_idx]) if hasattr(model, "predict_proba") else None
                    except Exception as exc:
                        failure = {
                            "algorithm_id": algorithm_id,
                            "dataset_id": dataset_name,
                            "seed": int(seed),
                            "fold": int(fold),
                            "exception_type": type(exc).__name__,
                            "message": str(exc),
                        }
                        failure_rows.append(failure)
                        if config.strict_algorithm_failures:
                            raise
                        continue
                    for metric_name in metric_names:
                        try:
                            estimate, se = _metric_score(metric_name, y[test_idx], pred, proba)
                        except Exception as exc:
                            failure_rows.append(
                                {
                                    "algorithm_id": algorithm_id,
                                    "dataset_id": dataset_name,
                                    "metric_name": metric_name,
                                    "seed": int(seed),
                                    "fold": int(fold),
                                    "exception_type": type(exc).__name__,
                                    "message": str(exc),
                                }
                            )
                            if config.strict_algorithm_failures:
                                raise
                            continue
                        rows.append(
                            {
                                "item_id": f"{algorithm_id}__{dataset_name}__{metric_name}",
                                "view_id": view_id,
                                "estimate": estimate,
                                "standard_error": max(float(se), config.min_standard_error),
                                "algorithm_id": algorithm_id,
                                "dataset_id": dataset_name,
                                "metric_name": metric_name,
                                "seed": int(seed),
                                "fold": int(fold),
                                "n_test": int(len(test_idx)),
                            }
                        )
    observations_full = pd.DataFrame(rows)
    if observations_full.empty:
        raise RuntimeError("local_real produced no observations.")
    truth = (
        observations_full.groupby("item_id", as_index=False)
        .agg(
            target_mean=("estimate", "mean"),
            target_se=("estimate", lambda values: float(np.std(values, ddof=1) / math.sqrt(max(len(values), 1))) if len(values) > 1 else config.min_standard_error),
        )
        .assign(target_source="local_real_repeated_cv")
    )
    truth["target_se"] = truth["target_se"].fillna(config.min_standard_error).clip(lower=config.min_standard_error)
    truth = _add_truth_aliases(truth)
    item_features = pd.DataFrame(item_feature_rows).drop_duplicates("item_id")
    view_metadata = (
        observations_full[["view_id", "seed", "fold"]]
        .drop_duplicates("view_id")
        .sort_values(["seed", "fold"])
        .reset_index(drop=True)
    )
    view_metadata["is_reference_view"] = view_metadata.index == 0
    view_metadata["view_type"] = "repeated_stratified_kfold"
    view_metadata["graph_design"] = config.graph_design
    metadata = {
        "algorithm_failures": failure_rows,
        "n_algorithm_failures": len(failure_rows),
        "dataset_features": dataset_features,
        "metric_names": list(metric_names),
    }
    return observations_full, item_features, view_metadata, truth, metadata


def _apply_local_real_graph_design(
    observations_full: pd.DataFrame,
    view_metadata: pd.DataFrame,
    *,
    graph_design: str,
    er_density: float,
    random_state: int,
) -> pd.DataFrame:
    canonical = canonical_graph_design(graph_design)
    if canonical == "all_views":
        return observations_full.copy()
    view_ids = view_metadata["view_id"].tolist()
    edges = graph_design_edges(canonical, len(view_ids), density=er_density, seed=random_state)
    item_ids = sorted(observations_full["item_id"].unique().tolist())
    edge_counts = np.full(len(edges), len(item_ids) // len(edges), dtype=np.int64)
    edge_counts[: len(item_ids) % len(edges)] += 1
    item_edge = np.repeat(np.arange(len(edges)), edge_counts)[: len(item_ids)]
    rng = np.random.default_rng(random_state)
    rng.shuffle(item_edge)
    allowed: set[tuple[str, str]] = set()
    for item_id, edge_idx in zip(item_ids, item_edge, strict=True):
        for view_idx in edges[int(edge_idx)]:
            allowed.add((item_id, view_ids[view_idx]))
    return observations_full.loc[
        [(item, view) in allowed for item, view in zip(observations_full["item_id"], observations_full["view_id"], strict=True)]
    ].copy()


def build_local_real_openml_tables(
    output_dir: str | Path,
    *,
    config: OpenMLBuildConfig | None = None,
    algorithms: list[str] | None = None,
) -> OpenMLTablePaths:
    config = config or OpenMLBuildConfig(mode="local_real")
    algorithms = algorithms or DEFAULT_ALGORITHMS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    observations_full, item_features, view_metadata, truth, metadata = _local_real_full_tables(config=config, algorithms=algorithms)
    observations = _apply_local_real_graph_design(
        observations_full,
        view_metadata,
        graph_design=config.graph_design,
        er_density=config.er_density,
        random_state=config.random_state,
    )
    paths = OpenMLTablePaths(
        root=output_dir,
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
    graph = build_coobservation_graph(
        item_idx=pd.Categorical(observations["item_id"], categories=item_features["item_id"]).codes,
        view_idx=pd.Categorical(observations["view_id"], categories=view_metadata["view_id"]).codes,
        estimate=observations["estimate"].to_numpy(dtype=np.float64),
        n_views=view_metadata.shape[0],
        min_shared_items=1,
    )
    payload = {
        "config": asdict(config),
        "algorithms": algorithms,
        "graph_design_canonical": canonical_graph_design(config.graph_design),
        "graph_diagnostic": graph.to_summary_dict(),
        "package_mode": "local_real_sklearn",
        **metadata,
    }
    paths.run_metadata.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    return paths


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


def load_openml_views(
    root: str | Path,
    *,
    standardize_features: bool = True,
) -> tuple[IndexedData, pd.DataFrame, pd.DataFrame]:
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


def build_openml_tables(
    output_dir: str | Path,
    *,
    config: OpenMLBuildConfig | None = None,
    algorithms: list[str] | None = None,
) -> OpenMLTablePaths:
    config = config or OpenMLBuildConfig()
    if config.mode == "toy":
        return build_toy_openml_tables(output_dir, config=config, algorithms=algorithms)
    if config.mode in {"local_real", "sklearn_real"}:
        return build_local_real_openml_tables(output_dir, config=config, algorithms=algorithms)
    if config.mode == "openml" and not openml_available():
        raise RuntimeError(openml_install_message())
    raise NotImplementedError(
        "Network OpenML execution is scaffolded but not enabled in this lightweight adapter yet; use mode='toy'."
    )


__all__ = [
    "DEFAULT_ALGORITHMS",
    "GRAPH_DESIGN_ALIASES",
    "OpenMLBuildConfig",
    "OpenMLTablePaths",
    "build_openml_tables",
    "build_local_real_openml_tables",
    "build_toy_openml_tables",
    "canonical_graph_design",
    "load_openml_views",
    "openml_available",
    "openml_install_message",
]
