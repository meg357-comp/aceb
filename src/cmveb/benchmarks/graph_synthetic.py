from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit

from cmveb.benchmarks.synthetic_runner import (
    compute_metrics,
    evaluate_pairwise_calibration,
    evaluate_stratified_coverage,
    skipped_metric_row,
    _item_view_counts_and_components,
)
from cmveb.graph_diagnostics import build_coobservation_graph, edge_design_matrix, full_column_singular_values
from cmveb.full_bayes import full_bayes_hierarchical
from cmveb.indexing import build_indexed_data
from cmveb.mash_baseline import mashr_multivariate_eb
from cmveb.models.baseline import (
    MethodOutput,
    adaptive_shrinkage_like,
    anchored_calibrated_eb,
    bootstrap_inverse_variance_all_views,
    feature_normal_eb_no_view_scale,
    inverse_variance_all_views,
    normal_eb_no_features,
    posterior_to_frame,
    posthoc_inverse_variance_all_views,
    random_effects_meta_analysis,
    raw_single_reference,
)
from cmveb.models.anchored_eb import (
    fit_non_reference_view_scales,
    fit_prior_hyperparameters,
    initialize_extra_noise,
    initialize_view_scale,
)
from cmveb.models.joint_eb import joint_eb_full
from cmveb.calibration import estimate_tau2_leave_view
from cmveb.posterior import compute_posterior, compute_posterior_from_prior, prior_from_features
from cmveb.preprocess import prepare_data
from cmveb.schemas import IndexedData
from cmveb.reporting import write_run_summary


GRAPH_DESIGNS = [
    "complete",
    "star",
    "star_plus_edge_closes_triangle",
    "chain",
    "even_cycle",
    "odd_cycle",
    "triangle_plus_tails",
    "disconnected",
    "random_erdos_renyi",
    "nearly_bipartite",
]

THETA_DISTRIBUTIONS = ["normal", "t", "mixture", "skewed", "bounded_logit"]
MISSINGNESS_MODES = ["none", "MCAR", "MAR", "MNAR"]

CELL_FACTOR_COLUMNS = [
    "graph_design",
    "n_items",
    "n_views",
    "tau_ratio",
    "loading_spread",
    "seed",
    "run_seed",
    "theta_distribution",
    "missingness",
]


@dataclass(slots=True)
class GraphSyntheticDataset:
    data: IndexedData
    truth: pd.DataFrame
    view_parameters: pd.DataFrame
    intended_edges: list[tuple[int, int]]
    metadata: dict[str, object]


@dataclass(slots=True)
class PairedGraphIntervention:
    datasets: dict[str, GraphSyntheticDataset]
    true_theta: np.ndarray
    true_view_parameters: pd.DataFrame
    metadata: dict[str, object]


@dataclass(slots=True)
class PairedInterventionSpec:
    name: str
    base_design_name: str
    plus_design_name: str
    base_edges: list[tuple[int, int]]
    plus_edges: list[tuple[int, int]]
    additions: list[tuple[tuple[int, int], int]]


@dataclass(slots=True)
class GraphDesignSpec:
    name: str
    n_views: int
    anchor_view: int
    edges: list[tuple[int, int]]
    valid: bool
    skip_reason: Optional[str]
    design_family: str
    expected_bipartite: Optional[bool]
    expected_has_odd_cycle: Optional[bool]
    expected_rank_deficient: Optional[bool]


def graph_design_edges(
    design: str,
    n_views: int,
    *,
    density: float = 0.35,
    seed: int = 0,
) -> list[tuple[int, int]]:
    return make_graph_design(design, n_views, anchor_view=0, density=density, seed=seed).edges


def _invalid_spec(name: str, n_views: int, anchor_view: int, reason: str, family: str) -> GraphDesignSpec:
    return GraphDesignSpec(
        name=name,
        n_views=int(n_views),
        anchor_view=int(anchor_view),
        edges=[],
        valid=False,
        skip_reason=reason,
        design_family=family,
        expected_bipartite=None,
        expected_has_odd_cycle=None,
        expected_rank_deficient=None,
    )


def _spec(
    name: str,
    n_views: int,
    anchor_view: int,
    edges: list[tuple[int, int]],
    *,
    family: str,
    expected_bipartite: Optional[bool],
    expected_rank_deficient: Optional[bool],
) -> GraphDesignSpec:
    clean_edges = sorted(set(tuple(sorted(edge)) for edge in edges if edge[0] != edge[1]))
    return GraphDesignSpec(
        name=name,
        n_views=int(n_views),
        anchor_view=int(anchor_view),
        edges=clean_edges,
        valid=True,
        skip_reason=None,
        design_family=family,
        expected_bipartite=expected_bipartite,
        expected_has_odd_cycle=None if expected_bipartite is None else not expected_bipartite,
        expected_rank_deficient=expected_rank_deficient,
    )


def make_graph_design(
    name: str,
    n_views: int,
    anchor_view: int = 0,
    *,
    density: float = 0.35,
    seed: int = 0,
) -> GraphDesignSpec:
    if n_views < 2:
        return _invalid_spec(name, n_views, anchor_view, "n_views must be at least 2.", name)
    if not 0 <= anchor_view < n_views:
        return _invalid_spec(name, n_views, anchor_view, "anchor_view is out of range.", name)
    if name == "complete":
        if n_views < 3:
            return _invalid_spec(name, n_views, anchor_view, "complete requires n_views >= 3.", "complete")
        edges = [(j, k) for j in range(n_views) for k in range(j + 1, n_views)]
        return _spec(name, n_views, anchor_view, edges, family="complete", expected_bipartite=False, expected_rank_deficient=False)
    if name == "star":
        if n_views < 3:
            return _invalid_spec(name, n_views, anchor_view, "star requires n_views >= 3.", "star")
        edges = [(anchor_view, k) for k in range(n_views) if k != anchor_view]
        return _spec(name, n_views, anchor_view, edges, family="star", expected_bipartite=True, expected_rank_deficient=True)
    if name == "star_plus_edge_closes_triangle":
        if n_views < 3:
            return _invalid_spec(name, n_views, anchor_view, "star_plus_edge_closes_triangle requires n_views >= 3.", "star")
        leaves = [view for view in range(n_views) if view != anchor_view]
        edges = [(anchor_view, k) for k in leaves]
        edges.append((leaves[0], leaves[1]))
        return _spec(
            name,
            n_views,
            anchor_view,
            edges,
            family="star_plus_edge",
            expected_bipartite=False,
            expected_rank_deficient=False,
        )
    if name == "chain":
        if n_views < 3:
            return _invalid_spec(name, n_views, anchor_view, "chain requires n_views >= 3.", "chain")
        edges = [(j, j + 1) for j in range(n_views - 1)]
        return _spec(name, n_views, anchor_view, edges, family="chain", expected_bipartite=True, expected_rank_deficient=True)
    if name == "even_cycle":
        if n_views < 4:
            return _invalid_spec(name, n_views, anchor_view, "even_cycle requires n_views >= 4.", "cycle")
        if n_views % 2 != 0:
            return _invalid_spec(name, n_views, anchor_view, "even_cycle requires an even n_views.", "cycle")
        edges = [(j, (j + 1) % n_views) for j in range(n_views)]
        return _spec(name, n_views, anchor_view, edges, family="cycle", expected_bipartite=True, expected_rank_deficient=True)
    if name == "odd_cycle":
        if n_views < 3:
            return _invalid_spec(name, n_views, anchor_view, "odd_cycle requires n_views >= 3.", "cycle")
        if n_views % 2 == 0:
            return _invalid_spec(name, n_views, anchor_view, "odd_cycle requires an odd n_views.", "cycle")
        edges = [(j, (j + 1) % n_views) for j in range(n_views)]
        return _spec(name, n_views, anchor_view, edges, family="cycle", expected_bipartite=False, expected_rank_deficient=False)
    if name == "even_cycle_plus_tails":
        if n_views < 4:
            return _invalid_spec(name, n_views, anchor_view, "even_cycle_plus_tails requires n_views >= 4.", "cycle_plus_tails")
        cycle_n = 4
        edges = [(j, (j + 1) % cycle_n) for j in range(cycle_n)]
        edges.extend((k - 1, k) for k in range(cycle_n, n_views))
        return _spec(name, n_views, anchor_view, edges, family="cycle_plus_tails", expected_bipartite=True, expected_rank_deficient=True)
    if name == "odd_cycle_plus_tails":
        if n_views < 3:
            return _invalid_spec(name, n_views, anchor_view, "odd_cycle_plus_tails requires n_views >= 3.", "cycle_plus_tails")
        cycle_n = 3
        edges = [(j, (j + 1) % cycle_n) for j in range(cycle_n)]
        edges.extend((k - 1, k) for k in range(cycle_n, n_views))
        return _spec(name, n_views, anchor_view, edges, family="cycle_plus_tails", expected_bipartite=False, expected_rank_deficient=False)
    if name == "triangle_plus_tails":
        if n_views < 3:
            return _invalid_spec(name, n_views, anchor_view, "triangle_plus_tails requires n_views >= 3.", "triangle_plus_tails")
        edges = [(0, 1), (1, 2), (0, 2)]
        edges.extend((k - 1, k) for k in range(3, n_views))
        return _spec(
            name,
            n_views,
            anchor_view,
            edges,
            family="triangle_plus_tails",
            expected_bipartite=False,
            expected_rank_deficient=False,
        )
    if name == "disconnected":
        if n_views < 4:
            return _invalid_spec(name, n_views, anchor_view, "disconnected requires n_views >= 4.", "disconnected")
        split = max(2, n_views // 2)
        edges = [(anchor_view, 1 if anchor_view != 1 else 0)]
        remaining = [view for view in range(split, n_views)]
        if len(remaining) >= 2:
            edges.extend((remaining[idx], remaining[idx + 1]) for idx in range(len(remaining) - 1))
        else:
            edges.append((2, 3))
        return _spec(name, n_views, anchor_view, edges, family="disconnected", expected_bipartite=True, expected_rank_deficient=True)
    if name == "random_erdos_renyi":
        rng = np.random.default_rng(seed)
        edges = [(j, k) for j in range(n_views) for k in range(j + 1, n_views) if rng.random() < density]
        edges.extend((j, j + 1) for j in range(n_views - 1))
        return _spec(name, n_views, anchor_view, edges, family="random", expected_bipartite=None, expected_rank_deficient=None)
    if name == "nearly_bipartite":
        if n_views < 3:
            return _invalid_spec(name, n_views, anchor_view, "nearly_bipartite requires n_views >= 3.", "nearly_bipartite")
        left = list(range(0, n_views, 2))
        right = list(range(1, n_views, 2))
        edges = [(j, k) for j in left for k in right if j != k]
        edges.append((0, 2))
        return _spec(name, n_views, anchor_view, edges, family="nearly_bipartite", expected_bipartite=False, expected_rank_deficient=False)
    raise ValueError(f"Unknown graph design: {name}")


def validate_observed_graph_matches_design(
    data: IndexedData,
    design_spec: GraphDesignSpec,
    min_shared_items: int,
) -> dict[str, object]:
    diagnostic = build_coobservation_graph(data, min_shared_items=min_shared_items)
    intended = set(tuple(sorted(edge)) for edge in design_spec.edges)
    observed = set(tuple(sorted(edge)) for edge in diagnostic.edges)
    missing = sorted(intended - observed)
    extra = sorted(observed - intended)
    return {
        "matches": not missing and not extra,
        "intended_edges": sorted(intended),
        "observed_edges": sorted(observed),
        "missing_edges": missing,
        "extra_edges": extra,
        "diagnostic": diagnostic,
    }


def _theta_values(rng: np.random.Generator, n_items: int, distribution: str) -> np.ndarray:
    if distribution == "normal":
        return rng.normal(0.0, 1.0, size=n_items)
    if distribution == "t":
        return rng.standard_t(df=3, size=n_items) / math.sqrt(3.0)
    if distribution == "mixture":
        component = rng.random(n_items) < 0.2
        theta = rng.normal(0.0, 0.45, size=n_items)
        theta[component] = rng.normal(0.0, 2.0, size=int(np.sum(component)))
        return theta
    if distribution == "skewed":
        return rng.exponential(1.0, size=n_items) - 1.0
    if distribution == "bounded_logit":
        return 2.0 * expit(rng.normal(0.0, 1.5, size=n_items)) - 1.0
    raise ValueError(f"Unknown theta distribution: {distribution}")


def _apply_missingness(
    rng: np.random.Generator,
    rows: list[dict[str, object]],
    theta: np.ndarray,
    *,
    mode: str,
) -> list[dict[str, object]]:
    if mode == "none":
        return rows
    kept = []
    for row in rows:
        item_idx = int(str(row["item_id"]).split("_")[-1])
        view_idx = int(str(row["view_id"]).split("_")[-1])
        if mode == "MCAR":
            drop_prob = 0.08
        elif mode == "MAR":
            drop_prob = 0.03 + 0.08 * (view_idx % 3 == 0)
        elif mode == "MNAR":
            drop_prob = 0.03 + 0.18 * (abs(theta[item_idx]) > np.quantile(np.abs(theta), 0.75))
        else:
            raise ValueError(f"Unknown missingness mode: {mode}")
        if rng.random() > drop_prob:
            kept.append(row)
    return kept


def simulate_graph_synthetic(
    *,
    graph_design: str,
    n_items: int = 1000,
    n_views: int = 5,
    n_features: int = 6,
    theta_distribution: str = "normal",
    tau_ratio: float = 0.5,
    loading_spread: float = 0.4,
    missingness: str = "none",
    view_bias: bool = False,
    bounded_transform: bool = False,
    seed: int = 0,
    er_density: float = 0.35,
) -> GraphSyntheticDataset:
    rng = np.random.default_rng(seed)
    design_spec = make_graph_design(graph_design, n_views, anchor_view=0, density=er_density, seed=seed)
    if not design_spec.valid:
        raise ValueError(f"Invalid graph design {graph_design}/{n_views}: {design_spec.skip_reason}")
    edges = design_spec.edges
    if not edges:
        raise ValueError("Graph design produced no edges.")
    item_ids = np.array([f"item_{idx:06d}" for idx in range(n_items)])
    view_ids = np.array([f"view_{idx:02d}" for idx in range(n_views)])
    theta = _theta_values(rng, n_items, theta_distribution)
    if bounded_transform:
        theta = 2.0 * expit(theta) - 1.0
    features = rng.normal(size=(n_items, max(n_features, 1)))
    features[:, 0] = 1.0
    prior_mean = np.zeros(n_items, dtype=np.float64)
    prior_var = np.full(n_items, float(np.var(theta) + 1e-6), dtype=np.float64)
    a = np.exp(rng.normal(0.0, loading_spread, size=n_views))
    a[0] = 1.0
    b = rng.normal(0.0, 0.15, size=n_views) if view_bias else np.zeros(n_views, dtype=np.float64)
    b[0] = 0.0
    mean_s = rng.uniform(0.12, 0.35, size=n_views)
    tau_multiplier = rng.uniform(0.7, 1.3, size=n_views)
    tau = tau_ratio * mean_s * tau_multiplier
    if graph_design == "nearly_bipartite" and edges:
        tau[-1] = max(tau[-1], 2.0 * float(np.mean(mean_s)))

    edge_counts = np.full(len(edges), n_items // len(edges), dtype=np.int64)
    edge_counts[: n_items % len(edges)] += 1
    item_edge = np.repeat(np.arange(len(edges)), edge_counts)[:n_items]
    rng.shuffle(item_edge)
    rows: list[dict[str, object]] = []
    for item_idx, edge_idx in enumerate(item_edge):
        for view_idx in edges[int(edge_idx)]:
            s = float(max(0.03, rng.lognormal(np.log(mean_s[view_idx]), 0.25)))
            row_tau = tau[view_idx]
            if graph_design == "nearly_bipartite" and tuple(edges[int(edge_idx)]) == (0, 2):
                row_tau = max(row_tau, 3.0 * s)
            y = rng.normal(b[view_idx] + a[view_idx] * theta[item_idx], math.sqrt(s * s + row_tau * row_tau))
            rows.append(
                {
                    "item_id": item_ids[item_idx],
                    "view_id": view_ids[view_idx],
                    "estimate": float(y),
                    "standard_error": s,
                }
            )
    rows = _apply_missingness(rng, rows, theta, mode=missingness)
    observations = pd.DataFrame(rows)
    item_features = pd.DataFrame(features, columns=[f"x{idx:02d}" for idx in range(features.shape[1])])
    item_features.insert(0, "item_id", item_ids)
    view_metadata = pd.DataFrame({"view_id": view_ids, "is_reference_view": [idx == 0 for idx in range(n_views)]})
    data = build_indexed_data(prepare_data(observations, item_features, view_metadata))
    truth = pd.DataFrame(
        {
            "item_id": item_ids,
            "theta": theta,
            "theta_true": theta,
            "prior_mean_true": prior_mean,
            "prior_var_true": prior_var,
        }
    )
    view_parameters = pd.DataFrame(
        {"view_id": view_ids, "a_true": a, "b_true": b, "tau_true": tau, "mean_s": mean_s}
    )
    metadata = {
        "graph_design": graph_design,
        "n_items_requested": int(n_items),
        "n_views": int(n_views),
        "theta_distribution": theta_distribution,
        "tau_ratio": float(tau_ratio),
        "loading_spread": float(loading_spread),
        "missingness": missingness,
        "bounded_transform": bool(bounded_transform),
        "intended_edges": edges,
        "design_spec": design_spec,
    }
    return GraphSyntheticDataset(data, truth, view_parameters, edges, metadata)


def _dataset_from_rows(
    *,
    rows: list[dict[str, object]],
    item_ids: np.ndarray,
    view_ids: np.ndarray,
    features: np.ndarray,
    theta: np.ndarray,
    prior_mean: np.ndarray,
    prior_var: np.ndarray,
    view_parameters: pd.DataFrame,
    intended_edges: list[tuple[int, int]],
    metadata: dict[str, object],
) -> GraphSyntheticDataset:
    observations = pd.DataFrame(rows)
    item_features = pd.DataFrame(features, columns=[f"x{idx:02d}" for idx in range(features.shape[1])])
    item_features.insert(0, "item_id", item_ids)
    view_metadata = pd.DataFrame({"view_id": view_ids, "is_reference_view": [idx == 0 for idx in range(len(view_ids))]})
    data = build_indexed_data(prepare_data(observations, item_features, view_metadata))
    truth = pd.DataFrame(
        {
            "item_id": item_ids,
            "theta": theta,
            "theta_true": theta,
            "prior_mean_true": prior_mean,
            "prior_var_true": prior_var,
        }
    )
    return GraphSyntheticDataset(data, truth, view_parameters.copy(), intended_edges, metadata)


def _paired_intervention_spec(intervention: str, n_views: int) -> PairedInterventionSpec:
    if intervention == "star_to_star_plus_edge":
        if n_views < 3:
            raise ValueError("star_to_star_plus_edge requires n_views >= 3.")
        base = make_graph_design("star", n_views)
        plus = make_graph_design("star_plus_edge_closes_triangle", n_views)
        return PairedInterventionSpec(
            name=intervention,
            base_design_name="star",
            plus_design_name="star_plus_edge_closes_triangle",
            base_edges=base.edges,
            plus_edges=plus.edges,
            additions=[((0, 1), 2)],
        )
    if intervention == "chain_to_triangle_plus_tail":
        if n_views < 3:
            raise ValueError("chain_to_triangle_plus_tail requires n_views >= 3.")
        base = make_graph_design("chain", n_views)
        plus = make_graph_design("triangle_plus_tails", n_views)
        return PairedInterventionSpec(
            name=intervention,
            base_design_name="chain",
            plus_design_name="triangle_plus_tails",
            base_edges=base.edges,
            plus_edges=plus.edges,
            additions=[((1, 2), 0)],
        )
    if intervention == "even_cycle_to_odd_cycle_bridge":
        if n_views < 4 or n_views % 2:
            raise ValueError("even_cycle_to_odd_cycle_bridge requires even n_views >= 4.")
        base = make_graph_design("even_cycle", n_views)
        plus_edges = sorted(set(base.edges + [(0, 2)]))
        return PairedInterventionSpec(
            name=intervention,
            base_design_name="even_cycle",
            plus_design_name="even_cycle_plus_odd_bridge",
            base_edges=base.edges,
            plus_edges=plus_edges,
            additions=[((0, 1), 2)],
        )
    if intervention == "disconnected_to_connected_with_odd_cycle":
        if n_views < 4:
            raise ValueError("disconnected_to_connected_with_odd_cycle requires n_views >= 4.")
        base_edges = [(0, 1), (2, 3)]
        base_edges.extend((view - 1, view) for view in range(4, n_views))
        plus_edges = sorted(set(base_edges + [(0, 2), (1, 2)]))
        return PairedInterventionSpec(
            name=intervention,
            base_design_name="disconnected",
            plus_design_name="connected_with_odd_cycle",
            base_edges=sorted(set(base_edges)),
            plus_edges=plus_edges,
            additions=[((0, 1), 2)],
        )
    raise ValueError(f"Unknown paired graph intervention: {intervention}")


def generate_paired_graph_intervention(
    *,
    n_items: int,
    n_views: int,
    intervention: str = "star_to_star_plus_edge",
    seed: int = 0,
    tau_ratio: float = 0.5,
    loading_spread: float = 0.4,
    n_items_per_edge: int = 40,
    budget_mode: str = "both",
    theta_distribution: str = "normal",
) -> PairedGraphIntervention:
    intervention_spec = _paired_intervention_spec(intervention, n_views)
    rng = np.random.default_rng(seed)
    base_edges = intervention_spec.base_edges
    plus_edges = intervention_spec.plus_edges
    total_items = n_items_per_edge * len(base_edges)
    item_ids = np.array([f"item_{idx:06d}" for idx in range(total_items)])
    view_ids = np.array([f"view_{idx:02d}" for idx in range(n_views)])
    theta = _theta_values(rng, total_items, theta_distribution)
    features = rng.normal(size=(total_items, 6))
    features[:, 0] = 1.0
    prior_mean = np.zeros(total_items, dtype=np.float64)
    prior_var = np.full(total_items, float(np.var(theta) + 1e-6), dtype=np.float64)
    a = np.exp(rng.normal(0.0, loading_spread, size=n_views))
    a[0] = 1.0
    b = np.zeros(n_views, dtype=np.float64)
    mean_s = rng.uniform(0.12, 0.35, size=n_views)
    tau = tau_ratio * mean_s * rng.uniform(0.7, 1.3, size=n_views)
    view_parameters = pd.DataFrame({"view_id": view_ids, "a_true": a, "b_true": b, "tau_true": tau, "mean_s": mean_s})

    edge_item_indices: dict[tuple[int, int], np.ndarray] = {}
    cursor = 0
    for edge in base_edges:
        edge_item_indices[edge] = np.arange(cursor, cursor + n_items_per_edge, dtype=np.int64)
        cursor += n_items_per_edge

    noise_by_item_view: dict[tuple[int, int], tuple[float, float]] = {}

    def measurement(item_idx: int, view_idx: int) -> tuple[float, float]:
        key = (item_idx, view_idx)
        if key not in noise_by_item_view:
            s = float(max(0.03, rng.lognormal(np.log(mean_s[view_idx]), 0.25)))
            y = rng.normal(b[view_idx] + a[view_idx] * theta[item_idx], math.sqrt(s * s + tau[view_idx] * tau[view_idx]))
            noise_by_item_view[key] = (float(y), s)
        return noise_by_item_view[key]

    def rows_for_star() -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for edge, items in edge_item_indices.items():
            for item_idx in items:
                for view_idx in edge:
                    y, s = measurement(int(item_idx), int(view_idx))
                    rows.append(
                        {
                            "item_id": item_ids[item_idx],
                            "view_id": view_ids[view_idx],
                            "estimate": y,
                            "standard_error": s,
                            "edge_block": f"{edge[0]}-{edge[1]}",
                        }
                    )
        return rows

    base_rows = rows_for_star()
    add_count = max(1, min(n_items_per_edge // 2, n_items_per_edge - 1))
    added_rows = []
    for source_edge, add_view in intervention_spec.additions:
        add_items = edge_item_indices[tuple(sorted(source_edge))][:add_count]
        for item_idx in add_items:
            y, s = measurement(int(item_idx), int(add_view))
            added_rows.append(
                {
                    "item_id": item_ids[item_idx],
                    "view_id": view_ids[add_view],
                    "estimate": y,
                    "standard_error": s,
                    "edge_block": f"{source_edge[0]}-{source_edge[1]}-plus-{add_view}",
                }
            )
    additive_rows = base_rows + added_rows
    budget_rows = list(base_rows)
    removal_edge = base_edges[-1]
    removal_view = removal_edge[1]
    removable = [
        idx
        for idx, row in enumerate(budget_rows)
        if row["edge_block"] == f"{removal_edge[0]}-{removal_edge[1]}" and row["view_id"] == view_ids[removal_view]
    ]
    for idx in sorted(removable[: len(added_rows)], reverse=True):
        budget_rows.pop(idx)
    budget_rows.extend(added_rows)

    pair_id = f"{intervention}_seed{seed}"
    common_meta = {
        "dataset_name": "paired_graph_intervention",
        "pair_id": pair_id,
        "intervention_name": intervention,
        "seed": int(seed),
        "tau_ratio": float(tau_ratio),
        "loading_spread": float(loading_spread),
        "n_items_per_edge": int(n_items_per_edge),
        "n_items_requested": int(n_items),
        "base_design_name": intervention_spec.base_design_name,
        "plus_design_name": intervention_spec.plus_design_name,
    }
    datasets = {
        "base": _dataset_from_rows(
            rows=base_rows,
            item_ids=item_ids,
            view_ids=view_ids,
            features=features,
            theta=theta,
            prior_mean=prior_mean,
            prior_var=prior_var,
            view_parameters=view_parameters,
            intended_edges=base_edges,
            metadata={**common_meta, "arm": "base", "intended_edges": base_edges},
        ),
        "plus_additive": _dataset_from_rows(
            rows=additive_rows,
            item_ids=item_ids,
            view_ids=view_ids,
            features=features,
            theta=theta,
            prior_mean=prior_mean,
            prior_var=prior_var,
            view_parameters=view_parameters,
            intended_edges=plus_edges,
            metadata={**common_meta, "arm": "plus_additive", "intended_edges": plus_edges},
        ),
        "plus_budget_matched": _dataset_from_rows(
            rows=budget_rows,
            item_ids=item_ids,
            view_ids=view_ids,
            features=features,
            theta=theta,
            prior_mean=prior_mean,
            prior_var=prior_var,
            view_parameters=view_parameters,
            intended_edges=plus_edges,
            metadata={**common_meta, "arm": "plus_budget_matched", "intended_edges": plus_edges},
        ),
    }
    if budget_mode == "additive":
        datasets = {key: value for key, value in datasets.items() if key in {"base", "plus_additive"}}
    elif budget_mode == "budget_matched":
        datasets = {key: value for key, value in datasets.items() if key in {"base", "plus_budget_matched"}}
    elif budget_mode != "both":
        raise ValueError("budget_mode must be additive, budget_matched, or both.")
    return PairedGraphIntervention(datasets=datasets, true_theta=theta.copy(), true_view_parameters=view_parameters.copy(), metadata=common_meta)


def _graph_conditioning_metrics(edges: list[tuple[int, int]], n_views: int, anchor_view: int = 0) -> dict[str, float]:
    matrix = edge_design_matrix(edges, n_views, anchor_view)
    conditioning = full_column_singular_values(matrix)
    return {
        "smallest_full_singular_value": float(conditioning["smallest_full_singular_value"]),
        "smallest_singular_value": float(conditioning["smallest_full_singular_value"]),
        "largest_singular_value": float(conditioning["largest_singular_value"]),
        "condition_number": float(conditioning["condition_number"]),
        "normalized_smallest_full_singular_value": float(conditioning["normalized_smallest_full_singular_value"]),
        "normalized_condition_number": float(conditioning["normalized_condition_number"]),
    }


def _pairwise_scores(
    posterior: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    close_quantile: float = 0.30,
) -> dict[str, float]:
    merged = posterior.merge(truth.loc[:, ["item_id", "theta"]], on="item_id", how="inner")
    mean = merged["posterior_mean"].to_numpy(dtype=np.float64)
    var = np.maximum(merged["posterior_var"].to_numpy(dtype=np.float64), 1e-18)
    theta = merged["theta"].to_numpy(dtype=np.float64)
    if mean.size < 2:
        return {"close_pair_ece": np.nan, "pairwise_brier": np.nan, "pairwise_log_score": np.nan}
    first, second = np.triu_indices(mean.size, k=1)
    gap = theta[first] - theta[second]
    nonzero = np.abs(gap) > 1e-12
    if not np.any(nonzero):
        close = np.ones_like(gap, dtype=bool)
    else:
        threshold = float(np.quantile(np.abs(gap[nonzero]), close_quantile))
        close = (np.abs(gap) <= threshold) & nonzero
    denom = np.sqrt(np.maximum(var[first] + var[second], 1e-18))
    prob = np.asarray(0.5 * (1.0 + np.vectorize(math.erf)((mean[first] - mean[second]) / (denom * math.sqrt(2.0)))))
    observed = (gap > 0.0).astype(float)
    prob = np.clip(prob, 1e-8, 1.0 - 1e-8)
    use = close if np.any(close) else np.ones_like(close, dtype=bool)
    return {
        "close_pair_ece": float(np.mean(np.abs(prob[use] - observed[use]))),
        "pairwise_brier": float(np.mean(np.square(prob - observed))),
        "pairwise_log_score": float(-np.mean(observed * np.log(prob) + (1.0 - observed) * np.log(1.0 - prob))),
    }


def _method_parameter_errors(diagnostics: dict[str, object], view_parameters: pd.DataFrame) -> dict[str, float]:
    true_a = view_parameters["a_true"].to_numpy(dtype=np.float64)
    true_tau = view_parameters["tau_true"].to_numpy(dtype=np.float64)
    view_scale = np.asarray(diagnostics.get("view_scale", np.full_like(true_a, np.nan)), dtype=np.float64)
    extra_noise = np.asarray(diagnostics.get("extra_noise", np.full_like(true_tau, np.nan)), dtype=np.float64)
    if view_scale.shape != true_a.shape:
        view_scale = np.full_like(true_a, np.nan)
    if extra_noise.shape != true_tau.shape:
        extra_noise = np.full_like(true_tau, np.nan)
    def rmse(values: np.ndarray) -> float:
        return float(np.sqrt(np.nanmean(np.square(values)))) if np.isfinite(values).any() else np.nan
    return {
        "loading_rmse": rmse(view_scale - true_a),
        "log_loading_rmse": rmse(np.log(np.maximum(view_scale, 1e-12)) - np.log(np.maximum(true_a, 1e-12))),
        "tau_rmse": rmse(extra_noise - true_tau),
    }


def run_paired_graph_intervention(
    *,
    output_dir: str | Path,
    seeds: list[int] | None = None,
    n_items: int = 200,
    n_views: int = 5,
    intervention: str = "star_to_star_plus_edge",
    tau_ratio: float = 0.5,
    loading_spread: float = 0.4,
    n_items_per_edge: int = 40,
    budget_mode: str = "both",
    max_iter: int = 20,
    method_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = seeds or [0, 1, 2]
    method_names = method_names or ["joint_eb_full", "anchored_calibrated_eb", "anchored_calibrated_eb_in_sample"]
    metric_rows: list[dict[str, object]] = []
    for seed in seeds:
        paired = generate_paired_graph_intervention(
            n_items=n_items,
            n_views=n_views,
            intervention=intervention,
            seed=int(seed),
            tau_ratio=tau_ratio,
            loading_spread=loading_spread,
            n_items_per_edge=n_items_per_edge,
            budget_mode=budget_mode,
        )
        for arm, dataset in paired.datasets.items():
            graph = build_coobservation_graph(dataset.data, min_shared_items=1).to_summary_dict()
            conditioning = _graph_conditioning_metrics(dataset.intended_edges, dataset.data.n_views)
            for method_name, posterior, diagnostics, runtime in run_graph_methods(
                dataset,
                max_iter=max_iter,
                include_mash=False,
                include_bootstrap=False,
                strict=False,
                method_names=method_names,
                include_oracles=False,
            ):
                base_metrics = compute_metrics(
                    data=dataset.data,
                    truth=dataset.truth,
                    posterior=posterior,
                    diagnostics=diagnostics,
                    size=str(dataset.data.n_items),
                    scenario=arm,
                    method_name=method_name,
                    runtime_seconds=runtime,
                    true_view_parameters=dataset.view_parameters,
                )
                pairwise_scores = _pairwise_scores(posterior, dataset.truth)
                parameter_errors = _method_parameter_errors(diagnostics, dataset.view_parameters)
                mean_width = float(2.0 * 1.6448536269514722 * posterior["posterior_sd"].mean())
                metric_rows.append(
                    {
                        **base_metrics,
                        **pairwise_scores,
                        **parameter_errors,
                        "dataset_name": dataset.metadata["dataset_name"],
                        "pair_id": dataset.metadata["pair_id"],
                        "intervention_name": intervention,
                        "arm": arm,
                        "seed": int(seed),
                        "delta_G": graph["rank_deficiency_delta"],
                        **conditioning,
                        "interval_width": mean_width,
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    deltas = paired_intervention_deltas(metrics)
    metrics.to_csv(output_dir / "intervention_metrics.csv", index=False)
    deltas.to_csv(output_dir / "intervention_deltas.csv", index=False)
    write_intervention_summary(metrics, deltas, output_dir / "intervention_summary.md")
    write_intervention_plots(deltas, output_dir / "plots")
    return metrics, deltas


def paired_intervention_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    delta_metrics = [
        "delta_G",
        "smallest_full_singular_value",
        "normalized_smallest_full_singular_value",
        "condition_number",
        "normalized_condition_number",
        "min_singular_value",
        "loading_rmse",
        "log_loading_rmse",
        "tau_rmse",
        "coverage_90",
        "false_best_rate",
        "close_pair_ece",
        "pairwise_brier_score",
        "close_pair_brier_score",
        "close_pair_log_score",
        "pairwise_brier",
        "pairwise_log_score",
        "interval_width",
        "coverage_width_score",
    ]
    metrics = metrics.assign(z_score_sd_distance=lambda frame: np.abs(frame["posterior_z_sd"] - 1.0))
    delta_metrics.append("z_score_sd_distance")
    delta_metrics = [metric for metric in delta_metrics if metric in metrics.columns]
    condition_metrics = [metric for metric in ["condition_number", "normalized_condition_number"] if metric in delta_metrics]
    delta_metrics = [metric for metric in delta_metrics if metric not in condition_metrics]
    base = metrics.loc[metrics["arm"] == "base"]
    for _, base_row in base.iterrows():
        mask = (
            (metrics["pair_id"] == base_row["pair_id"])
            & (metrics["seed"] == base_row["seed"])
            & (metrics["method_name"] == base_row["method_name"])
            & (metrics["arm"] != "base")
        )
        for _, plus_row in metrics.loc[mask].iterrows():
            row = {
                "pair_id": base_row["pair_id"],
                "seed": int(base_row["seed"]),
                "method_name": base_row["method_name"],
                "plus_arm": plus_row["arm"],
            }
            for metric in delta_metrics:
                row[f"delta_{metric}"] = float(plus_row[metric] - base_row[metric])
            for metric in condition_metrics:
                base_value = float(base_row[metric])
                plus_value = float(plus_row[metric])
                if np.isinf(base_value) and np.isfinite(plus_value):
                    transition = "infinite_to_finite"
                    improved = True
                    delta = np.nan
                elif np.isinf(base_value) and np.isinf(plus_value):
                    transition = "infinite_to_infinite"
                    improved = False
                    delta = np.nan
                elif np.isfinite(base_value) and np.isfinite(plus_value):
                    transition = "finite_to_finite"
                    delta = plus_value - base_value
                    improved = bool(delta < 0.0)
                elif np.isfinite(base_value) and np.isinf(plus_value):
                    transition = "finite_to_infinite"
                    improved = False
                    delta = np.nan
                else:
                    transition = "missing"
                    improved = False
                    delta = np.nan
                row[f"delta_{metric}"] = delta
                row[f"{metric}_transition"] = transition
                row[f"{metric}_improved"] = improved
            rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 1000) -> tuple[float, float]:
    if values.size == 0:
        return np.nan, np.nan
    draws = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def write_intervention_summary(metrics: pd.DataFrame, deltas: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    rng = np.random.default_rng(0)
    lines = ["# Paired Graph Intervention Summary", ""]
    lines.append(f"Rows: {len(metrics)} metrics, {len(deltas)} paired deltas")
    lines.append("")
    if deltas.empty:
        lines.append("No paired deltas were available.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    target_metrics = ["delta_coverage_90", "delta_close_pair_ece", "delta_pairwise_brier", "delta_false_best_rate"]
    lines.extend(["| method | plus_arm | metric | mean_delta | se | ci_low | ci_high | sign_consistency |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for (method, arm), group in deltas.groupby(["method_name", "plus_arm"]):
        for metric in target_metrics:
            values = group[metric].dropna().to_numpy(dtype=np.float64)
            if values.size == 0:
                continue
            ci_low, ci_high = _bootstrap_ci(values, rng)
            improves_negative = metric in {"delta_close_pair_ece", "delta_pairwise_brier", "delta_false_best_rate"}
            sign_consistency = np.mean(values < 0.0) if improves_negative else np.mean(values > 0.0)
            lines.append(
                f"| {method} | {arm} | {metric} | {values.mean():.4g} | {values.std(ddof=1) / np.sqrt(max(values.size, 1)) if values.size > 1 else 0.0:.4g} | {ci_low:.4g} | {ci_high:.4g} | {sign_consistency:.3f} |"
            )
    transition_cols = [col for col in ["condition_number_transition", "normalized_condition_number_transition"] if col in deltas]
    if transition_cols:
        lines.extend(["", "## Condition Number Transitions", ""])
        lines.append("| method | plus_arm | metric | frac_infinite_to_finite | frac_finite_improved | frac_finite_worsened |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: |")
        for (method, arm), group in deltas.groupby(["method_name", "plus_arm"]):
            for transition_col in transition_cols:
                metric = transition_col.replace("_transition", "")
                transition = group[transition_col].astype(str)
                finite = transition == "finite_to_finite"
                improved = group.get(f"{metric}_improved", pd.Series(False, index=group.index)).astype(bool)
                denom = max(len(group), 1)
                lines.append(
                    f"| {method} | {arm} | {metric} | "
                    f"{float(np.mean(transition == 'infinite_to_finite')):.3f} | "
                    f"{float(np.sum(finite & improved) / denom):.3f} | "
                    f"{float(np.sum(finite & ~improved) / denom):.3f} |"
                )
    aceb = deltas.loc[(deltas["method_name"] == "anchored_calibrated_eb") & (deltas["plus_arm"] == "plus_additive")]
    if not aceb.empty and "delta_close_pair_ece" in aceb:
        support = float(np.mean(aceb["delta_close_pair_ece"] < 0.0))
        if support >= 0.8:
            lines.append("\nACEB close-pair ECE improvement is supported across seeds.")
        else:
            lines.append("\nACEB close-pair ECE improvement is not supported strongly enough to claim.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_intervention_plots(deltas: pd.DataFrame, plots_dir: str | Path) -> None:
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    for metric in ["delta_coverage_90", "delta_close_pair_ece", "delta_pairwise_brier", "delta_false_best_rate"]:
        path = plots_dir / f"star_vs_plus_{metric}.svg"
        if deltas.empty or metric not in deltas:
            path.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>\n', encoding="utf-8")
            continue
        summary = deltas.groupby(["method_name", "plus_arm"], as_index=False)[metric].mean()
        values = summary[metric].to_numpy(dtype=np.float64)
        max_abs = max(float(np.max(np.abs(values))) if values.size else 1.0, 1e-9)
        width = max(720, 28 * len(values) + 160)
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="360">']
        parts.append(f'<text x="50" y="28" font-size="16" font-family="sans-serif">{metric}</text>')
        parts.append('<line x1="60" y1="180" x2="680" y2="180" stroke="black"/>')
        for idx, row in summary.iterrows():
            x = 80 + idx * 28
            val = float(row[metric])
            h = 120 * abs(val) / max_abs
            y = 180 - h if val >= 0 else 180
            color = "#4C78A8" if val >= 0 else "#E45756"
            parts.append(f'<rect x="{x}" y="{y:.2f}" width="18" height="{h:.2f}" fill="{color}"/>')
            label = f"{row['method_name'][:8]}:{row['plus_arm'].replace('plus_', '')[:6]}"
            parts.append(f'<text x="{x}" y="320" font-size="8" font-family="sans-serif" transform="rotate(60 {x},320)">{label}</text>')
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")


def _oracle_output(
    data: IndexedData,
    dataset: GraphSyntheticDataset,
    *,
    use_true_a: bool,
    use_true_tau: bool,
    method_name: str,
) -> MethodOutput:
    prior_mean = dataset.truth["prior_mean_true"].to_numpy(dtype=np.float64)
    prior_var = dataset.truth["prior_var_true"].to_numpy(dtype=np.float64)
    view_scale = dataset.view_parameters["a_true"].to_numpy(dtype=np.float64) if use_true_a else np.ones(data.n_views)
    extra_noise = dataset.view_parameters["tau_true"].to_numpy(dtype=np.float64) if use_true_tau else np.zeros(data.n_views)
    view_bias = dataset.view_parameters["b_true"].to_numpy(dtype=np.float64)
    posterior = compute_posterior_from_prior(data, prior_mean, prior_var, view_scale, extra_noise, view_bias=view_bias)
    diagnostics = {
        "view_scale": view_scale,
        "extra_noise": extra_noise,
        "view_bias": view_bias,
        "success": True,
        "oracle_true_a": bool(use_true_a),
        "oracle_true_tau": bool(use_true_tau),
    }
    diagnostics["graph_diagnostic"] = build_coobservation_graph(data, min_shared_items=1).to_summary_dict()
    return MethodOutput(posterior_to_frame(data, posterior, method_name), diagnostics)


def anchored_in_sample_ablation(data: IndexedData, *, max_iter: int = 80) -> MethodOutput:
    output = anchored_calibrated_eb(data, max_iter=max_iter, residual_calibration="in_sample")
    output.posterior["method_name"] = "anchored_calibrated_eb_in_sample"
    output.diagnostics["residual_calibration_method"] = "in_sample"
    return output


def anchored_no_graph_ablation(data: IndexedData, *, max_iter: int = 80) -> MethodOutput:
    output = anchored_calibrated_eb(data, max_iter=max_iter, residual_calibration="leave_view")
    output.posterior["method_name"] = "anchored_calibrated_eb_no_graph"
    output.diagnostics.pop("graph_diagnostic", None)
    return output


BASE_METHODS: dict[str, Callable[..., MethodOutput]] = {
    "raw_reference": raw_single_reference,
    "inverse_variance_all_views": inverse_variance_all_views,
    "normal_eb_no_features": normal_eb_no_features,
    "feature_normal_eb_no_view_scale": feature_normal_eb_no_view_scale,
    "joint_eb_full": joint_eb_full,
    "anchored_calibrated_eb": anchored_calibrated_eb,
    "anchored_calibrated_eb_in_sample": anchored_in_sample_ablation,
    "anchored_calibrated_eb_no_graph": anchored_no_graph_ablation,
    "random_effects_meta_analysis": random_effects_meta_analysis,
    "adaptive_shrinkage_like": adaptive_shrinkage_like,
    "posthoc_inverse_variance_all_views": posthoc_inverse_variance_all_views,
    "full_bayes_hierarchical": full_bayes_hierarchical,
}


ACEB_COMPONENT_VARIANTS = {
    "aceb_full_leave_view",
    "aceb_in_sample_residual",
    "aceb_fixed_loadings_a1",
    "aceb_global_tau",
    "aceb_no_tau",
    "aceb_oracle_loadings",
    "aceb_oracle_tau",
    "aceb_oracle_loadings_oracle_tau",
}


ITER_METHODS = {
    "normal_eb_no_features",
    "feature_normal_eb_no_view_scale",
    "joint_eb_full",
    "anchored_calibrated_eb",
    "anchored_calibrated_eb_in_sample",
    "anchored_calibrated_eb_no_graph",
    "adaptive_shrinkage_like",
}


def _renamed_output(output: MethodOutput, method_name: str) -> MethodOutput:
    posterior = output.posterior.copy()
    posterior["method_name"] = method_name
    diagnostics = dict(output.diagnostics)
    diagnostics["reported_method_name"] = method_name
    return MethodOutput(posterior, diagnostics)


def _true_view_scale(dataset: GraphSyntheticDataset) -> np.ndarray:
    true_a = dataset.view_parameters["a_true"].to_numpy(dtype=np.float64).copy()
    if true_a.size and np.isfinite(true_a[0]) and abs(true_a[0]) > 1e-12:
        true_a = true_a / true_a[0]
    if true_a.size:
        true_a[0] = 1.0
    return true_a


def _component_aceb_output(dataset: GraphSyntheticDataset, *, method_name: str, max_iter: int = 80) -> MethodOutput:
    data = dataset.data
    if method_name == "aceb_full_leave_view":
        output = anchored_calibrated_eb(data, max_iter=max_iter, residual_calibration="leave_view")
        output = _renamed_output(output, method_name)
        output.diagnostics["component_ablation"] = "full_leave_view"
        return output
    if method_name == "aceb_in_sample_residual":
        output = anchored_calibrated_eb(data, max_iter=max_iter, residual_calibration="in_sample")
        output = _renamed_output(output, method_name)
        output.diagnostics["component_ablation"] = "in_sample_residual"
        output.diagnostics["residual_calibration_method"] = "in_sample"
        return output

    view_scale = initialize_view_scale(data)
    extra_noise_init = initialize_extra_noise(data)
    history: list[dict[str, float]] = []
    w_mu, w_v, obj = fit_prior_hyperparameters(data, view_scale, extra_noise_init, max_iter=max_iter)
    history.append({"stage": 1.0, "objective": float(obj)})

    uses_oracle_loadings = method_name in {"aceb_oracle_loadings", "aceb_oracle_loadings_oracle_tau"}
    uses_oracle_tau = method_name in {"aceb_oracle_tau", "aceb_oracle_loadings_oracle_tau"}
    if uses_oracle_loadings:
        view_scale = _true_view_scale(dataset)
    elif method_name == "aceb_fixed_loadings_a1":
        view_scale = np.ones(data.n_views, dtype=np.float64)
    else:
        view_scale, obj = fit_non_reference_view_scales(
            data,
            w_mu,
            w_v,
            view_scale,
            extra_noise_init,
            max_iter=max_iter,
        )
        history.append({"stage": 2.0, "objective": float(obj)})

    # Refit the population prior after fixing/learning the loading scale.
    w_mu, w_v, obj = fit_prior_hyperparameters(data, view_scale, extra_noise_init, max_iter=max_iter)
    history.append({"stage": 2.5, "objective": float(obj)})
    prior_mean, prior_variance = prior_from_features(data.item_features, w_mu, w_v)
    view_bias = np.zeros(data.n_views, dtype=np.float64)

    if method_name == "aceb_no_tau":
        extra_noise = np.zeros(data.n_views, dtype=np.float64)
        tau_diag = {
            "residual_calibration_method": "forced_zero",
            "tau2_by_view": np.zeros(data.n_views, dtype=np.float64),
            "tau_by_view": np.zeros(data.n_views, dtype=np.float64),
        }
    elif uses_oracle_tau:
        extra_noise = dataset.view_parameters["tau_true"].to_numpy(dtype=np.float64).copy()
        tau_diag = {
            "residual_calibration_method": "oracle_tau",
            "tau2_by_view": extra_noise * extra_noise,
            "tau_by_view": extra_noise,
        }
    else:
        tau_result = estimate_tau2_leave_view(
            data,
            prior_mean,
            prior_variance,
            view_scale,
            extra_noise_init,
            bias=view_bias,
        )
        if method_name == "aceb_global_tau":
            weights = np.maximum(tau_result.n_rows_by_view.astype(np.float64), 0.0)
            if float(np.sum(weights)) > 0.0:
                tau2_global = float(np.sum(weights * tau_result.tau2_by_view) / np.sum(weights))
            else:
                tau2_global = float(np.mean(tau_result.tau2_by_view))
            extra_noise = np.full(data.n_views, math.sqrt(max(tau2_global, 0.0)), dtype=np.float64)
            tau_diag = tau_result.diagnostics()
            tau_diag.update(
                {
                    "residual_calibration_method": "leave_view_global_tau",
                    "global_tau": float(extra_noise[0]) if extra_noise.size else 0.0,
                    "tau2_by_view_before_global_pooling": tau_result.tau2_by_view,
                    "tau_by_view_before_global_pooling": tau_result.tau_by_view,
                    "tau2_by_view": extra_noise * extra_noise,
                    "tau_by_view": extra_noise,
                }
            )
        else:
            extra_noise = tau_result.tau_by_view
            tau_diag = tau_result.diagnostics()

    posterior = compute_posterior(data, w_mu, w_v, view_scale, extra_noise, view_bias=view_bias)
    diagnostics = {
        **tau_diag,
        "view_scale": view_scale,
        "extra_noise": extra_noise,
        "view_bias": view_bias,
        "history": history,
        "success": True,
        "component_ablation": method_name,
        "uses_oracle_loadings": bool(uses_oracle_loadings),
        "uses_oracle_tau": bool(uses_oracle_tau),
        "fixed_loadings_a1": bool(method_name == "aceb_fixed_loadings_a1"),
        "global_tau": bool(method_name == "aceb_global_tau"),
        "no_tau": bool(method_name == "aceb_no_tau"),
        "graph_diagnostic": build_coobservation_graph(data, min_shared_items=1).to_summary_dict(),
    }
    return MethodOutput(posterior_to_frame(data, posterior, method_name), diagnostics)


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
    return MethodOutput(
        posterior,
        {"failed": True, "failure_message": f"{type(exc).__name__}: {exc}", "success": False},
    )


def run_graph_methods(
    dataset: GraphSyntheticDataset,
    *,
    max_iter: int = 80,
    include_bootstrap: bool = False,
    include_mash: bool = True,
    strict: bool = False,
    method_names: list[str] | None = None,
    include_oracles: bool = True,
) -> list[tuple[str, pd.DataFrame, dict[str, object], float]]:
    methods = dict(BASE_METHODS)
    if include_bootstrap:
        methods["bootstrap_inverse_variance_all_views"] = bootstrap_inverse_variance_all_views
    if include_mash:
        methods["mashr_multivariate_eb"] = mashr_multivariate_eb
    if method_names is not None:
        known_oracles = {"oracle_true_a", "oracle_true_tau", "oracle_true_a_true_tau"}
        unknown = sorted(set(method_names) - set(methods) - known_oracles - ACEB_COMPONENT_VARIANTS)
        if unknown:
            raise ValueError(f"Unknown graph synthetic methods: {unknown}")
        methods = {name: methods[name] for name in method_names if name in methods}
    outputs: list[tuple[str, pd.DataFrame, dict[str, object], float]] = []
    component_method_names = [
        name for name in (method_names or []) if name in ACEB_COMPONENT_VARIANTS
    ]
    for name in component_method_names:
        start = time.perf_counter()
        try:
            output = _component_aceb_output(dataset, method_name=name, max_iter=max_iter)
        except Exception as exc:
            if strict:
                raise
            output = _failed_output(dataset.data, name, exc)
        outputs.append((name, output.posterior, output.diagnostics, time.perf_counter() - start))
    for name, method in methods.items():
        start = time.perf_counter()
        try:
            if name == "posthoc_inverse_variance_all_views":
                output = method(dataset.data, truth=dataset.truth)
            elif name == "full_bayes_hierarchical":
                output = method(dataset.data, iter_warmup=max(25, max_iter * 5), iter_sampling=max(25, max_iter * 5))
            elif name == "bootstrap_inverse_variance_all_views":
                output = method(dataset.data, B=5)
            elif name in ITER_METHODS:
                output = method(dataset.data, max_iter=max_iter)
            else:
                output = method(dataset.data)
        except Exception as exc:
            if strict:
                raise
            output = _failed_output(dataset.data, name, exc)
        outputs.append((name, output.posterior, output.diagnostics, time.perf_counter() - start))
    oracle_methods = {
        "oracle_true_a": {"use_true_a": True, "use_true_tau": False},
        "oracle_true_tau": {"use_true_a": False, "use_true_tau": True},
        "oracle_true_a_true_tau": {"use_true_a": True, "use_true_tau": True},
    }
    if method_names is not None:
        oracle_methods = {name: kwargs for name, kwargs in oracle_methods.items() if name in method_names}
    if not include_oracles:
        oracle_methods = {}
    for name, kwargs in oracle_methods.items():
        start = time.perf_counter()
        output = _oracle_output(dataset.data, dataset, method_name=name, **kwargs)
        outputs.append((name, output.posterior, output.diagnostics, time.perf_counter() - start))
    return outputs


def _svg_bar(frame: pd.DataFrame, metric: str, group: str, path: Path, title: str) -> None:
    if metric not in frame or group not in frame or "method_name" not in frame:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>\n", encoding="utf-8")
        return
    valid = frame.loc[np.isfinite(frame[metric].to_numpy(dtype=np.float64))].copy()
    if valid.empty:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>\n", encoding="utf-8")
        return
    summary = valid.groupby([group, "method_name"], as_index=False)[metric].mean()
    labels = [f"{row[group]}|{row['method_name']}"[:34] for _, row in summary.iterrows()]
    values = summary[metric].to_numpy(dtype=np.float64)
    width = max(900, 20 * len(values) + 160)
    height = 420
    max_value = max(float(np.nanmax(values)), 1e-9)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    parts.append(f'<text x="70" y="28" font-size="17" font-family="sans-serif">{title}</text>')
    for idx, value in enumerate(values):
        x = 90 + idx * 20
        bar_h = 280 * max(value, 0.0) / max_value
        y = 340 - bar_h
        parts.append(f'<rect x="{x}" y="{y:.2f}" width="14" height="{bar_h:.2f}" fill="#4C78A8"/>')
        if idx % max(1, len(values) // 30) == 0:
            parts.append(f'<text x="{x}" y="358" font-size="8" font-family="sans-serif" transform="rotate(55 {x},358)">{labels[idx]}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _svg_scatter(frame: pd.DataFrame, x_col: str, y_col: str, path: Path, title: str) -> None:
    if x_col not in frame or y_col not in frame:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>\n", encoding="utf-8")
        return
    valid = frame.loc[np.isfinite(frame[x_col].to_numpy(dtype=np.float64)) & np.isfinite(frame[y_col].to_numpy(dtype=np.float64))]
    width, height = 760, 460
    if valid.empty:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>\n", encoding="utf-8")
        return
    x = valid[x_col].to_numpy(dtype=np.float64)
    y = valid[y_col].to_numpy(dtype=np.float64)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    xmax = xmax if xmax > xmin else xmin + 1.0
    ymax = ymax if ymax > ymin else ymin + 1.0
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    parts.append(f'<text x="60" y="28" font-size="17" font-family="sans-serif">{title}</text>')
    parts.append('<line x1="70" y1="390" x2="720" y2="390" stroke="black"/>')
    parts.append('<line x1="70" y1="60" x2="70" y2="390" stroke="black"/>')
    for xi, yi in zip(x, y, strict=True):
        px = 70 + 650 * (xi - xmin) / (xmax - xmin)
        py = 390 - 330 * (yi - ymin) / (ymax - ymin)
        parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3" fill="#E45756" opacity="0.75"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_graph_plots(metrics: pd.DataFrame, pairwise: pd.DataFrame, output_dir: Path) -> None:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    _svg_bar(metrics, "coverage_90", "graph_design", plots / "coverage_vs_graph_design.svg", "Coverage vs graph design")
    _svg_bar(metrics, "false_best_rate", "graph_design", plots / "false_best_rate_vs_graph_design.svg", "False-best rate vs graph design")
    if "false_best_rate_095" in metrics:
        _svg_bar(metrics, "false_best_rate_095", "graph_design", plots / "false_best_rate_095_vs_graph_design.svg", "False-best rate at 0.95 vs graph design")
    if "close_pair_ece" in metrics:
        _svg_bar(metrics, "close_pair_ece", "graph_design", plots / "close_pair_ece_vs_graph_design.svg", "Close-pair ECE vs graph design")
    _svg_scatter(metrics, "delta_G", "posterior_z_sd", plots / "z_score_sd_vs_delta_G.svg", "Z-score SD vs delta(G)")
    if "loading_rmse" in metrics:
        _svg_scatter(metrics, "delta_G", "loading_rmse", plots / "loading_rmse_vs_delta_G.svg", "Loading RMSE vs delta(G)")
    if "coverage_width_score" in metrics:
        _svg_scatter(metrics, "rmse", "coverage_width_score", plots / "rmse_interval_score_tradeoff.svg", "RMSE vs interval score")
    _svg_scatter(metrics, "rmse", "coverage_90", plots / "rmse_coverage_tradeoff.svg", "RMSE vs coverage")
    if "graph_design" in pairwise:
        selected = pairwise.loc[pairwise["graph_design"].isin(["star", "star_plus_edge_closes_triangle", "odd_cycle"])]
    else:
        selected = pd.DataFrame()
    _svg_bar(selected.rename(columns={"mean_pred": "pairwise_mean_pred"}), "empirical_win_rate", "graph_design", plots / "pairwise_calibration_selected.svg", "Pairwise calibration")
    intervention = metrics.loc[metrics["graph_design"].isin(["star", "star_plus_edge_closes_triangle"])]
    _svg_bar(intervention, "coverage_90", "graph_design", plots / "graph_intervention_star_vs_triangle.svg", "Star vs triangle intervention")


def run_graph_synthetic_benchmark(
    *,
    output_dir: Path,
    graph_designs: list[str] | None = None,
    n_items_values: list[int] | None = None,
    n_views_values: list[int] | None = None,
    tau_ratios: list[float] | None = None,
    loading_spreads: list[float] | None = None,
    missingness_modes: list[str] | None = None,
    theta_distributions: list[str] | None = None,
    seed: int = 1,
    seeds: list[int] | None = None,
    max_iter: int = 60,
    include_bootstrap: bool = False,
    include_mash: bool = True,
    strict: bool = False,
    method_names: list[str] | None = None,
    include_oracles: bool = True,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_designs = graph_designs or GRAPH_DESIGNS
    n_items_values = n_items_values or [100]
    n_views_values = n_views_values or [5]
    tau_ratios = tau_ratios or [0.5]
    loading_spreads = loading_spreads or [0.4]
    missingness_modes = missingness_modes or ["none"]
    theta_distributions = theta_distributions or ["normal"]
    metrics_rows: list[dict[str, object]] = []
    pairwise_rows: list[pd.DataFrame] = []
    stratified_rows: list[pd.DataFrame] = []
    skipped_cells: list[dict[str, object]] = []
    method_diagnostics: dict[str, object] = {}
    graph_diagnostics: dict[str, object] = {}
    seeds = seeds or [seed]
    run_idx = 0
    for seed_value in seeds:
        for graph_design in graph_designs:
            for n_items in n_items_values:
                for n_views in n_views_values:
                    for tau_ratio in tau_ratios:
                        for loading_spread in loading_spreads:
                            for missingness in missingness_modes:
                                for theta_distribution in theta_distributions:
                                    run_seed = int(seed_value) + run_idx * 997
                                    run_idx += 1
                                    design_spec = make_graph_design(graph_design, n_views, anchor_view=0, seed=run_seed)
                                    if not design_spec.valid:
                                        skipped_cells.append(
                                            {
                                                "graph_design": graph_design,
                                                "n_views_config": n_views,
                                                "seed": int(seed_value),
                                                "run_seed": int(run_seed),
                                                "valid": False,
                                                "skip_reason": design_spec.skip_reason,
                                                "design_family": design_spec.design_family,
                                            }
                                        )
                                        graph_diagnostics[f"{graph_design}:seed{seed_value}:run{run_idx}"] = {
                                            "intended_edges": [],
                                            "observed_edges": [],
                                            "design_valid": False,
                                            "skip_reason": design_spec.skip_reason,
                                            "design_family": design_spec.design_family,
                                        }
                                        continue
                                    dataset = simulate_graph_synthetic(
                                        graph_design=graph_design,
                                        n_items=n_items,
                                        n_views=n_views,
                                        tau_ratio=tau_ratio,
                                        loading_spread=loading_spread,
                                        missingness=missingness,
                                        theta_distribution=theta_distribution,
                                        seed=run_seed,
                                    )
                                    graph = build_coobservation_graph(dataset.data, min_shared_items=1).to_summary_dict()
                                    validation = validate_observed_graph_matches_design(dataset.data, design_spec, min_shared_items=1)
                                    if not validation["matches"]:
                                        message = (
                                            f"Observed graph mismatch for {graph_design}/{n_views}: "
                                            f"missing={validation['missing_edges']} extra={validation['extra_edges']}"
                                        )
                                        if strict:
                                            raise ValueError(message)
                                        skipped_cells.append(
                                            {
                                                "graph_design": graph_design,
                                                "n_views_config": n_views,
                                                "seed": int(seed_value),
                                                "run_seed": int(run_seed),
                                                "valid": False,
                                                "skip_reason": message,
                                                "design_family": design_spec.design_family,
                                            }
                                        )
                                        continue
                                    graph_diagnostics[f"{graph_design}:seed{seed_value}:run{run_idx}"] = {
                                        **graph,
                                        "intended_edges": validation["intended_edges"],
                                        "observed_edges": validation["observed_edges"],
                                        "design_valid": True,
                                        "skip_reason": None,
                                        "design_family": design_spec.design_family,
                                        "expected_bipartite": design_spec.expected_bipartite,
                                        "expected_has_odd_cycle": design_spec.expected_has_odd_cycle,
                                        "expected_rank_deficient": design_spec.expected_rank_deficient,
                                    }
                                    for method_name, posterior, diagnostics, runtime in run_graph_methods(
                                        dataset,
                                        max_iter=max_iter,
                                        include_bootstrap=include_bootstrap,
                                        include_mash=include_mash,
                                        strict=strict,
                                        method_names=method_names,
                                        include_oracles=include_oracles,
                                    ):
                                        method_diagnostics[f"{graph_design}:seed{seed_value}:run{run_idx}:{method_name}"] = diagnostics
                                        common = {
                                            "graph_design": graph_design,
                                            "n_views_config": n_views,
                                            "tau_ratio": tau_ratio,
                                            "loading_spread": loading_spread,
                                            "missingness": missingness,
                                            "theta_distribution": theta_distribution,
                                            "seed": int(seed_value),
                                            "run_seed": int(run_seed),
                                            "delta_G": graph["rank_deficiency_delta"],
                                            "has_odd_cycle": graph["anchor_component_is_non_bipartite"],
                                            "n_graph_edges": graph["n_edges"],
                                            "graph_design_rank": graph.get("graph_design_rank", graph.get("signless_incidence_rank", np.nan)),
                                            "graph_design_num_columns": graph.get("graph_design_num_columns", np.nan),
                                            "smallest_full_singular_value": graph.get("smallest_full_singular_value", graph.get("smallest_singular_value", np.nan)),
                                            "smallest_singular_value": graph.get("smallest_full_singular_value", graph.get("smallest_singular_value", np.nan)),
                                            "largest_singular_value": graph.get("largest_singular_value", np.nan),
                                            "condition_number": graph.get("condition_number", np.nan),
                                            "normalized_smallest_full_singular_value": graph.get("normalized_smallest_full_singular_value", np.nan),
                                            "normalized_condition_number": graph.get("normalized_condition_number", np.nan),
                                            "component_count": graph.get("component_count", graph.get("n_components", np.nan)),
                                            "anchor_component_bipartite": graph.get("anchor_component_bipartite", np.nan),
                                            "anchor_component_has_odd_cycle": graph.get("anchor_component_has_odd_cycle", np.nan),
                                        }
                                        if diagnostics.get("skipped", False) or diagnostics.get("failed", False):
                                            row = skipped_metric_row(
                                                data=dataset.data,
                                                truth=dataset.truth,
                                                diagnostics=diagnostics,
                                                size=str(n_items),
                                                scenario=graph_design,
                                                method_name=method_name,
                                                runtime_seconds=runtime,
                                            )
                                            row.update(common)
                                            metrics_rows.append(row)
                                            continue
                                        pairwise_curve, pairwise_ece = evaluate_pairwise_calibration(
                                            posterior,
                                            dataset.truth,
                                            bins=10,
                                            max_pairs=10000,
                                            random_state=run_seed,
                                            pair_subset="all_pairs",
                                        )
                                        pairwise_curve.insert(0, "method_name", method_name)
                                        pairwise_curve.insert(0, "graph_design", graph_design)
                                        pairwise_curve["pairwise_ece"] = pairwise_ece
                                        pairwise_rows.append(pairwise_curve)
                                        close_curve, close_ece = evaluate_pairwise_calibration(
                                            posterior,
                                            dataset.truth,
                                            bins=10,
                                            max_pairs=10000,
                                            random_state=run_seed,
                                            pair_subset="close_pairs",
                                        )
                                        close_curve.insert(0, "method_name", method_name)
                                        close_curve.insert(0, "graph_design", graph_design)
                                        close_curve["pairwise_ece"] = close_ece
                                        pairwise_rows.append(close_curve)
                                        view_count, component = _item_view_counts_and_components(dataset.data, diagnostics)
                                        stratified = evaluate_stratified_coverage(
                                            posterior,
                                            dataset.truth,
                                            view_count=view_count,
                                            graph_component=component,
                                        )
                                        stratified.insert(0, "method_name", method_name)
                                        stratified.insert(0, "graph_design", graph_design)
                                        stratified_rows.append(stratified)
                                        row = compute_metrics(
                                            data=dataset.data,
                                            truth=dataset.truth,
                                            posterior=posterior,
                                            diagnostics=diagnostics,
                                            size=str(n_items),
                                            scenario=graph_design,
                                            method_name=method_name,
                                            runtime_seconds=runtime,
                                            true_view_parameters=dataset.view_parameters,
                                        )
                                        row.update(common)
                                        row["pairwise_ece"] = pairwise_ece
                                        metrics_rows.append(row)
    metrics = pd.DataFrame(metrics_rows)
    pairwise = pd.concat(pairwise_rows, ignore_index=True) if pairwise_rows else pd.DataFrame()
    stratified = pd.concat(stratified_rows, ignore_index=True) if stratified_rows else pd.DataFrame()
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pairwise.to_csv(output_dir / "pairwise_calibration.csv", index=False)
    stratified.to_csv(output_dir / "stratified_coverage.csv", index=False)
    pd.DataFrame(skipped_cells).to_csv(output_dir / "skipped_cells.csv", index=False)
    write_graph_plots(metrics, pairwise, output_dir)
    (output_dir / "method_diagnostics.json").write_text(json.dumps(_json_ready(method_diagnostics), indent=2), encoding="utf-8")
    (output_dir / "graph_diagnostics.json").write_text(json.dumps(_json_ready(graph_diagnostics), indent=2), encoding="utf-8")
    return metrics


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def load_graph_config(config_path: str | Path) -> dict[str, object]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Synthetic graph config must be a mapping.")
    return loaded


def _config_list(config: dict[str, object], key: str, default: list[object]) -> list[object]:
    value = config.get(key, default)
    return list(value) if isinstance(value, list) else [value]


def _config_list_alias(config: dict[str, object], key: str, aliases: list[str], default: list[object]) -> list[object]:
    for candidate in [key, *aliases]:
        if candidate in config:
            return _config_list(config, candidate, default)
    return list(default)


def _runtime_medians_by_method(reference_metrics: str | Path = "results/synthetic_graph/synthetic_graph_small_full/metrics.csv") -> dict[str, float]:
    path = Path(reference_metrics)
    if not path.exists():
        return {}
    try:
        metrics = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return {}
    if metrics.empty or "method_name" not in metrics or "runtime_seconds" not in metrics:
        return {}
    grouped = metrics.groupby("method_name")["runtime_seconds"].median(numeric_only=True)
    medians = {str(method): float(value) for method, value in grouped.dropna().items()}
    aceb_median = medians.get("anchored_calibrated_eb")
    if aceb_median is not None:
        for method in ACEB_COMPONENT_VARIANTS:
            medians.setdefault(method, aceb_median)
    return medians


def validate_synthetic_graph_config(config: dict[str, object]) -> None:
    config_type = str(config.get("config_type", config.get("experiment_type", "synthetic_graph")))
    methods = [str(method) for method in _config_list(config, "methods", [])]
    known_methods = set(BASE_METHODS) | {"oracle_true_a", "oracle_true_tau", "oracle_true_a_true_tau"} | ACEB_COMPONENT_VARIANTS
    unknown_methods = sorted(set(methods) - known_methods)
    if unknown_methods:
        raise ValueError(f"Unknown methods in config: {unknown_methods}")
    if config_type == "synthetic_graph_intervention":
        valid_interventions = {
            "star_to_star_plus_edge",
            "chain_to_triangle_plus_tail",
            "even_cycle_to_odd_cycle_bridge",
            "disconnected_to_connected_with_odd_cycle",
        }
        interventions = [str(value) for value in _config_list(config, "interventions", [])]
        unknown = sorted(set(interventions) - valid_interventions)
        if unknown:
            raise ValueError(f"Unknown paired interventions in config: {unknown}")
        budget_modes = [str(value) for value in _config_list(config, "budget_mode", ["both"])]
        bad_budget = sorted(set(budget_modes) - {"additive", "budget_matched", "both"})
        if bad_budget:
            raise ValueError(f"Unknown budget_mode values: {bad_budget}")
        return
    graph_designs = [str(value) for value in _config_list(config, "graph_designs", GRAPH_DESIGNS)]
    unknown_designs = sorted(set(graph_designs) - set(GRAPH_DESIGNS))
    if unknown_designs:
        raise ValueError(f"Unknown graph designs in config: {unknown_designs}")
    for missingness in _config_list_alias(config, "missingness_modes", ["missingness"], ["none"]):
        if str(missingness) not in MISSINGNESS_MODES:
            raise ValueError(f"Unknown missingness mode: {missingness}")
    for distribution in _config_list_alias(config, "theta_distributions", ["theta_distribution"], ["normal"]):
        if str(distribution) not in THETA_DISTRIBUTIONS:
            raise ValueError(f"Unknown theta distribution: {distribution}")


def build_synthetic_graph_manifest(config: dict[str, object]) -> pd.DataFrame:
    validate_synthetic_graph_config(config)
    config_type = str(config.get("config_type", config.get("experiment_type", "synthetic_graph")))
    if config_type == "synthetic_graph_intervention":
        interventions = [str(value) for value in _config_list(config, "interventions", ["star_to_star_plus_edge"])]
        budget_modes = [str(value) for value in _config_list(config, "budget_mode", ["both"])]
        n_items_values = [int(value) for value in _config_list_alias(config, "n_items_values", ["n_items"], [100])]
        n_views_values = [int(value) for value in _config_list_alias(config, "n_views_values", ["n_views"], [5])]
        tau_ratios = [float(value) for value in _config_list_alias(config, "tau_ratios", ["tau_ratio"], [0.5])]
        loading_spreads = [float(value) for value in _config_list_alias(config, "loading_spreads", ["loading_spread"], [0.4])]
        missingness_modes = [str(value) for value in _config_list_alias(config, "missingness_modes", ["missingness"], ["none"])]
        theta_distributions = [str(value) for value in _config_list_alias(config, "theta_distributions", ["theta_distribution"], ["normal"])]
        seeds = [int(value) for value in _config_list(config, "seeds", [int(config.get("random_state", config.get("seed", 1)))])]
        rows: list[dict[str, object]] = []
        run_idx = 0
        for seed_value in seeds:
            for intervention in interventions:
                for budget_mode in budget_modes:
                    for n_items in n_items_values:
                        for n_views in n_views_values:
                            for tau_ratio in tau_ratios:
                                for loading_spread in loading_spreads:
                                    for missingness in missingness_modes:
                                        for theta_distribution in theta_distributions:
                                            run_seed = int(seed_value) + run_idx * 997
                                            try:
                                                spec = _paired_intervention_spec(intervention, n_views)
                                                valid = True
                                                skip_reason = None
                                                family = spec.name
                                            except ValueError as exc:
                                                valid = False
                                                skip_reason = str(exc)
                                                family = intervention
                                            rows.append(
                                                {
                                                    "cell_id": run_idx,
                                                    "config_type": config_type,
                                                    "graph_design": intervention,
                                                    "intervention_name": intervention,
                                                    "budget_mode": budget_mode,
                                                    "n_items": n_items,
                                                    "n_views": n_views,
                                                    "tau_ratio": tau_ratio,
                                                    "loading_spread": loading_spread,
                                                    "seed": int(seed_value),
                                                    "run_seed": int(run_seed),
                                                    "theta_distribution": theta_distribution,
                                                    "missingness": missingness,
                                                    "status": "pending" if valid else "skipped",
                                                    "design_valid": bool(valid),
                                                    "skip_reason": skip_reason,
                                                    "design_family": family,
                                                }
                                            )
                                            run_idx += 1
        return pd.DataFrame(rows)
    graph_designs = _config_list(config, "graph_designs", GRAPH_DESIGNS)
    n_items_values = [int(value) for value in _config_list_alias(config, "n_items_values", ["n_items"], [100])]
    n_views_values = [int(value) for value in _config_list_alias(config, "n_views_values", ["n_views"], [5])]
    tau_ratios = [float(value) for value in _config_list_alias(config, "tau_ratios", ["tau_ratio"], [0.5])]
    loading_spreads = [float(value) for value in _config_list_alias(config, "loading_spreads", ["loading_spread"], [0.4])]
    missingness_modes = [str(value) for value in _config_list_alias(config, "missingness_modes", ["missingness"], ["none"])]
    theta_distributions = [str(value) for value in _config_list_alias(config, "theta_distributions", ["theta_distribution"], ["normal"])]
    seeds = [int(value) for value in _config_list(config, "seeds", [int(config.get("random_state", config.get("seed", 1)))])]
    er_density = float(config.get("er_density", 0.35))
    rows: list[dict[str, object]] = []
    run_idx = 0
    for seed_value in seeds:
        for graph_design in graph_designs:
            for n_items in n_items_values:
                for n_views in n_views_values:
                    for tau_ratio in tau_ratios:
                        for loading_spread in loading_spreads:
                            for missingness in missingness_modes:
                                for theta_distribution in theta_distributions:
                                    run_seed = int(seed_value) + run_idx * 997
                                    spec = make_graph_design(str(graph_design), n_views, seed=run_seed, density=er_density)
                                    rows.append(
                                        {
                                            "cell_id": run_idx,
                                            "graph_design": str(graph_design),
                                            "n_items": n_items,
                                            "n_views": n_views,
                                            "tau_ratio": tau_ratio,
                                            "loading_spread": loading_spread,
                                            "seed": int(seed_value),
                                            "run_seed": int(run_seed),
                                            "theta_distribution": theta_distribution,
                                            "missingness": missingness,
                                            "status": "pending" if spec.valid else "skipped",
                                            "design_valid": bool(spec.valid),
                                            "skip_reason": spec.skip_reason,
                                            "design_family": spec.design_family,
                                        }
                                    )
                                    run_idx += 1
    return pd.DataFrame(rows)


def synthetic_graph_dry_run_summary(config: dict[str, object]) -> dict[str, object]:
    validate_synthetic_graph_config(config)
    config_type = str(config.get("config_type", config.get("experiment_type", "synthetic_graph")))
    methods = [str(method) for method in _config_list(config, "methods", list(BASE_METHODS))]
    runtime_medians = _runtime_medians_by_method()
    if config_type == "synthetic_graph_intervention":
        interventions = [str(value) for value in _config_list(config, "interventions", ["star_to_star_plus_edge"])]
        budget_modes = [str(value) for value in _config_list(config, "budget_mode", ["both"])]
        n_items_values = [int(value) for value in _config_list_alias(config, "n_items_values", ["n_items"], [100])]
        n_views_values = [int(value) for value in _config_list_alias(config, "n_views_values", ["n_views"], [5])]
        tau_ratios = [float(value) for value in _config_list_alias(config, "tau_ratios", ["tau_ratio"], [0.5])]
        loading_spreads = [float(value) for value in _config_list_alias(config, "loading_spreads", ["loading_spread"], [0.4])]
        theta_distributions = [str(value) for value in _config_list_alias(config, "theta_distributions", ["theta_distribution"], ["normal"])]
        missingness_modes = [str(value) for value in _config_list_alias(config, "missingness_modes", ["missingness"], ["none"])]
        seeds = [int(value) for value in _config_list(config, "seeds", [0])]
        total = (
            len(interventions)
            * len(budget_modes)
            * len(n_items_values)
            * len(n_views_values)
            * len(tau_ratios)
            * len(loading_spreads)
            * len(theta_distributions)
            * len(missingness_modes)
            * len(seeds)
        )
        skipped = 0
        for intervention in interventions:
            for n_views in n_views_values:
                try:
                    _paired_intervention_spec(intervention, n_views)
                except ValueError:
                    skipped += len(budget_modes) * len(n_items_values) * len(tau_ratios) * len(loading_spreads) * len(theta_distributions) * len(missingness_modes) * len(seeds)
        valid = total - skipped
        method_fits = valid * len(methods) * 2
        estimated_seconds = valid * 2 * sum(runtime_medians.get(method, np.nan) for method in methods if np.isfinite(runtime_medians.get(method, np.nan)))
        return {
            "config_type": config_type,
            "total_requested_cells": total,
            "valid_cells": valid,
            "skipped_cells": skipped,
            "expected_method_fits": method_fits,
            "estimated_runtime_seconds": estimated_seconds if estimated_seconds > 0 else np.nan,
            "methods": methods,
        }
    manifest = build_synthetic_graph_manifest(config)
    valid = int(manifest["design_valid"].sum()) if "design_valid" in manifest else len(manifest)
    skipped = int((~manifest["design_valid"]).sum()) if "design_valid" in manifest else 0
    method_fits = valid * len(methods)
    estimated_seconds = valid * sum(runtime_medians.get(method, np.nan) for method in methods if np.isfinite(runtime_medians.get(method, np.nan)))
    return {
        "config_type": config_type,
        "total_requested_cells": int(len(manifest)),
        "valid_cells": valid,
        "skipped_cells": skipped,
        "expected_method_fits": int(method_fits),
        "estimated_runtime_seconds": estimated_seconds if estimated_seconds > 0 else np.nan,
        "methods": methods,
    }


def create_manifest_from_config(config_path: str | Path, run_dir: str | Path | None = None) -> Path:
    config = load_graph_config(config_path)
    if run_dir is None:
        run_id = str(config.get("run_id", "synthetic_graph"))
        run_dir = Path("results") / "synthetic_graph" / run_id
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_synthetic_graph_manifest(config)
    path = run_dir / "manifest.csv"
    manifest.to_csv(path, index=False)
    return path


def _empty_artifacts(output_dir: Path, *, graph_info: dict[str, object], status: dict[str, object]) -> None:
    pd.DataFrame().to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame().to_csv(output_dir / "pairwise_calibration.csv", index=False)
    pd.DataFrame().to_csv(output_dir / "stratified_coverage.csv", index=False)
    (output_dir / "graph_diagnostics.json").write_text(json.dumps(_json_ready(graph_info), indent=2), encoding="utf-8")
    (output_dir / "method_diagnostics.json").write_text("{}\n", encoding="utf-8")
    (output_dir / "status.json").write_text(json.dumps(_json_ready(status), indent=2), encoding="utf-8")


def run_synthetic_graph_cell(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    cell_id: int,
    output_root: str | Path,
    force: bool = False,
) -> Path:
    config = load_graph_config(config_path)
    manifest = pd.read_csv(manifest_path)
    matches = manifest.loc[manifest["cell_id"].astype(int) == int(cell_id)]
    if matches.empty:
        raise ValueError(f"cell_id {cell_id} not found in {manifest_path}")
    row = matches.iloc[0].to_dict()
    output_root = Path(output_root)
    final_dir = output_root / "cells" / str(int(cell_id))
    status_path = final_dir / "status.json"
    if status_path.exists() and not force:
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            status = {}
        if status.get("status") == "success":
            return final_dir
    tmp_dir = output_root / "cells" / f".tmp_{os.getpid()}_{int(cell_id)}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    start_time = datetime.now(timezone.utc).isoformat()
    try:
        status = _run_cell_into_directory(config=config, row=row, cell_id=int(cell_id), output_dir=tmp_dir, start=start, start_time=start_time)
    except Exception as exc:
        if bool(config.get("strict", False)):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        status = {
            "cell_id": int(cell_id),
            "status": "failed",
            "start_time": start_time,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": float(time.perf_counter() - start),
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
        _empty_artifacts(tmp_dir, graph_info={}, status=status)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)
    return final_dir


def _run_cell_into_directory(
    *,
    config: dict[str, object],
    row: dict[str, object],
    cell_id: int,
    output_dir: Path,
    start: float,
    start_time: str,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_type = str(config.get("config_type", config.get("experiment_type", "synthetic_graph")))
    if config_type == "synthetic_graph_intervention":
        return _run_intervention_cell_into_directory(
            config=config,
            row=row,
            cell_id=cell_id,
            output_dir=output_dir,
            start=start,
            start_time=start_time,
        )
    graph_design = str(row["graph_design"])
    n_views = int(row["n_views"])
    run_seed = int(row["run_seed"])
    er_density = float(config.get("er_density", 0.35))
    spec = make_graph_design(graph_design, n_views, seed=run_seed, density=er_density)
    if not spec.valid:
        status = {
            "cell_id": cell_id,
            "status": "skipped",
            "start_time": start_time,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": float(time.perf_counter() - start),
            "skip_reason": spec.skip_reason,
        }
        graph_info = {
            "cell_id": cell_id,
            "graph_design": graph_design,
            "n_views": n_views,
            "intended_edges": [],
            "observed_edges": [],
            "design_valid": False,
            "skip_reason": spec.skip_reason,
            "design_family": spec.design_family,
        }
        _empty_artifacts(output_dir, graph_info=graph_info, status=status)
        return status

    dataset = simulate_graph_synthetic(
        graph_design=graph_design,
        n_items=int(row["n_items"]),
        n_views=n_views,
        tau_ratio=float(row["tau_ratio"]),
        loading_spread=float(row["loading_spread"]),
        missingness=str(row["missingness"]),
        theta_distribution=str(row["theta_distribution"]),
        seed=run_seed,
        er_density=er_density,
    )
    validation = validate_observed_graph_matches_design(dataset.data, spec, min_shared_items=int(config.get("min_shared_items", 1)))
    graph = validation["diagnostic"].to_summary_dict()
    graph_info = {
        **graph,
        "cell_id": cell_id,
        "graph_design": graph_design,
        "n_views": n_views,
        "intended_edges": validation["intended_edges"],
        "observed_edges": validation["observed_edges"],
        "design_valid": bool(validation["matches"]),
        "skip_reason": None if validation["matches"] else "observed graph does not match intended graph",
        "missing_edges": validation["missing_edges"],
        "extra_edges": validation["extra_edges"],
        "design_family": spec.design_family,
        "expected_bipartite": spec.expected_bipartite,
        "expected_has_odd_cycle": spec.expected_has_odd_cycle,
        "expected_rank_deficient": spec.expected_rank_deficient,
    }
    if not validation["matches"]:
        status = {
            "cell_id": cell_id,
            "status": "skipped",
            "start_time": start_time,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": float(time.perf_counter() - start),
            "skip_reason": graph_info["skip_reason"],
        }
        _empty_artifacts(output_dir, graph_info=graph_info, status=status)
        return status

    metrics_rows: list[dict[str, object]] = []
    pairwise_rows: list[pd.DataFrame] = []
    stratified_rows: list[pd.DataFrame] = []
    method_diagnostics: dict[str, object] = {}
    for method_name, posterior, diagnostics, runtime in run_graph_methods(
        dataset,
        max_iter=int(config.get("max_iter", 20)),
        include_bootstrap=bool(config.get("include_bootstrap", False)),
        include_mash=bool(config.get("include_mash", False)),
        strict=bool(config.get("strict", False)),
        method_names=list(config["methods"]) if "methods" in config else None,
        include_oracles=bool(config.get("include_oracles", True)),
    ):
        method_diagnostics[method_name] = diagnostics
        common = {
            "cell_id": cell_id,
            "graph_design": graph_design,
            "n_views_config": n_views,
            "tau_ratio": float(row["tau_ratio"]),
            "loading_spread": float(row["loading_spread"]),
            "missingness": str(row["missingness"]),
            "theta_distribution": str(row["theta_distribution"]),
            "seed": int(row["seed"]),
            "run_seed": run_seed,
            "delta_G": graph["rank_deficiency_delta"],
            "has_odd_cycle": graph["anchor_component_is_non_bipartite"],
            "n_graph_edges": graph["n_edges"],
            "graph_design_rank": graph.get("graph_design_rank", graph.get("signless_incidence_rank", np.nan)),
            "graph_design_num_columns": graph.get("graph_design_num_columns", np.nan),
            "smallest_full_singular_value": graph.get("smallest_full_singular_value", graph.get("smallest_singular_value", np.nan)),
            "smallest_singular_value": graph.get("smallest_full_singular_value", graph.get("smallest_singular_value", np.nan)),
            "largest_singular_value": graph.get("largest_singular_value", np.nan),
            "condition_number": graph.get("condition_number", np.nan),
            "normalized_smallest_full_singular_value": graph.get("normalized_smallest_full_singular_value", np.nan),
            "normalized_condition_number": graph.get("normalized_condition_number", np.nan),
            "component_count": graph.get("component_count", graph.get("n_components", np.nan)),
            "anchor_component_bipartite": graph.get("anchor_component_bipartite", np.nan),
            "anchor_component_has_odd_cycle": graph.get("anchor_component_has_odd_cycle", np.nan),
        }
        if diagnostics.get("skipped", False) or diagnostics.get("failed", False):
            metric_row = skipped_metric_row(
                data=dataset.data,
                truth=dataset.truth,
                diagnostics=diagnostics,
                size=str(int(row["n_items"])),
                scenario=graph_design,
                method_name=method_name,
                runtime_seconds=runtime,
            )
            metric_row.update(common)
            metrics_rows.append(metric_row)
            continue
        pairwise_curve, pairwise_ece = evaluate_pairwise_calibration(
            posterior,
            dataset.truth,
            bins=10,
            max_pairs=int(config.get("max_pairs", 10000)),
            random_state=run_seed,
            pair_subset="all_pairs",
        )
        pairwise_curve.insert(0, "cell_id", cell_id)
        pairwise_curve.insert(1, "method_name", method_name)
        pairwise_curve.insert(2, "graph_design", graph_design)
        pairwise_curve["pairwise_ece"] = pairwise_ece
        pairwise_rows.append(pairwise_curve)
        close_curve, close_ece = evaluate_pairwise_calibration(
            posterior,
            dataset.truth,
            bins=10,
            max_pairs=int(config.get("max_pairs", 10000)),
            random_state=run_seed,
            pair_subset="close_pairs",
        )
        close_curve.insert(0, "cell_id", cell_id)
        close_curve.insert(1, "method_name", method_name)
        close_curve.insert(2, "graph_design", graph_design)
        close_curve["pairwise_ece"] = close_ece
        pairwise_rows.append(close_curve)
        view_count, component = _item_view_counts_and_components(dataset.data, diagnostics)
        stratified = evaluate_stratified_coverage(posterior, dataset.truth, view_count=view_count, graph_component=component)
        stratified.insert(0, "cell_id", cell_id)
        stratified.insert(1, "method_name", method_name)
        stratified.insert(2, "graph_design", graph_design)
        stratified_rows.append(stratified)
        metric_row = compute_metrics(
            data=dataset.data,
            truth=dataset.truth,
            posterior=posterior,
            diagnostics=diagnostics,
            size=str(int(row["n_items"])),
            scenario=graph_design,
            method_name=method_name,
            runtime_seconds=runtime,
            true_view_parameters=dataset.view_parameters,
        )
        metric_row.update(common)
        metric_row["pairwise_ece"] = pairwise_ece
        metrics_rows.append(metric_row)

    pd.DataFrame(metrics_rows).to_csv(output_dir / "metrics.csv", index=False)
    (pd.concat(pairwise_rows, ignore_index=True) if pairwise_rows else pd.DataFrame()).to_csv(
        output_dir / "pairwise_calibration.csv",
        index=False,
    )
    (pd.concat(stratified_rows, ignore_index=True) if stratified_rows else pd.DataFrame()).to_csv(
        output_dir / "stratified_coverage.csv",
        index=False,
    )
    (output_dir / "graph_diagnostics.json").write_text(json.dumps(_json_ready(graph_info), indent=2), encoding="utf-8")
    (output_dir / "method_diagnostics.json").write_text(json.dumps(_json_ready(method_diagnostics), indent=2), encoding="utf-8")
    status = {
        "cell_id": cell_id,
        "status": "success",
        "start_time": start_time,
        "end_time": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": float(time.perf_counter() - start),
    }
    (output_dir / "status.json").write_text(json.dumps(_json_ready(status), indent=2), encoding="utf-8")
    return status


def _run_intervention_cell_into_directory(
    *,
    config: dict[str, object],
    row: dict[str, object],
    cell_id: int,
    output_dir: Path,
    start: float,
    start_time: str,
) -> dict[str, object]:
    intervention = str(row.get("intervention_name", row.get("graph_design", "star_to_star_plus_edge")))
    n_views = int(row["n_views"])
    run_seed = int(row["run_seed"])
    budget_mode = str(row.get("budget_mode", "both"))
    try:
        _paired_intervention_spec(intervention, n_views)
    except ValueError as exc:
        status = {
            "cell_id": cell_id,
            "status": "skipped",
            "start_time": start_time,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": float(time.perf_counter() - start),
            "skip_reason": str(exc),
        }
        graph_info = {
            "cell_id": cell_id,
            "config_type": "synthetic_graph_intervention",
            "intervention_name": intervention,
            "budget_mode": budget_mode,
            "n_views": n_views,
            "design_valid": False,
            "skip_reason": str(exc),
        }
        _empty_artifacts(output_dir, graph_info=graph_info, status=status)
        return status

    paired = generate_paired_graph_intervention(
        n_items=int(row["n_items"]),
        n_views=n_views,
        intervention=intervention,
        seed=run_seed,
        tau_ratio=float(row["tau_ratio"]),
        loading_spread=float(row["loading_spread"]),
        n_items_per_edge=int(config.get("n_items_per_edge", 40)),
        budget_mode=budget_mode,
        theta_distribution=str(row["theta_distribution"]),
    )
    metrics_rows: list[dict[str, object]] = []
    pairwise_rows: list[pd.DataFrame] = []
    stratified_rows: list[pd.DataFrame] = []
    graph_info: dict[str, object] = {
        "cell_id": cell_id,
        "config_type": "synthetic_graph_intervention",
        "intervention_name": intervention,
        "budget_mode": budget_mode,
        "n_views": n_views,
        "design_valid": True,
        "skip_reason": None,
        "arms": {},
    }
    method_diagnostics: dict[str, object] = {}

    for arm, dataset in paired.datasets.items():
        graph = build_coobservation_graph(dataset.data, min_shared_items=int(config.get("min_shared_items", 1))).to_summary_dict()
        conditioning = _graph_conditioning_metrics(dataset.intended_edges, dataset.data.n_views)
        graph_info["arms"][arm] = {
            **graph,
            **conditioning,
            "arm": arm,
            "intended_edges": dataset.intended_edges,
            "observed_edges": graph.get("edge_n_shared", {}),
            "design_valid": True,
            "skip_reason": None,
        }
        method_diagnostics[arm] = {}
        for method_name, posterior, diagnostics, runtime in run_graph_methods(
            dataset,
            max_iter=int(config.get("max_iter", 20)),
            include_bootstrap=bool(config.get("include_bootstrap", False)),
            include_mash=bool(config.get("include_mash", False)),
            strict=bool(config.get("strict", False)),
            method_names=list(config["methods"]) if "methods" in config else None,
            include_oracles=bool(config.get("include_oracles", False)),
        ):
            method_diagnostics[arm][method_name] = diagnostics
            common = {
                "cell_id": cell_id,
                "config_type": "synthetic_graph_intervention",
                "graph_design": intervention,
                "intervention_name": intervention,
                "budget_mode": budget_mode,
                "arm": arm,
                "pair_id": dataset.metadata.get("pair_id"),
                "dataset_name": dataset.metadata.get("dataset_name"),
                "n_views_config": n_views,
                "tau_ratio": float(row["tau_ratio"]),
                "loading_spread": float(row["loading_spread"]),
                "missingness": str(row["missingness"]),
                "theta_distribution": str(row["theta_distribution"]),
                "seed": int(row["seed"]),
                "run_seed": run_seed,
                "delta_G": graph.get("rank_deficiency_delta", graph.get("delta_G", np.nan)),
                "has_odd_cycle": graph.get("anchor_component_is_non_bipartite", np.nan),
                "n_graph_edges": graph.get("n_edges", np.nan),
                "graph_design_rank": graph.get("graph_design_rank", graph.get("signless_incidence_rank", np.nan)),
                "graph_design_num_columns": graph.get("graph_design_num_columns", np.nan),
                "smallest_full_singular_value": conditioning.get("smallest_full_singular_value", np.nan),
                "smallest_singular_value": conditioning.get("smallest_full_singular_value", np.nan),
                "largest_singular_value": conditioning.get("largest_singular_value", np.nan),
                "condition_number": conditioning.get("condition_number", np.nan),
                "normalized_smallest_full_singular_value": conditioning.get("normalized_smallest_full_singular_value", np.nan),
                "normalized_condition_number": conditioning.get("normalized_condition_number", np.nan),
                "component_count": graph.get("component_count", graph.get("n_components", np.nan)),
                "anchor_component_bipartite": graph.get("anchor_component_bipartite", np.nan),
                "anchor_component_has_odd_cycle": graph.get("anchor_component_has_odd_cycle", np.nan),
            }
            if diagnostics.get("skipped", False) or diagnostics.get("failed", False):
                metric_row = skipped_metric_row(
                    data=dataset.data,
                    truth=dataset.truth,
                    diagnostics=diagnostics,
                    size=str(int(row["n_items"])),
                    scenario=f"{intervention}:{arm}",
                    method_name=method_name,
                    runtime_seconds=runtime,
                )
                metric_row.update(common)
                metrics_rows.append(metric_row)
                continue
            pairwise_curve, pairwise_ece = evaluate_pairwise_calibration(
                posterior,
                dataset.truth,
                bins=10,
                max_pairs=int(config.get("max_pairs", 10000)),
                random_state=run_seed,
                pair_subset="all_pairs",
            )
            pairwise_curve.insert(0, "cell_id", cell_id)
            pairwise_curve.insert(1, "method_name", method_name)
            pairwise_curve.insert(2, "graph_design", intervention)
            pairwise_curve["intervention_name"] = intervention
            pairwise_curve["budget_mode"] = budget_mode
            pairwise_curve["arm"] = arm
            pairwise_curve["pairwise_ece"] = pairwise_ece
            pairwise_rows.append(pairwise_curve)
            close_curve, close_ece = evaluate_pairwise_calibration(
                posterior,
                dataset.truth,
                bins=10,
                max_pairs=int(config.get("max_pairs", 10000)),
                random_state=run_seed,
                pair_subset="close_pairs",
            )
            close_curve.insert(0, "cell_id", cell_id)
            close_curve.insert(1, "method_name", method_name)
            close_curve.insert(2, "graph_design", intervention)
            close_curve["intervention_name"] = intervention
            close_curve["budget_mode"] = budget_mode
            close_curve["arm"] = arm
            close_curve["pairwise_ece"] = close_ece
            pairwise_rows.append(close_curve)
            view_count, component = _item_view_counts_and_components(dataset.data, diagnostics)
            stratified = evaluate_stratified_coverage(posterior, dataset.truth, view_count=view_count, graph_component=component)
            stratified.insert(0, "cell_id", cell_id)
            stratified.insert(1, "method_name", method_name)
            stratified.insert(2, "graph_design", intervention)
            stratified["intervention_name"] = intervention
            stratified["budget_mode"] = budget_mode
            stratified["arm"] = arm
            stratified_rows.append(stratified)
            metric_row = compute_metrics(
                data=dataset.data,
                truth=dataset.truth,
                posterior=posterior,
                diagnostics=diagnostics,
                size=str(int(row["n_items"])),
                scenario=f"{intervention}:{arm}",
                method_name=method_name,
                runtime_seconds=runtime,
                true_view_parameters=dataset.view_parameters,
            )
            metric_row.update(common)
            metric_row["pairwise_ece"] = pairwise_ece
            metrics_rows.append(metric_row)

    pd.DataFrame(metrics_rows).to_csv(output_dir / "metrics.csv", index=False)
    (pd.concat(pairwise_rows, ignore_index=True) if pairwise_rows else pd.DataFrame()).to_csv(
        output_dir / "pairwise_calibration.csv",
        index=False,
    )
    (pd.concat(stratified_rows, ignore_index=True) if stratified_rows else pd.DataFrame()).to_csv(
        output_dir / "stratified_coverage.csv",
        index=False,
    )
    (output_dir / "graph_diagnostics.json").write_text(json.dumps(_json_ready(graph_info), indent=2), encoding="utf-8")
    (output_dir / "method_diagnostics.json").write_text(json.dumps(_json_ready(method_diagnostics), indent=2), encoding="utf-8")
    status = {
        "cell_id": cell_id,
        "status": "success",
        "start_time": start_time,
        "end_time": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": float(time.perf_counter() - start),
    }
    (output_dir / "status.json").write_text(json.dumps(_json_ready(status), indent=2), encoding="utf-8")
    return status


def aggregate_synthetic_graph_run(run_dir: str | Path) -> dict[str, Path]:
    run_dir = Path(run_dir)
    cells_dir = run_dir / "cells"
    metrics_frames: list[pd.DataFrame] = []
    pairwise_frames: list[pd.DataFrame] = []
    stratified_frames: list[pd.DataFrame] = []
    graph_diagnostics: dict[str, object] = {}
    method_diagnostics: dict[str, object] = {}
    skipped_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []
    for cell_dir in sorted(cells_dir.glob("*"), key=lambda path: int(path.name) if path.name.isdigit() else 10**12):
        if not cell_dir.is_dir() or not cell_dir.name.isdigit():
            continue
        cell_id = int(cell_dir.name)
        status_path = cell_dir / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"status": "failed"}
        if status.get("status") == "skipped":
            skipped_rows.append({"cell_id": cell_id, **status})
        if status.get("status") == "failed":
            failed_rows.append({"cell_id": cell_id, **status})
        for filename, frames in [
            ("metrics.csv", metrics_frames),
            ("pairwise_calibration.csv", pairwise_frames),
            ("stratified_coverage.csv", stratified_frames),
        ]:
            path = cell_dir / filename
            if path.exists() and path.stat().st_size > 1:
                try:
                    frame = pd.read_csv(path)
                except pd.errors.EmptyDataError:
                    continue
                if not frame.empty:
                    frames.append(frame)
        graph_path = cell_dir / "graph_diagnostics.json"
        if graph_path.exists():
            graph_diagnostics[str(cell_id)] = json.loads(graph_path.read_text(encoding="utf-8"))
        method_path = cell_dir / "method_diagnostics.json"
        if method_path.exists():
            method_diagnostics[str(cell_id)] = json.loads(method_path.read_text(encoding="utf-8"))

    metrics = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    pairwise = pd.concat(pairwise_frames, ignore_index=True) if pairwise_frames else pd.DataFrame()
    stratified = pd.concat(stratified_frames, ignore_index=True) if stratified_frames else pd.DataFrame()
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    pairwise.to_csv(run_dir / "pairwise_calibration.csv", index=False)
    stratified.to_csv(run_dir / "stratified_coverage.csv", index=False)
    pd.DataFrame(skipped_rows).to_csv(run_dir / "skipped_cells.csv", index=False)
    pd.DataFrame(failed_rows).to_csv(run_dir / "failed_cells.csv", index=False)
    (run_dir / "graph_diagnostics.json").write_text(json.dumps(_json_ready(graph_diagnostics), indent=2), encoding="utf-8")
    (run_dir / "method_diagnostics.json").write_text(json.dumps(_json_ready(method_diagnostics), indent=2), encoding="utf-8")
    if {"arm", "intervention_name", "pair_id", "method_name"}.issubset(metrics.columns):
        try:
            deltas = paired_intervention_deltas(metrics)
            deltas.to_csv(run_dir / "intervention_deltas.csv", index=False)
            metrics.to_csv(run_dir / "intervention_metrics.csv", index=False)
            write_intervention_summary(metrics, deltas, run_dir / "intervention_summary.md")
            write_intervention_plots(deltas, run_dir / "plots")
        except Exception as exc:
            (run_dir / "intervention_summary_error.txt").write_text(
                f"{type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
    write_graph_plots(metrics, pairwise, run_dir)
    write_run_summary(run_dir)
    return {
        "metrics": run_dir / "metrics.csv",
        "pairwise_calibration": run_dir / "pairwise_calibration.csv",
        "stratified_coverage": run_dir / "stratified_coverage.csv",
        "skipped_cells": run_dir / "skipped_cells.csv",
        "failed_cells": run_dir / "failed_cells.csv",
    }


def run_synthetic_graph_resumable(
    *,
    config_path: str | Path,
    run_dir: str | Path | None = None,
    force: bool = False,
    n_workers: int = 1,
) -> Path:
    config = load_graph_config(config_path)
    if run_dir is None:
        run_dir = Path("results") / "synthetic_graph" / str(config.get("run_id", "synthetic_graph"))
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.csv"
    if not manifest_path.exists():
        create_manifest_from_config(config_path, run_dir)
    manifest = pd.read_csv(manifest_path)

    def should_run(cell_id: int) -> bool:
        status_path = run_dir / "cells" / str(cell_id) / "status.json"
        if force or not status_path.exists():
            return True
        try:
            return json.loads(status_path.read_text(encoding="utf-8")).get("status") != "success"
        except json.JSONDecodeError:
            return True

    cell_ids = [int(value) for value in manifest["cell_id"].tolist() if should_run(int(value))]
    if n_workers <= 1:
        for cell_id in cell_ids:
            run_synthetic_graph_cell(config_path=config_path, manifest_path=manifest_path, cell_id=cell_id, output_root=run_dir, force=force)
    else:
        from multiprocessing import Pool

        args = [(str(config_path), str(manifest_path), cell_id, str(run_dir), force) for cell_id in cell_ids]
        with Pool(processes=n_workers) as pool:
            pool.starmap(_run_cell_star, args)
    aggregate_synthetic_graph_run(run_dir)
    return run_dir


def _run_cell_star(config_path: str, manifest_path: str, cell_id: int, output_root: str, force: bool) -> str:
    return str(run_synthetic_graph_cell(config_path=config_path, manifest_path=manifest_path, cell_id=cell_id, output_root=output_root, force=force))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run graph-identifiability synthetic benchmarks.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/graph_synthetic"))
    parser.add_argument("--graph-designs", nargs="+", default=GRAPH_DESIGNS, choices=GRAPH_DESIGNS)
    parser.add_argument("--n-items", nargs="+", type=int, default=[100])
    parser.add_argument("--n-views", nargs="+", type=int, default=[5])
    parser.add_argument("--tau-ratios", nargs="+", type=float, default=[0.5])
    parser.add_argument("--loading-spreads", nargs="+", type=float, default=[0.4])
    parser.add_argument("--missingness", nargs="+", default=["none"], choices=MISSINGNESS_MODES)
    parser.add_argument("--theta-distributions", nargs="+", default=["normal"], choices=THETA_DISTRIBUTIONS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--include-bootstrap", action="store_true")
    parser.add_argument("--no-mash", action="store_true")
    args = parser.parse_args(argv)
    metrics = run_graph_synthetic_benchmark(
        output_dir=args.output_dir,
        graph_designs=args.graph_designs,
        n_items_values=args.n_items,
        n_views_values=args.n_views,
        tau_ratios=args.tau_ratios,
        loading_spreads=args.loading_spreads,
        missingness_modes=args.missingness,
        theta_distributions=args.theta_distributions,
        seed=args.seed,
        max_iter=args.max_iter,
        include_bootstrap=args.include_bootstrap,
        include_mash=not args.no_mash,
    )
    print(f"Wrote {len(metrics)} graph synthetic rows to {args.output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
