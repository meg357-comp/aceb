from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from cmveb.schemas import IndexedData


@dataclass(slots=True)
class EdgeMoment:
    view_j: int
    view_k: int
    n_shared: int
    edge_moment: float
    is_positive: bool


@dataclass(slots=True)
class ComponentDiagnostic:
    component_id: int
    views: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    is_bipartite: bool
    has_odd_cycle: bool
    signless_incidence_rank: int
    rank_deficiency_delta: int


@dataclass(slots=True)
class GraphDiagnostic:
    n_views: int
    n_edges: int
    edges: tuple[tuple[int, int], ...]
    edge_moments: tuple[EdgeMoment, ...]
    component_id_by_view: np.ndarray
    components: tuple[ComponentDiagnostic, ...]
    anchor_view: int
    anchor_component_id: int
    anchor_component_is_non_bipartite: bool
    signless_incidence_rank: int
    rank_deficiency_delta: int
    graph_design_num_columns: int
    smallest_full_singular_value: float
    largest_singular_value: float
    condition_number: float
    normalized_smallest_full_singular_value: float
    normalized_condition_number: float
    nonzero_returned_singular_values: tuple[float, ...]
    min_shared_items: int
    min_positive_moment: float

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "n_views": self.n_views,
            "n_edges": self.n_edges,
            "n_components": len(self.components),
            "component_id_by_view": self.component_id_by_view.copy(),
            "anchor_view": self.anchor_view,
            "anchor_component_id": self.anchor_component_id,
            "anchor_component_is_non_bipartite": self.anchor_component_is_non_bipartite,
            "anchor_component_bipartite": not self.anchor_component_is_non_bipartite,
            "anchor_component_has_odd_cycle": self.anchor_component_is_non_bipartite,
            "signless_incidence_rank": self.signless_incidence_rank,
            "rank_deficiency_delta": self.rank_deficiency_delta,
            "graph_design_rank": self.signless_incidence_rank,
            "graph_design_num_columns": self.graph_design_num_columns,
            "delta_G": self.rank_deficiency_delta,
            "smallest_full_singular_value": self.smallest_full_singular_value,
            "smallest_singular_value": self.smallest_full_singular_value,
            "largest_singular_value": self.largest_singular_value,
            "condition_number": self.condition_number,
            "normalized_smallest_full_singular_value": self.normalized_smallest_full_singular_value,
            "normalized_condition_number": self.normalized_condition_number,
            "nonzero_returned_singular_values": list(self.nonzero_returned_singular_values),
            "component_count": len(self.components),
            "component_is_bipartite": [component.is_bipartite for component in self.components],
            "component_has_odd_cycle": [component.has_odd_cycle for component in self.components],
            "component_rank_deficiency_delta": [
                component.rank_deficiency_delta for component in self.components
            ],
            "edge_n_shared": [edge.n_shared for edge in self.edge_moments],
            "edge_moment": [edge.edge_moment for edge in self.edge_moments],
            "edge_is_positive": [edge.is_positive for edge in self.edge_moments],
        }


def _arrays_from_input(
    data: IndexedData | None,
    item_idx: np.ndarray | None,
    view_idx: np.ndarray | None,
    estimate: np.ndarray | None,
    n_views: int | None,
    anchor_view: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    if data is not None:
        item_idx = data.item_idx
        view_idx = data.view_idx
        estimate = data.estimate
        n_views = data.n_views
        if anchor_view is None:
            reference = np.flatnonzero(data.is_reference_view)
            anchor_view = int(reference[0]) if reference.size else 0
    if item_idx is None or view_idx is None or estimate is None or n_views is None:
        raise ValueError("Provide either IndexedData or item_idx, view_idx, estimate, and n_views.")
    if anchor_view is None:
        anchor_view = 0
    item_idx = np.asarray(item_idx, dtype=np.int64)
    view_idx = np.asarray(view_idx, dtype=np.int64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if item_idx.shape != view_idx.shape or item_idx.shape != estimate.shape:
        raise ValueError("item_idx, view_idx, and estimate must have the same shape.")
    if n_views <= 0:
        raise ValueError("n_views must be positive.")
    if not (0 <= anchor_view < n_views):
        raise ValueError("anchor_view must be a valid view index.")
    return item_idx, view_idx, estimate, int(n_views), int(anchor_view)


def _center_estimates(view_idx: np.ndarray, estimate: np.ndarray, n_views: int, center: bool) -> np.ndarray:
    if not center:
        return estimate.astype(np.float64, copy=True)
    centered = estimate.astype(np.float64, copy=True)
    for view in range(n_views):
        rows = view_idx == view
        if np.any(rows):
            centered[rows] -= float(np.mean(centered[rows]))
    return centered


def _view_item_maps(
    item_idx: np.ndarray,
    view_idx: np.ndarray,
    estimate: np.ndarray,
    n_views: int,
) -> list[dict[int, float]]:
    maps: list[dict[int, float]] = [dict() for _ in range(n_views)]
    for item, view, value in zip(item_idx.tolist(), view_idx.tolist(), estimate.tolist(), strict=True):
        maps[view][item] = value
    return maps


def _adjacency(n_views: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    adjacency = [[] for _ in range(n_views)]
    for j, k in edges:
        adjacency[j].append(k)
        adjacency[k].append(j)
    return adjacency


def _component_labels(n_views: int, edges: list[tuple[int, int]]) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    adjacency = _adjacency(n_views, edges)
    labels = np.full(n_views, -1, dtype=np.int64)
    components: list[tuple[int, ...]] = []
    for start in range(n_views):
        if labels[start] != -1:
            continue
        component_id = len(components)
        queue = [start]
        labels[start] = component_id
        views = []
        for view in queue:
            views.append(view)
            for neighbor in adjacency[view]:
                if labels[neighbor] == -1:
                    labels[neighbor] = component_id
                    queue.append(neighbor)
        components.append(tuple(sorted(views)))
    return labels, components


def _is_bipartite_component(views: tuple[int, ...], edges: list[tuple[int, int]]) -> bool:
    view_set = set(views)
    adjacency = {view: [] for view in views}
    for j, k in edges:
        if j in view_set and k in view_set:
            adjacency[j].append(k)
            adjacency[k].append(j)
    color: dict[int, int] = {}
    for start in views:
        if start in color:
            continue
        color[start] = 0
        queue = [start]
        for view in queue:
            for neighbor in adjacency[view]:
                if neighbor not in color:
                    color[neighbor] = 1 - color[view]
                    queue.append(neighbor)
                elif color[neighbor] == color[view]:
                    return False
    return True


def _component_design_matrix(component_edges: list[tuple[int, int]], views: tuple[int, ...], anchor_view: int) -> np.ndarray:
    if not component_edges:
        return np.zeros((0, 1 + sum(view != anchor_view for view in views)), dtype=np.float64)
    non_anchor_views = [view for view in views if view != anchor_view]
    column_by_view = {view: idx + 1 for idx, view in enumerate(non_anchor_views)}
    matrix = np.zeros((len(component_edges), 1 + len(non_anchor_views)), dtype=np.float64)
    matrix[:, 0] = 1.0
    for row_idx, (j, k) in enumerate(component_edges):
        if j in column_by_view:
            matrix[row_idx, column_by_view[j]] = 1.0
        if k in column_by_view:
            matrix[row_idx, column_by_view[k]] = 1.0
    return matrix


def edge_design_matrix(edges: list[tuple[int, int]] | tuple[tuple[int, int], ...], n_views: int, anchor_view: int) -> np.ndarray:
    non_anchor_views = [view for view in range(n_views) if view != anchor_view]
    column_by_view = {view: idx + 1 for idx, view in enumerate(non_anchor_views)}
    matrix = np.zeros((len(edges), 1 + len(non_anchor_views)), dtype=np.float64)
    matrix[:, 0] = 1.0
    for row_idx, (j, k) in enumerate(edges):
        if j in column_by_view:
            matrix[row_idx, column_by_view[j]] = 1.0
        if k in column_by_view:
            matrix[row_idx, column_by_view[k]] = 1.0
    return matrix


def full_column_singular_values(matrix: np.ndarray, *, tol: float = 1e-7) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.float64)
    n_edges = int(matrix.shape[0]) if matrix.ndim == 2 else 0
    n_columns = int(matrix.shape[1]) if matrix.ndim == 2 else 0
    if n_columns == 0:
        return {
            "singular_values_full": np.zeros(0, dtype=np.float64),
            "rank": 0,
            "delta": 0,
            "smallest_full_singular_value": 0.0,
            "largest_singular_value": 0.0,
            "condition_number": float("inf"),
            "normalized_smallest_full_singular_value": 0.0,
            "normalized_condition_number": float("inf"),
            "nonzero_returned_singular_values": np.zeros(0, dtype=np.float64),
        }
    gram = matrix.T @ matrix
    eigvals = np.maximum(np.linalg.eigvalsh(gram), 0.0)
    singular_values = np.sqrt(eigvals)
    singular_values.sort()
    returned = np.linalg.svd(matrix, compute_uv=False) if matrix.size else np.zeros(0, dtype=np.float64)
    smallest = float(singular_values[0]) if singular_values.size else 0.0
    largest = float(singular_values[-1]) if singular_values.size else 0.0
    rank_tol = max(tol, largest * tol)
    rank = int(np.sum(singular_values > rank_tol))
    if smallest <= rank_tol:
        smallest = 0.0
    condition = float("inf") if smallest <= rank_tol else float(largest / smallest)
    normalizer = math.sqrt(max(n_edges, 1))
    normalized_smallest = float(smallest / normalizer)
    normalized_largest = float(largest / normalizer)
    normalized_rank_tol = max(tol, normalized_largest * tol)
    normalized_condition = float("inf") if normalized_smallest <= normalized_rank_tol else float(normalized_largest / normalized_smallest)
    return {
        "singular_values_full": singular_values,
        "rank": rank,
        "delta": int(n_columns - rank),
        "smallest_full_singular_value": smallest,
        "largest_singular_value": largest,
        "condition_number": condition,
        "normalized_smallest_full_singular_value": normalized_smallest,
        "normalized_condition_number": normalized_condition,
        "nonzero_returned_singular_values": returned,
    }


def graph_rank_deficiency(
    edges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    n_views: int,
    anchor_view: int,
) -> dict[str, Any]:
    edge_list = [tuple(sorted(edge)) for edge in edges]
    matrix = edge_design_matrix(edge_list, n_views, anchor_view)
    conditioning = full_column_singular_values(matrix)
    rank = int(conditioning["rank"])
    delta = int(conditioning["delta"])
    labels, components = _component_labels(n_views, edge_list)
    component_rows = []
    for component_id, views in enumerate(components):
        component_edges = [edge for edge in edge_list if labels[edge[0]] == component_id and labels[edge[1]] == component_id]
        component_matrix = _component_design_matrix(component_edges, views, anchor_view)
        component_conditioning = full_column_singular_values(component_matrix)
        component_rank = int(component_conditioning["rank"])
        component_delta = int(component_conditioning["delta"])
        component_rows.append(
            {
                "component_id": component_id,
                "views": views,
                "rank": component_rank,
                "delta": component_delta,
                "n_columns": int(component_matrix.shape[1]),
                "n_edges": len(component_edges),
                "contains_anchor": bool(anchor_view in views),
            }
        )
    return {
        "design_matrix": matrix,
        "rank": rank,
        "delta": delta,
        "conditioning": conditioning,
        "component_id_by_view": labels,
        "components": component_rows,
    }


def build_coobservation_graph(
    data: IndexedData | None = None,
    *,
    item_idx: np.ndarray | None = None,
    view_idx: np.ndarray | None = None,
    estimate: np.ndarray | None = None,
    n_views: int | None = None,
    min_shared_items: int = 20,
    min_positive_moment: float = 1e-12,
    center: bool = True,
    anchor_view: int | None = None,
) -> GraphDiagnostic:
    item_idx, view_idx, estimate, n_views, anchor_view = _arrays_from_input(
        data,
        item_idx,
        view_idx,
        estimate,
        n_views,
        anchor_view,
    )
    centered = _center_estimates(view_idx, estimate, n_views, center)
    view_maps = _view_item_maps(item_idx, view_idx, centered, n_views)
    edges: list[tuple[int, int]] = []
    edge_moments: list[EdgeMoment] = []
    for j in range(n_views):
        items_j = set(view_maps[j])
        for k in range(j + 1, n_views):
            shared = sorted(items_j & set(view_maps[k]))
            n_shared = len(shared)
            if n_shared < min_shared_items:
                continue
            values_j = np.array([view_maps[j][item] for item in shared], dtype=np.float64)
            values_k = np.array([view_maps[k][item] for item in shared], dtype=np.float64)
            moment = float(np.mean(values_j * values_k)) if n_shared else float("nan")
            edges.append((j, k))
            edge_moments.append(
                EdgeMoment(
                    view_j=j,
                    view_k=k,
                    n_shared=n_shared,
                    edge_moment=moment,
                    is_positive=bool(moment > min_positive_moment),
                )
            )

    component_id_by_view, component_views = _component_labels(n_views, edges)
    rank_info = graph_rank_deficiency(edges, n_views, anchor_view)
    design_matrix = rank_info["design_matrix"]
    conditioning = rank_info["conditioning"]
    component_diagnostics: list[ComponentDiagnostic] = []
    for component_id, views in enumerate(component_views):
        component_edges = tuple(
            edge for edge in edges if component_id_by_view[edge[0]] == component_id and component_id_by_view[edge[1]] == component_id
        )
        is_bipartite = _is_bipartite_component(views, edges)
        component_matrix = _component_design_matrix(list(component_edges), views, anchor_view)
        component_conditioning = full_column_singular_values(component_matrix)
        component_rank = int(component_conditioning["rank"])
        component_delta = int(component_conditioning["delta"])
        component_diagnostics.append(
            ComponentDiagnostic(
                component_id=component_id,
                views=views,
                edges=component_edges,
                is_bipartite=is_bipartite,
                has_odd_cycle=not is_bipartite,
                signless_incidence_rank=component_rank,
                rank_deficiency_delta=component_delta,
            )
        )

    anchor_component_id = int(component_id_by_view[anchor_view])
    anchor_component = component_diagnostics[anchor_component_id]
    return GraphDiagnostic(
        n_views=n_views,
        n_edges=len(edges),
        edges=tuple(edges),
        edge_moments=tuple(edge_moments),
        component_id_by_view=component_id_by_view,
        components=tuple(component_diagnostics),
        anchor_view=anchor_view,
        anchor_component_id=anchor_component_id,
        anchor_component_is_non_bipartite=not anchor_component.is_bipartite,
        signless_incidence_rank=int(rank_info["rank"]),
        rank_deficiency_delta=int(rank_info["delta"]),
        graph_design_num_columns=int(design_matrix.shape[1]) if design_matrix.ndim == 2 else 0,
        smallest_full_singular_value=float(conditioning["smallest_full_singular_value"]),
        largest_singular_value=float(conditioning["largest_singular_value"]),
        condition_number=float(conditioning["condition_number"]),
        normalized_smallest_full_singular_value=float(conditioning["normalized_smallest_full_singular_value"]),
        normalized_condition_number=float(conditioning["normalized_condition_number"]),
        nonzero_returned_singular_values=tuple(float(value) for value in conditioning["nonzero_returned_singular_values"]),
        min_shared_items=int(min_shared_items),
        min_positive_moment=float(min_positive_moment),
    )
