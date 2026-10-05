from __future__ import annotations

import numpy as np

from cmveb.graph_diagnostics import build_coobservation_graph, edge_design_matrix, full_column_singular_values, graph_rank_deficiency


def arrays_for_edges(edges: list[tuple[int, int]], n_views: int, n_shared: int = 3):
    item_idx = []
    view_idx = []
    estimate = []
    item = 0
    for edge_number, (j, k) in enumerate(edges):
        for shared_number in range(n_shared):
            value = float(edge_number + shared_number + 1)
            item_idx.extend([item, item])
            view_idx.extend([j, k])
            estimate.extend([value, value + 0.5])
            item += 1
    return np.array(item_idx), np.array(view_idx), np.array(estimate, dtype=np.float64)


def build_from_edges(edges: list[tuple[int, int]], n_views: int, anchor_view: int = 0):
    item_idx, view_idx, estimate = arrays_for_edges(edges, n_views)
    return build_coobservation_graph(
        item_idx=item_idx,
        view_idx=view_idx,
        estimate=estimate,
        n_views=n_views,
        min_shared_items=3,
        center=False,
        anchor_view=anchor_view,
    )


def test_triangle_graph_non_bipartite_and_full_rank_with_anchor() -> None:
    graph = build_from_edges([(0, 1), (1, 2), (0, 2)], n_views=3)
    assert graph.n_edges == 3
    assert len(graph.components) == 1
    assert not graph.components[0].is_bipartite
    assert graph.components[0].has_odd_cycle
    assert graph.anchor_component_is_non_bipartite
    assert graph.rank_deficiency_delta == 0
    assert graph.signless_incidence_rank == edge_design_matrix(graph.edges, 3, 0).shape[1]


def test_star_graph_centered_at_anchor_is_bipartite_and_rank_deficient() -> None:
    graph = build_from_edges([(0, 1), (0, 2), (0, 3)], n_views=4)
    assert graph.components[0].is_bipartite
    assert not graph.components[0].has_odd_cycle
    assert graph.rank_deficiency_delta >= 1


def test_star_graph_has_zero_full_smallest_singular_value() -> None:
    graph = build_from_edges([(0, 1), (0, 2)], n_views=3)
    assert graph.rank_deficiency_delta > 0
    np.testing.assert_allclose(graph.smallest_full_singular_value, 0.0, atol=1e-10)
    returned = np.linalg.svd(edge_design_matrix(graph.edges, 3, 0), compute_uv=False)
    assert np.min(returned) > graph.smallest_full_singular_value


def test_star_plus_edge_triangle_has_positive_full_smallest_singular_value() -> None:
    graph = build_from_edges([(0, 1), (0, 2), (1, 2)], n_views=3)
    assert graph.rank_deficiency_delta == 0
    assert graph.smallest_full_singular_value > 1e-10


def test_chain_graph_is_bipartite_and_rank_deficient() -> None:
    graph = build_from_edges([(0, 1), (1, 2)], n_views=3)
    assert graph.components[0].is_bipartite
    assert graph.rank_deficiency_delta >= 1
    np.testing.assert_allclose(graph.smallest_full_singular_value, 0.0, atol=1e-10)


def test_even_cycle_is_bipartite_and_rank_deficient() -> None:
    graph = build_from_edges([(0, 1), (1, 2), (2, 3), (0, 3)], n_views=4)
    assert graph.components[0].is_bipartite
    assert graph.rank_deficiency_delta >= 1


def test_odd_cycle_is_non_bipartite_and_full_rank() -> None:
    graph = build_from_edges([(0, 1), (1, 2), (2, 3), (0, 3), (1, 3)], n_views=4)
    assert not graph.components[0].is_bipartite
    assert graph.components[0].has_odd_cycle
    assert graph.rank_deficiency_delta == 0
    assert graph.smallest_full_singular_value > 1e-10


def test_full_column_singular_values_include_implicit_zeros_for_rectangular_design() -> None:
    matrix = edge_design_matrix([(0, 1), (0, 2)], n_views=3, anchor_view=0)
    conditioning = full_column_singular_values(matrix)
    returned = np.linalg.svd(matrix, compute_uv=False)
    assert returned.shape[0] == matrix.shape[0]
    assert conditioning["singular_values_full"].shape[0] == matrix.shape[1]
    assert conditioning["delta"] == 1
    np.testing.assert_allclose(conditioning["smallest_full_singular_value"], 0.0, atol=1e-10)


def test_disconnected_graph_reports_multiple_components_and_larger_deficiency() -> None:
    graph = build_from_edges([(0, 1), (1, 2), (0, 2), (3, 4)], n_views=5)
    assert len(graph.components) == 2
    assert graph.component_id_by_view[0] == graph.component_id_by_view[2]
    assert graph.component_id_by_view[3] == graph.component_id_by_view[4]
    assert graph.component_id_by_view[0] != graph.component_id_by_view[3]
    assert graph.rank_deficiency_delta > 0
    rank_info = graph_rank_deficiency(graph.edges, graph.n_views, graph.anchor_view)
    assert rank_info["delta"] == graph.rank_deficiency_delta


def test_edge_moments_use_only_shared_items() -> None:
    item_idx = np.array([0, 0, 1, 1, 2, 2])
    view_idx = np.array([0, 1, 0, 1, 0, 2])
    estimate = np.array([1.0, 2.0, 3.0, 4.0, 100.0, 5.0])
    graph = build_coobservation_graph(
        item_idx=item_idx,
        view_idx=view_idx,
        estimate=estimate,
        n_views=3,
        min_shared_items=2,
        center=False,
        anchor_view=0,
    )
    assert graph.edges == ((0, 1),)
    assert graph.edge_moments[0].n_shared == 2
    assert graph.edge_moments[0].edge_moment == 7.0


def test_non_positive_moment_edges_are_flagged() -> None:
    item_idx = np.array([0, 0, 1, 1])
    view_idx = np.array([0, 1, 0, 1])
    estimate = np.array([1.0, -1.0, 1.0, -1.0])
    graph = build_coobservation_graph(
        item_idx=item_idx,
        view_idx=view_idx,
        estimate=estimate,
        n_views=2,
        min_shared_items=2,
        min_positive_moment=1e-12,
        center=False,
        anchor_view=0,
    )
    assert graph.edge_moments[0].edge_moment == -1.0
    assert not graph.edge_moments[0].is_positive
