from __future__ import annotations

from cmveb.benchmarks.graph_synthetic import (
    GRAPH_DESIGNS,
    graph_design_edges,
    make_graph_design,
    run_graph_synthetic_benchmark,
    simulate_graph_synthetic,
    validate_observed_graph_matches_design,
)
from cmveb.graph_diagnostics import build_coobservation_graph, graph_rank_deficiency


def diagnostic_for_design(design: str, n_views: int = 5):
    dataset = simulate_graph_synthetic(
        graph_design=design,
        n_items=180,
        n_views=n_views,
        tau_ratio=0.1,
        loading_spread=0.2,
        seed=11,
    )
    return build_coobservation_graph(dataset.data, min_shared_items=1)


def test_each_graph_design_generates_expected_edges() -> None:
    for design in GRAPH_DESIGNS:
        n_views = 6 if design in {"even_cycle", "disconnected", "random_erdos_renyi"} else 5
        dataset = simulate_graph_synthetic(graph_design=design, n_items=240, n_views=n_views, seed=7)
        diagnostic = build_coobservation_graph(dataset.data, min_shared_items=1)
        assert set(diagnostic.edges) == set(dataset.intended_edges), design


def test_even_cycle_with_three_views_is_invalid() -> None:
    spec = make_graph_design("even_cycle", 3)
    assert spec.valid is False
    assert "n_views >= 4" in str(spec.skip_reason)


def test_even_cycle_with_four_views_is_bipartite_and_rank_deficient() -> None:
    spec = make_graph_design("even_cycle", 4)
    assert spec.valid is True
    diagnostic = diagnostic_for_design("even_cycle", n_views=4)
    assert diagnostic.anchor_component_is_non_bipartite is False
    assert diagnostic.rank_deficiency_delta > 0


def test_odd_cycle_with_even_views_is_invalid() -> None:
    spec = make_graph_design("odd_cycle", 4)
    assert spec.valid is False
    assert "odd n_views" in str(spec.skip_reason)


def test_odd_cycle_with_five_views_is_non_bipartite_and_full_rank() -> None:
    spec = make_graph_design("odd_cycle", 5)
    assert spec.valid is True
    diagnostic = diagnostic_for_design("odd_cycle", n_views=5)
    assert diagnostic.anchor_component_is_non_bipartite is True
    assert diagnostic.rank_deficiency_delta == 0


def test_star_plus_edge_closes_triangle_delta_zero_for_three_views() -> None:
    spec = make_graph_design("star_plus_edge_closes_triangle", 3, anchor_view=0)
    assert spec.valid is True
    rank = graph_rank_deficiency(spec.edges, n_views=3, anchor_view=0)
    assert rank["delta"] == 0


def test_generated_observed_graph_matches_intended_graph_for_valid_designs() -> None:
    for design in GRAPH_DESIGNS:
        n_views = 6 if design in {"even_cycle", "disconnected", "random_erdos_renyi"} else 5
        spec = make_graph_design(design, n_views, seed=17)
        if not spec.valid:
            continue
        dataset = simulate_graph_synthetic(graph_design=design, n_items=240, n_views=n_views, seed=17)
        validation = validate_observed_graph_matches_design(dataset.data, spec, min_shared_items=1)
        assert validation["matches"], design


def test_star_plus_edge_has_lower_delta_than_star() -> None:
    n_views = 5
    star = graph_rank_deficiency(graph_design_edges("star", n_views), n_views=n_views, anchor_view=0)
    triangle = graph_rank_deficiency(
        graph_design_edges("star_plus_edge_closes_triangle", n_views),
        n_views=n_views,
        anchor_view=0,
    )
    assert triangle["delta"] < star["delta"]


def test_odd_cycle_is_non_bipartite() -> None:
    diagnostic = diagnostic_for_design("odd_cycle", n_views=5)
    assert diagnostic.anchor_component_is_non_bipartite is True
    assert any(not component.is_bipartite for component in diagnostic.components)


def test_chain_even_cycle_and_star_are_bipartite() -> None:
    for design, n_views in [("chain", 5), ("even_cycle", 6), ("star", 5)]:
        diagnostic = diagnostic_for_design(design, n_views=n_views)
        assert diagnostic.anchor_component_is_non_bipartite is False
        assert all(component.is_bipartite for component in diagnostic.components)


def test_small_graph_benchmark_smoke_runs_non_optional_methods(tmp_path) -> None:
    metrics = run_graph_synthetic_benchmark(
        output_dir=tmp_path / "graph_synthetic",
        graph_designs=["star", "star_plus_edge_closes_triangle"],
        n_items_values=[40],
        n_views_values=[5],
        tau_ratios=[0.2],
        loading_spreads=[0.2],
        missingness_modes=["none"],
        theta_distributions=["normal"],
        seed=21,
        max_iter=2,
        include_bootstrap=False,
        include_mash=False,
    )
    out = tmp_path / "graph_synthetic"
    assert not metrics.empty
    assert (out / "metrics.csv").exists()
    assert (out / "pairwise_calibration.csv").exists()
    assert (out / "stratified_coverage.csv").exists()
    assert (out / "plots" / "coverage_vs_graph_design.svg").exists()
    assert (out / "plots" / "graph_intervention_star_vs_triangle.svg").exists()
    assert {"graph_design", "delta_G"}.issubset(metrics.columns)
    assert {"raw_reference", "anchored_calibrated_eb", "oracle_true_a"}.issubset(set(metrics["method_name"]))


def test_invalid_designs_are_skipped_by_runner(tmp_path) -> None:
    metrics = run_graph_synthetic_benchmark(
        output_dir=tmp_path / "graph_synthetic",
        graph_designs=["even_cycle", "odd_cycle"],
        n_items_values=[40],
        n_views_values=[3, 4],
        tau_ratios=[0.2],
        loading_spreads=[0.2],
        missingness_modes=["none"],
        theta_distributions=["normal"],
        seed=21,
        max_iter=2,
        include_bootstrap=False,
        include_mash=False,
        method_names=["inverse_variance_all_views"],
        include_oracles=False,
    )
    skipped = __import__("pandas").read_csv(tmp_path / "graph_synthetic" / "skipped_cells.csv")
    assert not skipped.empty
    assert ((skipped["graph_design"] == "even_cycle") & (skipped["n_views_config"] == 3)).any()
    assert ((skipped["graph_design"] == "odd_cycle") & (skipped["n_views_config"] == 4)).any()
    assert not metrics.empty
