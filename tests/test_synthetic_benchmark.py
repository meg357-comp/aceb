from __future__ import annotations

from cmveb.benchmarks.synthetic_runner import run_synthetic_benchmark


def test_synthetic_benchmark_smoke(tmp_path) -> None:
    metrics = run_synthetic_benchmark(
        output_dir=tmp_path / "synthetic",
        sizes=["small"],
        scenarios=["correct_specification"],
        seed=11,
        max_iter=10,
    )
    assert not metrics.empty
    assert (tmp_path / "synthetic" / "metrics.csv").exists()
    assert (tmp_path / "synthetic" / "pairwise_calibration.csv").exists()
    assert (tmp_path / "synthetic" / "stratified_coverage.csv").exists()
    assert (tmp_path / "synthetic" / "summary.md").exists()
    assert (tmp_path / "synthetic" / "plots" / "rmse.svg").exists()
    assert set(metrics["method_name"]) >= {
        "anchored_calibrated_eb",
        "joint_eb_full",
        "posthoc_inverse_variance_all_views",
    }
    assert "mae" in metrics.columns
    assert "pairwise_ece" in metrics.columns
    assert "posterior_rank_entropy" in metrics.columns
    assert "graph_rank_deficiency_delta" in metrics.columns
