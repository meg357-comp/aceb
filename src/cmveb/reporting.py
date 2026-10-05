from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _format_float(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.4g}"


def write_run_summary(run_dir: str | Path, *, title: str = "CMVEB Run Summary") -> Path:
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.csv"
    summary_path = run_dir / "summary.md"
    lines = [f"# {title}", ""]
    if not metrics_path.exists():
        lines.extend(["No `metrics.csv` was found.", ""])
        summary_path.write_text("\n".join(lines), encoding="utf-8")
        return summary_path

    metrics = pd.read_csv(metrics_path)
    lines.extend(
        [
            f"Metric rows: {len(metrics)}",
            f"Methods: {', '.join(sorted(metrics['method_name'].dropna().astype(str).unique())) if 'method_name' in metrics else 'unknown'}",
            "",
        ]
    )
    if "failed" in metrics:
        n_failed = int(metrics["failed"].fillna(False).astype(bool).sum())
        lines.append(f"Failed method rows: {n_failed}")
    if "skipped" in metrics:
        n_skipped = int(metrics["skipped"].fillna(False).astype(bool).sum())
        lines.append(f"Skipped method rows: {n_skipped}")
    if "rmse" in metrics:
        valid = metrics.loc[np.isfinite(metrics["rmse"].to_numpy(dtype=float))]
        if not valid.empty:
            best = valid.sort_values("rmse").iloc[0]
            lines.extend(["", "## Best RMSE", "", f"`{best['method_name']}`: {_format_float(float(best['rmse']))}"])
    if "coverage_90" in metrics:
        valid = metrics.loc[np.isfinite(metrics["coverage_90"].to_numpy(dtype=float))]
        if not valid.empty:
            coverage = valid.assign(coverage_gap=np.abs(valid["coverage_90"] - 0.90)).sort_values("coverage_gap").iloc[0]
            lines.extend(
                [
                    "",
                    "## Closest 90% Coverage",
                    "",
                    f"`{coverage['method_name']}`: coverage={_format_float(float(coverage['coverage_90']))}",
                ]
            )
    if {"method_name", "rmse", "coverage_90"}.issubset(metrics.columns):
        grouped = metrics.groupby("method_name", as_index=False)[["rmse", "coverage_90"]].mean(numeric_only=True)
        lines.extend(["", "## Mean Metrics", "", "| method_name | rmse | coverage_90 |", "| --- | --- | --- |"])
        for row in grouped.itertuples(index=False):
            lines.append(f"| {row.method_name} | {_format_float(float(row.rmse))} | {_format_float(float(row.coverage_90))} |")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def aggregate_seed_summaries(run_dirs: list[str | Path], output_path: str | Path) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        path = Path(run_dir) / "metrics.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["source_run_dir"] = str(Path(run_dir))
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return combined


def make_paper_tables(results_root: str | Path, output_dir: str | Path) -> dict[str, Path]:
    results_root = Path(results_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_files = list(results_root.rglob("metrics.csv"))
    frames = []
    for path in metrics_files:
        frame = pd.read_csv(path)
        frame["source"] = str(path.parent)
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    outputs: dict[str, Path] = {}

    def write(name: str, frame: pd.DataFrame) -> None:
        path = output_dir / name
        frame.to_csv(path, index=False)
        outputs[name] = path

    if metrics.empty:
        for name in [
            "table_synthetic_main.csv",
            "table_graph_intervention.csv",
            "table_baselines.csv",
            "table_openml_main.csv",
            "table_helm_main.csv",
        ]:
            write(name, pd.DataFrame())
        return outputs

    group_cols = [col for col in ["graph_design", "scenario", "method_name"] if col in metrics.columns]
    summary = metrics.groupby(group_cols, as_index=False).mean(numeric_only=True) if group_cols else metrics
    write("table_synthetic_main.csv", summary)
    graph = metrics.loc[metrics.get("graph_design", pd.Series("", index=metrics.index)).isin(["star", "star_plus_edge", "star_plus_edge_closes_triangle"])]
    write("table_graph_intervention.csv", graph)
    write("table_baselines.csv", metrics.groupby("method_name", as_index=False).mean(numeric_only=True) if "method_name" in metrics else metrics)
    write("table_openml_main.csv", metrics.loc[metrics["source"].str.contains("openml", case=False, na=False)])
    write("table_helm_main.csv", metrics.loc[metrics["source"].str.contains("helm", case=False, na=False)])
    return outputs


def make_paper_figures(results_root: str | Path, output_dir: str | Path) -> dict[str, Path]:
    results_root = Path(results_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_files = list(results_root.rglob("metrics.csv"))
    metrics = pd.concat([pd.read_csv(path) for path in metrics_files], ignore_index=True) if metrics_files else pd.DataFrame()
    figure_names = [
        "fig_coverage_by_graph.svg",
        "fig_false_best_by_graph.svg",
        "fig_pairwise_calibration.svg",
        "fig_star_vs_triangle_intervention.svg",
        "fig_openml_leaderboard_uncertainty.svg",
        "fig_helm_topk_stability.svg",
    ]
    outputs = {}
    for name in figure_names:
        path = output_dir / name
        _write_placeholder_svg(path, name, metrics)
        outputs[name] = path
    return outputs


def _write_placeholder_svg(path: Path, title: str, metrics: pd.DataFrame) -> None:
    n_rows = 0 if metrics.empty else len(metrics)
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="220">\n'
        f'<text x="32" y="48" font-size="20" font-family="sans-serif">{title}</text>\n'
        f'<text x="32" y="86" font-size="13" font-family="sans-serif">metric rows: {n_rows}</text>\n'
        "</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")
