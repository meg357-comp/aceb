from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OPTIONAL_METHOD_MARKERS = ("mash", "full_bayes", "cmdstan", "rscript")


@dataclass(slots=True)
class GateCheck:
    section: str
    check: str
    status: str
    detail: str


@dataclass(slots=True)
class GateThresholds:
    coverage_low: float = 0.84
    coverage_high: float = 0.96
    min_z_distance_gain: float = 0.0
    min_false_best_gain: float = 0.0


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _status_rank(status: str) -> int:
    return {"PASS": 0, "WARN": 1, "FAIL": 2}.get(status, 1)


def _overall_status(checks: list[GateCheck]) -> str:
    worst = max((_status_rank(check.status) for check in checks), default=1)
    if worst >= 2:
        return "FAIL"
    if worst == 1:
        return "PASS_WITH_WARNINGS"
    return "PASS"


def _json_graph_records(graph_json: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key, value in graph_json.items():
        if isinstance(value, dict):
            record = dict(value)
            record.setdefault("record_id", key)
            records.append(record)
    return records


def _diagnostic_records(method_json: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    records: list[tuple[str, dict[str, Any]]] = []
    direct_keys = {
        "residual_calibration_method",
        "view_scale",
        "extra_noise",
        "success",
        "skipped",
        "failed",
        "graph_diagnostic",
    }
    for key, value in method_json.items():
        if isinstance(value, dict):
            if direct_keys & set(value):
                records.append((key, value))
            elif any(isinstance(inner, dict) for inner in value.values()):
                for method, diag in value.items():
                    if isinstance(diag, dict):
                        records.append((f"{key}:{method}", diag))
            else:
                records.append((key, value))
    return records


def _check_required_outputs(run_dir: Path, manifest: pd.DataFrame, checks: list[GateCheck]) -> None:
    required = ["metrics.csv", "pairwise_calibration.csv", "stratified_coverage.csv", "graph_diagnostics.json"]
    missing = [name for name in required if not (run_dir / name).exists()]
    checks.append(
        GateCheck(
            "hard",
            "run-level required outputs",
            "FAIL" if missing else "PASS",
            "missing: " + ", ".join(missing) if missing else "all run-level outputs present",
        )
    )
    cells_dir = run_dir / "cells"
    if manifest.empty or not cells_dir.exists():
        checks.append(GateCheck("hard", "per-cell output completeness", "WARN", "no manifest/cells directory; skipped cell-level completeness check"))
        return
    missing_cells = []
    for row in manifest.itertuples(index=False):
        if bool(getattr(row, "design_valid", True)) and str(getattr(row, "status", "pending")) != "skipped":
            cell_dir = cells_dir / str(int(getattr(row, "cell_id")))
            missing_files = [name for name in ["metrics.csv", "pairwise_calibration.csv", "stratified_coverage.csv", "graph_diagnostics.json", "status.json"] if not (cell_dir / name).exists()]
            if missing_files:
                missing_cells.append(f"{getattr(row, 'cell_id')}:{','.join(missing_files)}")
    checks.append(
        GateCheck(
            "hard",
            "valid graph cells have artifacts",
            "FAIL" if missing_cells else "PASS",
            "; ".join(missing_cells[:10]) if missing_cells else "all valid cells have required artifacts",
        )
    )


def _check_hard_failures(
    checks: list[GateCheck],
    *,
    graph_records: list[dict[str, Any]],
    failed_cells: pd.DataFrame,
    skipped_cells: pd.DataFrame,
    method_records: list[tuple[str, dict[str, Any]]],
    legacy_run: bool = False,
) -> None:
    mismatches = [
        record.get("record_id", "?")
        for record in graph_records
        if record.get("design_valid", True) is True
        and (record.get("missing_edges") not in (None, [], "") or record.get("extra_edges") not in (None, [], ""))
    ]
    checks.append(GateCheck("hard", "observed graph matches intended graph", "FAIL" if mismatches else "PASS", "; ".join(map(str, mismatches[:10])) if mismatches else "no graph mismatches recorded"))

    invalid_cycle_runs = []
    for record in graph_records:
        name = str(record.get("graph_design", record.get("record_id", "")))
        n_views = int(record.get("n_views", record.get("n_views_config", 0)) or 0)
        valid = bool(record.get("design_valid", True))
        if "even_cycle" in name and (n_views < 4 or n_views % 2 != 0) and valid:
            invalid_cycle_runs.append(f"{name}/J={n_views}")
        if "odd_cycle" in name and (n_views < 3 or n_views % 2 == 0) and valid:
            invalid_cycle_runs.append(f"{name}/J={n_views}")
    invalid_status = "WARN" if legacy_run and invalid_cycle_runs else "FAIL" if invalid_cycle_runs else "PASS"
    invalid_detail = "; ".join(invalid_cycle_runs[:10]) if invalid_cycle_runs else "invalid even/odd cycle combinations were not run"
    if legacy_run and invalid_cycle_runs:
        invalid_detail = "legacy pre-checkpoint run: " + invalid_detail
    checks.append(GateCheck("hard", "invalid cycle designs skipped", invalid_status, invalid_detail))

    if failed_cells.empty:
        checks.append(GateCheck("hard", "failed cells", "PASS", "no failed cells recorded"))
    else:
        text = " ".join(str(value).lower() for value in failed_cells.to_numpy().ravel())
        optional_only = all(marker in text for marker in OPTIONAL_METHOD_MARKERS) if text else False
        checks.append(GateCheck("hard", "failed cells", "WARN" if optional_only else "FAIL", f"{len(failed_cells)} failed cell rows"))

    aceb_records = [(key, diag) for key, diag in method_records if "anchored_calibrated_eb" in key and "in_sample" not in key]
    bad_aceb = [
        key
        for key, diag in aceb_records
        if not diag.get("skipped", False)
        and not diag.get("failed", False)
        and diag.get("residual_calibration_method") != "leave_view"
    ]
    if not aceb_records:
        checks.append(GateCheck("hard", "ACEB leave-view diagnostics", "WARN", "no ACEB diagnostics found"))
    else:
        checks.append(GateCheck("hard", "ACEB leave-view diagnostics", "FAIL" if bad_aceb else "PASS", "; ".join(bad_aceb[:10]) if bad_aceb else "all ACEB diagnostics report leave_view"))


def _check_graph_theorem(metrics: pd.DataFrame, graph_records: list[dict[str, Any]], checks: list[GateCheck], *, legacy_run: bool = False) -> None:
    source = metrics.copy()
    if source.empty and graph_records:
        source = pd.DataFrame(graph_records).rename(columns={"rank_deficiency_delta": "delta_G", "n_components": "component_count"})
    if source.empty or "graph_design" not in source:
        checks.append(GateCheck("graph", "graph theorem diagnostics", "WARN", "no graph-design rows available"))
        return
    if "delta_G" not in source and "graph_rank_deficiency_delta" in source:
        source["delta_G"] = source["graph_rank_deficiency_delta"]
    graph_rows = source.drop_duplicates([col for col in ["graph_design", "n_views_config", "seed", "run_seed"] if col in source])

    def designs(names: set[str]) -> pd.DataFrame:
        return graph_rows.loc[graph_rows["graph_design"].isin(names)]

    bip = designs({"star", "chain", "even_cycle"})
    bad_bip = bip.loc[pd.to_numeric(bip.get("delta_G", np.nan), errors="coerce") <= 0] if not bip.empty else pd.DataFrame()
    bip_status = "WARN" if legacy_run and not bad_bip.empty else "FAIL" if not bad_bip.empty else "PASS"
    checks.append(GateCheck("graph", "bipartite designs have positive delta_G", bip_status, f"{len(bad_bip)} bad rows" if not bad_bip.empty else "star/chain/even_cycle positive where present"))

    full = designs({"odd_cycle", "triangle_plus_tails", "complete", "star_plus_edge_closes_triangle"})
    bad_full = full.loc[pd.to_numeric(full.get("delta_G", np.nan), errors="coerce") > 0] if not full.empty else pd.DataFrame()
    checks.append(GateCheck("graph", "non-bipartite/complete designs are full rank", "FAIL" if not bad_full.empty else "PASS", f"{len(bad_full)} deficient rows" if not bad_full.empty else "delta_G is zero where expected"))

    disc = designs({"disconnected"})
    if disc.empty:
        checks.append(GateCheck("graph", "disconnected graph check", "WARN", "no disconnected rows"))
    else:
        comp_col = "component_count" if "component_count" in disc else "graph_n_components"
        bad_disc = disc.loc[(pd.to_numeric(disc.get(comp_col, np.nan), errors="coerce") <= 1) | (pd.to_numeric(disc.get("delta_G", np.nan), errors="coerce") <= 0)]
        checks.append(GateCheck("graph", "disconnected has multiple components and deficiency", "FAIL" if not bad_disc.empty else "PASS", f"{len(bad_disc)} bad disconnected rows" if not bad_disc.empty else "disconnected rows pass"))


def _mean_metric(metrics: pd.DataFrame, method: str, metric: str, subset: pd.Series | None = None) -> float:
    if metrics.empty or metric not in metrics:
        return float("nan")
    rows = metrics.loc[metrics["method_name"] == method]
    if subset is not None:
        rows = rows.loc[subset.reindex(rows.index, fill_value=False)]
    values = pd.to_numeric(rows[metric], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def _check_method_sanity(metrics: pd.DataFrame, checks: list[GateCheck], thresholds: GateThresholds) -> None:
    if metrics.empty or "method_name" not in metrics:
        checks.append(GateCheck("method", "method sanity", "WARN", "metrics.csv is empty"))
        return
    aceb = "anchored_calibrated_eb"
    joint = "joint_eb_full"
    insample = "anchored_calibrated_eb_in_sample"
    raw = "raw_reference" if "raw_reference" in set(metrics["method_name"]) else "raw_single_reference"

    aceb_cov = _mean_metric(metrics, aceb, "coverage_90")
    cov_ok = thresholds.coverage_low <= aceb_cov <= thresholds.coverage_high if np.isfinite(aceb_cov) else False
    checks.append(GateCheck("method", "ACEB leave-view coverage near nominal", "PASS" if cov_ok else "WARN", f"mean coverage_90={aceb_cov:.4g}; range=[{thresholds.coverage_low},{thresholds.coverage_high}]"))

    aceb_z = abs(_mean_metric(metrics, aceb, "posterior_z_sd") - 1.0)
    joint_z = abs(_mean_metric(metrics, joint, "posterior_z_sd") - 1.0)
    insample_z = abs(_mean_metric(metrics, insample, "posterior_z_sd") - 1.0)
    z_ok = np.isfinite(aceb_z) and (not np.isfinite(joint_z) or aceb_z <= joint_z - thresholds.min_z_distance_gain) and (not np.isfinite(insample_z) or aceb_z <= insample_z - thresholds.min_z_distance_gain)
    checks.append(GateCheck("method", "ACEB z-score SD closer to 1", "PASS" if z_ok else "WARN", f"abs gaps: ACEB={aceb_z:.4g}, joint={joint_z:.4g}, in_sample={insample_z:.4g}"))

    confound = metrics.get("loading_spread", pd.Series(False, index=metrics.index)).astype(str).isin(["1.0", "1"])
    joint_cov = _mean_metric(metrics, joint, "coverage_90", confound if "loading_spread" in metrics else None)
    checks.append(GateCheck("method", "joint EB under-covers in high-confounding cells", "PASS" if np.isfinite(joint_cov) and joint_cov < thresholds.coverage_low else "WARN", f"joint high-confounding coverage_90={joint_cov:.4g}"))

    insample_cov = _mean_metric(metrics, insample, "coverage_90")
    checks.append(GateCheck("method", "in-sample ACEB under-covers relative to leave-view", "PASS" if np.isfinite(insample_cov) and np.isfinite(aceb_cov) and insample_cov < aceb_cov else "WARN", f"in_sample={insample_cov:.4g}, leave_view={aceb_cov:.4g}"))

    aceb_fb = _mean_metric(metrics, aceb, "false_best_rate_095" if "false_best_rate_095" in metrics else "false_best_rate")
    raw_fb = _mean_metric(metrics, raw, "false_best_rate_095" if "false_best_rate_095" in metrics else "false_best_rate")
    joint_fb = _mean_metric(metrics, joint, "false_best_rate_095" if "false_best_rate_095" in metrics else "false_best_rate")
    fb_ok = np.isfinite(aceb_fb) and (not np.isfinite(raw_fb) or aceb_fb <= raw_fb - thresholds.min_false_best_gain) and (not np.isfinite(joint_fb) or aceb_fb <= joint_fb - thresholds.min_false_best_gain)
    checks.append(GateCheck("method", "ACEB false-best lower than raw and joint", "PASS" if fb_ok else "WARN", f"ACEB={aceb_fb:.4g}, raw={raw_fb:.4g}, joint={joint_fb:.4g}"))


def _bootstrap_ci(values: np.ndarray, random_state: int = 0, n_boot: int = 1000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(random_state)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _check_intervention(run_dir: Path, checks: list[GateCheck]) -> pd.DataFrame:
    deltas_path = run_dir / "intervention_deltas.csv"
    if not deltas_path.exists():
        alt = run_dir / "paired_intervention" / "intervention_deltas.csv"
        deltas_path = alt if alt.exists() else deltas_path
    deltas = _read_csv(deltas_path)
    if deltas.empty:
        checks.append(GateCheck("intervention", "paired star intervention", "WARN", "no intervention_deltas.csv found"))
        return pd.DataFrame()
    plus = deltas.loc[deltas.get("plus_arm", "") == "plus_additive"].copy()
    if plus.empty:
        plus = deltas.copy()
    dg = pd.to_numeric(plus.get("delta_delta_G", np.nan), errors="coerce")
    sv_col = "delta_smallest_full_singular_value" if "delta_smallest_full_singular_value" in plus else "delta_min_singular_value"
    sv = pd.to_numeric(plus.get(sv_col, np.nan), errors="coerce")
    checks.append(GateCheck("intervention", "delta_G improves", "FAIL" if not np.any(dg < 0) else "PASS", f"fraction improved={np.mean(dg < 0):.3f}" if dg.notna().any() else "missing delta_delta_G"))
    checks.append(GateCheck("intervention", "smallest full singular value improves", "WARN" if not np.any(sv > 0) else "PASS", f"{sv_col}; fraction improved={np.mean(sv > 0):.3f}" if sv.notna().any() else f"missing {sv_col}"))
    recovery_cols = [col for col in ["delta_loading_rmse", "delta_log_loading_rmse"] if col in plus]
    recovery_improved = any(np.nanmean(pd.to_numeric(plus[col], errors="coerce") < 0) > 0.5 for col in recovery_cols)
    decision_cols = [col for col in ["delta_false_best_rate", "delta_close_pair_ece", "delta_pairwise_brier_score", "delta_close_pair_brier_score"] if col in plus]
    decision_improved = any(np.nanmean(pd.to_numeric(plus[col], errors="coerce") < 0) > 0.5 for col in decision_cols)
    status = "PASS" if recovery_improved and decision_improved else "WARN"
    checks.append(GateCheck("intervention", "recovery and close-decision metrics improve", status, f"recovery_improved={recovery_improved}; decision_improved={decision_improved}; all-pair ECE is not a hard criterion"))
    return plus


def _markdown_table(rows: list[GateCheck], section: str) -> str:
    selected = [row for row in rows if row.section == section]
    if not selected:
        return "_No checks._"
    lines = ["| check | status | detail |", "| --- | --- | --- |"]
    for row in selected:
        detail = row.detail.replace("\n", " ").replace("|", "\\|")
        lines.append(f"| {row.check} | {row.status} | {detail} |")
    return "\n".join(lines)


def _intervention_summary_table(deltas: pd.DataFrame) -> str:
    if deltas.empty:
        return "_No paired intervention deltas available._"
    metrics = [
        col
        for col in [
            "delta_delta_G",
            "delta_smallest_full_singular_value",
            "delta_normalized_smallest_full_singular_value",
            "delta_min_singular_value",
            "delta_loading_rmse",
            "delta_log_loading_rmse",
            "delta_false_best_rate",
            "delta_close_pair_ece",
            "delta_pairwise_brier_score",
        ]
        if col in deltas
    ]
    lines = ["| metric | mean_delta | ci_low | ci_high | improve_fraction |", "| --- | ---: | ---: | ---: | ---: |"]
    for metric in metrics:
        values = pd.to_numeric(deltas[metric], errors="coerce").dropna().to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        ci_low, ci_high = _bootstrap_ci(values)
        higher_better = metric in {
            "delta_smallest_full_singular_value",
            "delta_normalized_smallest_full_singular_value",
            "delta_min_singular_value",
        }
        improve = np.mean(values > 0) if higher_better else np.mean(values < 0)
        lines.append(f"| {metric} | {np.mean(values):.4g} | {ci_low:.4g} | {ci_high:.4g} | {improve:.3f} |")
    transition_cols = [col for col in ["condition_number_transition", "normalized_condition_number_transition"] if col in deltas]
    if transition_cols:
        lines.extend(
            [
                "",
                "| condition_metric | frac_infinite_to_finite | frac_finite_improved | frac_finite_worsened | frac_infinite_to_infinite |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for transition_col in transition_cols:
            metric = transition_col.replace("_transition", "")
            transition = deltas[transition_col].astype(str)
            finite = transition == "finite_to_finite"
            improved = deltas.get(f"{metric}_improved", pd.Series(False, index=deltas.index)).astype(bool)
            denom = max(len(deltas), 1)
            lines.append(
                f"| {metric} | "
                f"{float(np.mean(transition == 'infinite_to_finite')):.3f} | "
                f"{float(np.sum(finite & improved) / denom):.3f} | "
                f"{float(np.sum(finite & ~improved) / denom):.3f} | "
                f"{float(np.mean(transition == 'infinite_to_infinite')):.3f} |"
            )
    return "\n".join(lines)


def run_synthetic_small_gate(run_dir: str | Path, thresholds: GateThresholds | None = None) -> tuple[str, Path, list[GateCheck]]:
    run_dir = Path(run_dir)
    thresholds = thresholds or GateThresholds()
    metrics = _read_csv(run_dir / "metrics.csv")
    pairwise = _read_csv(run_dir / "pairwise_calibration.csv")
    stratified = _read_csv(run_dir / "stratified_coverage.csv")
    failed_cells = _read_csv(run_dir / "failed_cells.csv")
    skipped_cells = _read_csv(run_dir / "skipped_cells.csv")
    manifest = _read_csv(run_dir / "manifest.csv")
    graph_records = _json_graph_records(_read_json(run_dir / "graph_diagnostics.json"))
    method_records = _diagnostic_records(_read_json(run_dir / "method_diagnostics.json"))

    checks: list[GateCheck] = []
    _check_required_outputs(run_dir, manifest, checks)
    legacy_run = manifest.empty or not (run_dir / "cells").exists()
    _check_hard_failures(
        checks,
        graph_records=graph_records,
        failed_cells=failed_cells,
        skipped_cells=skipped_cells,
        method_records=method_records,
        legacy_run=legacy_run,
    )
    _check_graph_theorem(metrics, graph_records, checks, legacy_run=legacy_run)
    _check_method_sanity(metrics, checks, thresholds)
    intervention_deltas = _check_intervention(run_dir, checks)

    status = _overall_status(checks)
    recommendation = "ready for medium" if status == "PASS" else "fix before medium" if status == "FAIL" else "review warnings before medium"
    lines = [
        "# Synthetic Small Gate Report",
        "",
        f"Run directory: `{run_dir}`",
        f"Status: **{status}**",
        f"Recommendation: **{recommendation}**",
        "",
        "Pairwise ECE is treated as a secondary diagnostic: it can be dominated by easy, far-apart pairs. The gate prioritizes graph validity, loading/noise recovery, interval calibration, close-pair decisions, and false-best behavior.",
        "",
        "## Input Summary",
        "",
        f"- metrics rows: {len(metrics)}",
        f"- pairwise calibration rows: {len(pairwise)}",
        f"- stratified coverage rows: {len(stratified)}",
        f"- graph diagnostic records: {len(graph_records)}",
        f"- method diagnostic records: {len(method_records)}",
        f"- skipped cell rows: {len(skipped_cells)}",
        f"- failed cell rows: {len(failed_cells)}",
        "",
        "## Hard Checks",
        "",
        _markdown_table(checks, "hard"),
        "",
        "## Graph Checks",
        "",
        _markdown_table(checks, "graph"),
        "",
        "## Method Sanity Checks",
        "",
        _markdown_table(checks, "method"),
        "",
        "## Intervention Checks",
        "",
        _markdown_table(checks, "intervention"),
        "",
        "## Paired Intervention Deltas",
        "",
        _intervention_summary_table(intervention_deltas),
        "",
    ]
    report_path = run_dir / "SYNTHETIC_SMALL_GATE_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return status, report_path, checks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check whether the small synthetic graph stage is ready for medium.")
    parser.add_argument("--run-dir", type=Path, default=Path("results/synthetic_graph/synthetic_graph_small_full"))
    parser.add_argument("--coverage-low", type=float, default=0.84)
    parser.add_argument("--coverage-high", type=float, default=0.96)
    args = parser.parse_args(argv)
    status, report_path, _ = run_synthetic_small_gate(
        args.run_dir,
        GateThresholds(coverage_low=args.coverage_low, coverage_high=args.coverage_high),
    )
    print(f"{status}: {report_path}")
    raise SystemExit(1 if status == "FAIL" else 0)


if __name__ == "__main__":
    main()
