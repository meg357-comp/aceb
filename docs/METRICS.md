# Metrics

`cmveb` reports scalar posterior metrics, leaderboard/ranking metrics, pairwise calibration, stratified coverage, and graph diagnostics.

## Scalar Metrics

- `rmse`: root mean squared error of posterior mean.
- `mae`: mean absolute error of posterior mean.
- `bias`: mean posterior mean error.
- `correlation`: Pearson correlation between truth and posterior mean.
- `spearman_correlation`: Spearman rank correlation.
- `kendall_tau`: Kendall rank correlation.
- `sign_accuracy`: fraction with matching signs.
- `coverage_90`: empirical 90% interval coverage.
- `posterior_z_mean`: mean of `(posterior_mean - truth) / posterior_sd`.
- `posterior_z_sd`: standard deviation of posterior z-scores.
- `z_score_sd_abs_error`: `abs(posterior_z_sd - 1)`.
- `variance_ratio`: mean posterior variance divided by mean squared error.
- `average_interval_width_90`, `median_interval_width_90`: 90% interval widths.
- `coverage_width_score`: proper interval score; it rewards narrow intervals only when they cover truth.
- `predictive_log_likelihood`: row-level predictive log likelihood under method diagnostics.

## Synthetic Parameter Recovery

Graph-identifiability claims are primarily about recovering view loadings and calibrated uncertainty, not only pairwise ECE. Synthetic graph runs therefore report, when truth is available:

- `loading_rmse`, `log_loading_rmse`, `loading_mae`, `loading_corr`
- `tau_rmse`, `tau_mae`
- `bias_rmse`, when true additive view bias is available
- `prior_scale_error`, when a comparable prior scale is reported by a method

Methods that do not estimate view loadings or extra noise receive `NaN` for these metrics rather than failing the benchmark.

## Ranking Metrics

- `top_k_overlap`: legacy top 10% overlap.
- `top_1_overlap`, `top_5_overlap`, `top_10_overlap`, etc.: overlap between true and estimated top-k sets.
- `ndcg_at_k`: normalized discounted cumulative gain at k, implemented without sklearn.
- `posterior_rank_entropy`: Monte Carlo posterior rank-distribution entropy.
- `false_best_rate`: among items declared best with posterior best probability above a threshold, fraction that are not the true best.
- `false_best_rate_075`, `false_best_rate_090`, `false_best_rate_095`: false-best rates at decision thresholds.
- `abstention_rate_075`, `abstention_rate_090`, `abstention_rate_095`: fraction of no-decision items at each threshold.
- `selective_accuracy_075`, `selective_accuracy_090`, `selective_accuracy_095`: accuracy among declared winners.
- `true_best_in_credible_top_1`, `true_best_in_credible_top_3`, `true_best_in_credible_top_5`: whether the true best is in the posterior mean top-k set.
- `probability_select_true_best`: posterior probability assigned to the true best item being best.

## Pairwise Win Probabilities

For independent plug-in Gaussian posteriors:

```text
pi_ik = Phi((m_i - m_k) / sqrt(V_i + V_k))
```

`pairwise_calibration.csv` bins `pi_ik` and compares it with:

```text
1{theta_i > theta_k}
```

Columns:

- `pair_subset`: `all_pairs`, `close_pairs`, or another labeled subset.
- `bin_left`, `bin_right`
- `mean_pred`
- `empirical_win_rate`
- `n_pairs`
- `pairwise_ece`

Pairwise ECE is secondary for the graph-identifiability theorem because most item pairs are easy and far apart; a method can look well calibrated overall while still failing on scale/loadings or on close leaderboard decisions. The primary synthetic diagnostics are loading recovery, graph conditioning, interval score, close-pair metrics, and selective false-best behavior.

Additional scalar pairwise metrics in `metrics.csv` include:

- `pairwise_brier_score`, `pairwise_log_score`
- `all_pair_ece`
- `close_pair_ece`, `close_pair_brier_score`, `close_pair_log_score`
- `top_decile_pair_ece`

## Stratified Coverage

`stratified_coverage.csv` reports coverage by:

- view-count quantiles
- posterior-variance quantiles
- absolute-effect quantiles
- true-rank bins
- graph component, when graph diagnostics are available

## Graph Diagnostics In Metric Rows

`metrics.csv` includes graph-level diagnostics copied from `graph_diagnostic`:

- `graph_n_views`
- `graph_n_edges`
- `graph_n_components`
- `graph_anchor_component_is_non_bipartite`
- `graph_rank_deficiency_delta`
- `graph_signless_incidence_rank`
- `graph_design_rank`, `graph_design_num_columns`, `delta_G`
- `smallest_singular_value`, `largest_singular_value`, `condition_number`
- `component_count`
- `anchor_component_bipartite`, `anchor_component_has_odd_cycle`

## Calibration And Benchmark Uncertainty Baselines

`posthoc_inverse_variance_all_views` splits items into fit and calibration sets, estimates a scalar variance inflation

```text
c^2 = max(1, mean_i z_i^2)
```

and multiplies final posterior variances by `c^2`. In synthetic runs, `z_i` uses known truth. Without truth, the wrapper falls back to held-out-view residual calibration.

Bootstrap baselines are available as opt-in synthetic benchmark ablations with `--include-bootstraps`:

- `bootstrap_inverse_variance_all_views`
- `bootstrap_reference_view`
- `bootstrap_aceb_variance_correction`

They sample summary rows parametrically as `y_ij^b ~ Normal(y_ij, s_ij^2)`, refit the base method, and add the bootstrap variance of posterior means to the base posterior variance.
