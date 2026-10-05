# ACEB Implementation Contract Audit

Audit date: 2026-07-10

This audit records the implementation contract actually used by the ACEB experiments in this repository. It is intentionally factual and code-driven: when a value was not present in code, configs, or persisted diagnostics, it is marked `NOT_RECOVERED` rather than inferred from the paper draft.

## Executive Summary

The reported ACEB method is the staged implementation in `cmveb.models.anchored_eb.fit_staged_calibrated_eb`, exposed through `cmveb.models.baseline.anchored_calibrated_eb`. It uses a fixed reference view, feature-based Gaussian empirical-Bayes marginal likelihood, L2 penalties on prior parameters and non-reference loading deviations, and a per-view positive-part residual-noise estimate. The default and main reported residual calibration is `leave_view`; the in-sample ablation explicitly calls the same fitter with `residual_calibration="in_sample"`.

There is no implemented admissible-anchor optimizer in the main fit path. The anchor is whichever single view is marked `is_reference_view=True`; synthetic and adapter-generated real-data runs use view 0 / first sorted view as the reference. Multi-anchor fitting and anchor-stable decision intersection were not used in the reported main results. Anchor sweeps were a separate ablation.

The graph diagnostic edge rule is threshold-only on shared items: an edge is included when a view pair shares at least `min_shared_items` items. Empirical cross-view moments are centered and recorded, and a positivity flag is written, but positive moment is not required for edge inclusion.

Post-hoc inverse variance is not uniformly an unsupervised baseline. In the synthetic graph runner and MedHELM runner, truth is passed to `posthoc_inverse_variance_all_views`, so that baseline is target-calibrated/transductive there. In SummEval and the same-coverage/negative-control scripts, it is called with `truth=None` before any explicit calibration split.

## Implementation Locations

| Quantity | Value used | Source | Applies to | Uncertainty / not found |
|---|---|---|---|---|
| ACEB entry point | `anchored_calibrated_eb` wraps `fit_staged_calibrated_eb` | `src/cmveb/models/baseline.py:319-351`; `src/cmveb/models/anchored_eb.py:271-367` | all ACEB reported runs | recovered |
| ACEB class wrapper | `AnchoredEBModel.fit` calls staged fitter | `src/cmveb/models/anchored_eb.py:370-401` | API wrapper, not necessary for scripts | recovered |
| MedHELM runner | `run_helm_benchmark` | `src/cmveb/benchmarks/helm_runner.py:756-920` | MedHELM real run | recovered |
| Synthetic graph runner | `run_graph_synthetic_benchmark`, `run_graph_methods` | `src/cmveb/benchmarks/graph_synthetic.py:1211-1275,1357-1450` | main graph/intervention/component ablations | recovered |
| SummEval runner | `scripts/run_summeval_appendix_pilot.py` | `scripts/run_summeval_appendix_pilot.py:54-62,184-205,518-526` | SummEval appendix pilot | recovered |

## Graph Edge Construction

The graph diagnostic function is `build_coobservation_graph`. For each view pair `(j,k)`, the implementation constructs the shared item set. If `n_shared < min_shared_items`, the pair is skipped. Otherwise the edge is included and a centered empirical moment is recorded:

`moment = mean((y_j - view_mean_j) * (y_k - view_mean_k))`.

The moment is flagged with `edge_is_positive = moment > min_positive_moment`, where `min_positive_moment=1e-12`, but this flag does not gate edge inclusion. Zero and negative moments are therefore retained as graph edges if they pass the shared-item threshold. There is no edge standard error and no significance threshold. Source: `src/cmveb/graph_diagnostics.py:310-404`.

Singular-value diagnostics use the signless incidence design matrix. The full-column rank tolerance is `max(1e-7, largest_singular_value * 1e-7)`, and the normalized smallest singular value is `smallest_full_singular_value / sqrt(max(n_edges, 1))`. Source: `src/cmveb/graph_diagnostics.py:225-268`.

| Experiment family | `min_shared_items` | Source | Notes |
|---|---:|---|---|
| Main synthetic graph | 1 | `src/cmveb/benchmarks/graph_synthetic.py:1433-1434` | `configs/synthetic_graph_main_paper.yaml` does not override it. |
| Synthetic intervention | 1 | `src/cmveb/benchmarks/graph_synthetic.py:780`; `src/cmveb/benchmarks/graph_synthetic.py:2166` | Paired intervention diagnostics use threshold 1. |
| Component/oracle ablation | 1 | `configs/synthetic_aceb_component_oracle_ablation.yaml`; `src/cmveb/benchmarks/graph_synthetic.py:1188` | Graph diagnostics attached to component outputs. |
| Anchor sensitivity | 1 | `configs/synthetic_anchor_sensitivity.yaml`; `scripts/run_anchor_sensitivity_ablation.py:275-288` | Anchor sweep validates and summarizes with threshold 1. |
| Misspecification compact | 1 | `configs/synthetic_misspec_compact.yaml`; `scripts/run_misspec_compact_ablation.py:275-280` | Threshold 1. |
| Edge-selection ablation | 1 | `configs/synthetic_graph_edge_selection.yaml`; `scripts/run_edge_selection_ablation.py:341-345` | Threshold 1. |
| Same-coverage synthetic | `NOT_APPLICABLE` | `scripts/run_synthetic_same_coverage_recalibration.py:65-84,440-455` | Uses simulated graph classes; no separate co-observation graph diagnostic is built. |
| Negative control | `NOT_APPLICABLE` | `scripts/run_negative_control_ablation.py:44-58,65-113` | Dense complete observations; no graph diagnostic threshold is used. |
| MedHELM full fast | 10 | `configs/helm_real_pairwise_gap_full_fast.yaml`; `src/cmveb/benchmarks/helm_runner.py:766` | Run-level graph diagnostics use 10; baseline-attached method diagnostics use 1 internally. |
| MedHELM joint subset | 10 | `configs/helm_real_pairwise_gap_joint_subset.yaml`; `src/cmveb/benchmarks/helm_runner.py:766` | Same graph threshold as full-fast. |
| SummEval appendix | 1 in method diagnostics | `src/cmveb/models/baseline.py:35-38`; `results/summeval/appendix_pilot/method_diagnostics.json` | No separate graph-regime output. |

## Anchor Rule

| Quantity | Value used | Source | Applies to | Uncertainty / not found |
|---|---|---|---|---|
| Candidate anchor set | Exactly one view in `view_metadata` marked `is_reference_view=True` | `src/cmveb/schemas.py:121-130`; `src/cmveb/indexing.py:17-55` | all ACEB runs | recovered |
| Admissibility conditions | No degree, overlap, rank, or conditioning threshold in fit path | `src/cmveb/schemas.py:121-130` | all ACEB runs | recovered |
| If no admissible anchor | Validation error if zero or multiple reference views | `src/cmveb/schemas.py:127-130` | all ACEB runs | recovered |
| Synthetic primary anchor | view 0 | `src/cmveb/benchmarks/graph_synthetic.py:1402,417` | synthetic graph/intervention/ablations | recovered |
| MedHELM primary anchor | first sorted `view_id` is reference | `src/cmveb/datasets/helm_adapter.py:366-379` | HELM/MedHELM | recovered |
| Multi-anchor fitting | Not used in main results | absence in main runners; anchor sweep uses `scripts/run_anchor_sensitivity_ablation.py:296-308` | main results | recovered as not used |
| Anchor-stable decision intersection | Not used | no implementation path found in inspected runners | main results | recovered as not used |
| Main leaderboard decision threshold | 0.95 | `configs/helm_real_pairwise_gap_full_fast.yaml`; `src/cmveb/benchmarks/helm_runner.py:918` | HELM grouped leaderboard | recovered |
| Selective-decision thresholds | 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99 | `src/cmveb/evaluate.py:304-363` | real pairwise decision curves | recovered |

## ACEB Objective and Penalty

The staged ACEB fitter has three stages.

1. Prior hyperparameters are fit by minimizing negative Gaussian marginal log likelihood plus
   `l2_prior * (sum(w_mu^2) + sum(w_v^2))`.
   The default `l2_prior` is `1e-4`.
   Source: `src/cmveb/models/anchored_eb.py:45-68`; `src/cmveb/config.py:35`.

2. Non-reference view loadings are fit by minimizing negative Gaussian marginal log likelihood plus
   `l2_view_scale * sum((a_j - 1)^2)` over non-reference views. The reference view loading is fixed at 1. The default `l2_view_scale` is `1e-4`.
   Source: `src/cmveb/models/anchored_eb.py:71-102`; `src/cmveb/config.py:36`.

3. Per-view residual noise `tau_j` is estimated by residual moments rather than likelihood optimization. Source: `src/cmveb/models/anchored_eb.py:330-361`; `src/cmveb/calibration.py:95-163`.

Bias terms `b_j` exist in the code, but the default is `fit_view_bias=False`, so reported ACEB diagnostics sampled from main/intervention/SummEval runs have zero fitted bias and `center_reference=False`. Source: `src/cmveb/config.py:41-42`; `src/cmveb/models/anchored_eb.py:322-328,363-366`.

The prior mean is feature-based (`X @ w_mu`), and the prior variance is `softplus(X @ w_v) + eps`. If no item features exist, an intercept is inserted; otherwise an intercept is added and non-intercept features are standardized by default. Source: `src/cmveb/posterior.py:19-28`; `src/cmveb/preprocess.py:27-47`.

No config override of `l2_prior` or `l2_view_scale` was found in the reported configs inspected for this audit.

## Transformations, Floors, and Clipping

| Quantity | Value used | Source | Applies to | Uncertainty / not found |
|---|---|---|---|---|
| Enforce `a_j > 0` | Optimize `log_scale`; set `a_j=exp(log_scale)` for non-reference views; reference fixed to 1 | `src/cmveb/models/anchored_eb.py:87-101` | ACEB | recovered |
| Prior variance | `softplus(X @ w_v) + eps`, `eps=1e-8` | `src/cmveb/posterior.py:19-28`; `src/cmveb/config.py:28` | ACEB and EB baselines | recovered |
| Initial non-reference loading | 0.5 | `src/cmveb/models/anchored_eb.py:20-28`; `src/cmveb/config.py:30` | ACEB | recovered |
| Initial `tau_j` | `1e-4` | `src/cmveb/models/anchored_eb.py:31-32`; `src/cmveb/config.py:31` | ACEB | recovered |
| `tau_j` lower/upper | lower 0, upper 10 | `src/cmveb/config.py:32-33`; `src/cmveb/calibration.py:152-154` | ACEB | recovered |
| Positive-part `tau` truncation | `tau2=max(tau2_raw,0)` before clipping | `src/cmveb/calibration.py:134-163` | ACEB | recovered |
| Preprocess SE floor | `1e-8` | `src/cmveb/config.py:19-23`; `src/cmveb/preprocess.py:10-25` | all prepared data | recovered |
| HELM standard-error floor | 0.02 in full-fast config | `configs/helm_real_pairwise_gap_full_fast.yaml`; `src/cmveb/datasets/helm_adapter.py:58-69,107-133` | MedHELM full-fast | recovered |
| SummEval observation SE | fixed 0.75 | `scripts/run_summeval_appendix_pilot.py:54-58,520-525` | SummEval | recovered |
| SummEval target SE floor | 0.10 | `scripts/run_summeval_appendix_pilot.py:54-58,100-109,520-525` | SummEval | recovered |
| Posterior variance floor in output/evaluation | `1e-18` | `src/cmveb/models/baseline.py:71-80`; `src/cmveb/evaluate.py:28-31` | metrics/posteriors | recovered |
| Leave-view variance floor | `1e-12` | `src/cmveb/calibration.py:51-92` | leave-view tau calibration | recovered |
| Pairwise probability clipping | `[1e-8, 1 - 1e-8]` | `src/cmveb/evaluate.py:366-413` | pairwise calibration ECE | recovered |
| Bounded-binomial logit clipping | `NOT_RECOVERED` | Config mentions bounded-binomial logit; exact clipping implementation line not inspected | misspecification ablation | not recovered |

## Optimizer and Initialization

The optimizer is SciPy L-BFGS-B. The code does not pass a `jac`, so gradients are numerical finite-difference gradients supplied by SciPy. The only explicit stopping option is `maxiter`; `ftol` and `gtol` are not set, so their numerical values are SciPy defaults and are not recovered from the repository.

Source: `src/cmveb/models/anchored_eb.py:66,98`; joint EB source `src/cmveb/models/baseline.py:306`.

There is no restart loop in the staged ACEB prior or loading fits.

Initialization:

- reference loading: 1.0
- non-reference loading: 0.5
- extra residual noise: `1e-4`
- `w_mu`: zeros
- `w_v[0]`: inverse-softplus of the variance of centered estimates
- bias: zeros unless `fit_view_bias=True`

Source: `src/cmveb/models/anchored_eb.py:20-42`; `src/cmveb/config.py:27-42`.

| Experiment family | ACEB `max_iter` | Source |
|---|---:|---|
| Main synthetic graph | 8 | `configs/synthetic_graph_main_paper.yaml` |
| Synthetic intervention | 8 | `configs/synthetic_graph_intervention_main_paper.yaml` |
| Component/oracle ablation | 40 | `configs/synthetic_aceb_component_oracle_ablation.yaml` |
| Anchor sensitivity | 40 | `configs/synthetic_anchor_sensitivity.yaml` |
| Misspecification compact | 40 | `configs/synthetic_misspec_compact.yaml` |
| Edge selection | 35 | `configs/synthetic_graph_edge_selection.yaml` |
| Same-coverage synthetic | 100 | `scripts/run_synthetic_same_coverage_recalibration.sh:12`; `scripts/run_synthetic_same_coverage_recalibration.py:65-84` |
| Negative control | 100 | `scripts/run_negative_control_ablation.sh:14`; `scripts/run_negative_control_ablation.py:44-58` |
| MedHELM full fast | 3 | `configs/helm_real_pairwise_gap_full_fast.yaml` |
| MedHELM joint subset | 3 | `configs/helm_real_pairwise_gap_joint_subset.yaml` |
| SummEval appendix | 80 | `scripts/run_summeval_appendix_pilot.py:54-62` |

The staged ACEB implementation does not check `result.success` from the SciPy optimizer before continuing. It records objective history and tau diagnostics. Runners catch exceptions and write failed outputs unless `strict=True`. Source: `src/cmveb/models/anchored_eb.py:300-367`; `src/cmveb/benchmarks/helm_runner.py:403-430`; `src/cmveb/benchmarks/graph_synthetic.py:1237-1262`.

## Residual Calibration

Default residual calibration is `leave_view`. Source: `src/cmveb/config.py:37`; `src/cmveb/models/anchored_eb.py:283-291`.

For each row, the leave-view posterior is computed by subtracting the row's sufficient-statistic contribution from the item posterior precision and natural parameter. Then

`tau2_raw_j = mean((y_ij - b_j - a_j m_{i,-row})^2 - a_j^2 V_{i,-row} - s_ij^2)`

over rows in view `j`. The implemented `tau2_j` is the positive part of this raw value, clipped to the configured lower/upper bounds. Source: `src/cmveb/calibration.py:51-163`.

The in-sample ablation uses:

`(y_ij - b_j - a_j m_i)^2 + a_j^2 V_i - s_ij^2`

summarized by view with the same positive-part/clipping step. Source: `src/cmveb/models/anchored_eb.py:135-159`.

`unit_crossfit` is implemented but was not used in the reported main results. It refits nuisance parameters on item folds, then uses leave-view posterior on held-out items. Source: `src/cmveb/models/anchored_eb.py:162-249`.

The default tau scope is per-view. A global tau variant exists only in the component/oracle ablation (`aceb_global_tau`), where leave-view tau estimates are pooled across views. Source: `src/cmveb/benchmarks/graph_synthetic.py:1152-1169`.

After tau is updated, posterior summaries are recomputed. Source: `src/cmveb/models/anchored_eb.py:360-362`.

## Baseline Settings

| Baseline | Exact implementation | Target labels used? | Source |
|---|---|---|---|
| Inverse variance | Per-item inverse-variance mean over all observed views; posterior variance is reciprocal total precision with `1e-18` precision floor. | No | `src/cmveb/models/baseline.py:187-199` |
| Random effects | Per-item DerSimonian-Laird-style variance `max(0,(Q-(n-1))/c)`, then weights `1/(se^2 + random_effect_variance_i)`. | No | `src/cmveb/models/baseline.py:354-384` |
| ASH-like | Collapse views to inverse-variance estimate/variance; fit zero-centered normal mixture on grid `[0] + geomspace(max_abs/20, max_abs*2, 12)` by EM. | No | `src/cmveb/models/baseline.py:395-450` |
| Post-hoc IV | Inflate inverse-variance posterior variance by scalar `c^2`; if truth is passed, `c^2=max(1,mean(z^2))` on a random calibration subset; otherwise use held-out views. | Yes in synthetic graph runner and MedHELM runner; no in SummEval and same-coverage/negative-control pre-calibration calls | `src/cmveb/models/baseline.py:453-518`; `src/cmveb/benchmarks/helm_runner.py:395-397`; `src/cmveb/benchmarks/graph_synthetic.py:1248-1249`; `scripts/run_summeval_appendix_pilot.py:200-202` |
| Joint EB full | Joint L-BFGS-B marginal likelihood over feature prior, non-reference log loadings, and log tau; reference loading fixed to 1; tau upper-clipped at 10; penalty `1e-4` on feature prior parameters only. | No target labels | `src/cmveb/models/baseline.py:267-316` |

MedHELM full-fast excluded `joint_eb_full` as computationally infeasible; the joint subset config uses a `joint_eb_full_timeout_seconds` of 1800. Sources: `configs/helm_real_pairwise_gap_full_fast.yaml`; `configs/helm_real_pairwise_gap_joint_subset.yaml`; `src/cmveb/benchmarks/helm_runner.py:790-800`.

## Real-Data Specific Contract

HELM/MedHELM requires `higher_is_better` by default. Lower-is-better metrics are multiplied by `-1` for both estimates and targets, and raw values are preserved. Source: `src/cmveb/datasets/helm_adapter.py:168-196`.

HELM/MedHELM raises an error if `target_estimate` is missing unless `allow_view_mean_truth=True`. Full-fast uses `allow_view_mean_truth: false`. Source: `src/cmveb/datasets/helm_adapter.py:199-210`; `configs/helm_real_pairwise_gap_full_fast.yaml`.

Pairwise-gap item ids are constructed as:

`model_a + "__minus__" + model_b + "__" + scenario_id + "__" + metric_name`

Source: `src/cmveb/datasets/helm_adapter.py:299-307`.

Target-aware real-data metrics use denominator:

`sqrt(posterior_var + target_standard_error^2)`

Source: `src/cmveb/evaluate.py:139-227`.

## Values Not Recovered

| Quantity | Status | Why |
|---|---|---|
| Exact command line for every reported run | `NOT_RECOVERED` | Some run scripts/configs exist, but a complete command log for every artifact was not found in inspected paths. |
| SciPy `ftol` and `gtol` | `SCIPY_DEFAULT_NOT_EXPLICITLY_SET` | The code only sets `options={"maxiter": max_iter}`. |
| Formal admissible-anchor function beyond reference-view validation | `NOT_RECOVERED` / not implemented | No function or threshold was found in the actual ACEB fit path. |
| Anchor-stable decision intersection | not used | No implementation path found in inspected main runners. |
| Exact bounded-binomial logit clipping line | `NOT_RECOVERED` | Config describes the stressor; exact implementation line was not inspected in this audit. |

