from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm, spearmanr

from cmveb.ranking import (
    best_probabilities,
    false_best_rate,
    pairwise_win_probabilities,
    posterior_rank_entropy,
    probability_select_true_best,
    topk_probabilities_normal_approx,
)
from cmveb.schemas import PosteriorState


def _truth_theta(truth_df: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(truth_df, pd.DataFrame):
        if "theta" in truth_df.columns:
            return truth_df["theta"].to_numpy(dtype=np.float64)
        if "truth" in truth_df.columns:
            return truth_df["truth"].to_numpy(dtype=np.float64)
        raise ValueError("truth dataframe must contain theta or truth.")
    return np.asarray(truth_df, dtype=np.float64)


def _posterior_arrays(posterior_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mean = posterior_df["posterior_mean"].to_numpy(dtype=np.float64)
    variance = np.maximum(posterior_df["posterior_var"].to_numpy(dtype=np.float64), 1e-18)
    return mean, variance


def posterior_metrics(posterior: PosteriorState, truth: pd.DataFrame | np.ndarray) -> dict[str, float]:
    theta = _truth_theta(truth)
    error = posterior.mean - theta
    lower = posterior.mean - 1.96 * np.sqrt(posterior.variance)
    upper = posterior.mean + 1.96 * np.sqrt(posterior.variance)
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "bias": float(np.mean(error)),
        "coverage_95": float(np.mean((theta >= lower) & (theta <= upper))),
    }


def evaluate_point_metrics(posterior_df: pd.DataFrame, truth_df: pd.DataFrame) -> dict[str, float]:
    theta = _truth_theta(truth_df)
    mean, variance = _posterior_arrays(posterior_df)
    error = mean - theta
    spear = spearmanr(theta, mean).statistic
    kend = kendalltau(theta, mean).statistic
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "correlation": float(np.corrcoef(theta, mean)[0, 1]) if np.std(theta) > 0.0 and np.std(mean) > 0.0 else np.nan,
        "sign_accuracy": float(np.mean(np.sign(theta) == np.sign(mean))),
        "spearman_correlation": float(spear) if np.isfinite(spear) else np.nan,
        "kendall_tau": float(kend) if np.isfinite(kend) else np.nan,
        "variance_ratio": float(np.mean(variance) / max(float(np.mean(error * error)), 1e-18)),
    }


def evaluate_interval_metrics(
    posterior_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    *,
    level: float = 0.90,
) -> dict[str, float]:
    theta = _truth_theta(truth_df)
    mean, variance = _posterior_arrays(posterior_df)
    sd = np.sqrt(variance)
    z_value = float(norm.ppf(0.5 + level / 2.0))
    z = (mean - theta) / sd
    lower = mean - z_value * sd
    upper = mean + z_value * sd
    width = upper - lower
    alpha = 1.0 - level
    lower_miss = np.maximum(lower - theta, 0.0)
    upper_miss = np.maximum(theta - upper, 0.0)
    interval_score = width + (2.0 / alpha) * lower_miss + (2.0 / alpha) * upper_miss
    return {
        f"coverage_{int(level * 100)}": float(np.mean((theta >= mean - z_value * sd) & (theta <= mean + z_value * sd))),
        "posterior_z_mean": float(np.mean(z)),
        "posterior_z_sd": float(np.std(z)),
        f"average_interval_width_{int(level * 100)}": float(np.mean(width)),
        f"median_interval_width_{int(level * 100)}": float(np.median(width)),
        "coverage_width_score": float(np.mean(interval_score)),
        "z_score_sd_abs_error": float(abs(np.std(z) - 1.0)),
    }


def _target_arrays(
    posterior_df: pd.DataFrame,
    truth_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_column = (
        "theta"
        if "theta" in truth_df.columns
        else "target_estimate"
        if "target_estimate" in truth_df.columns
        else "target_mean"
        if "target_mean" in truth_df.columns
        else "truth"
        if "truth" in truth_df.columns
        else None
    )
    if target_column is None:
        raise ValueError("truth dataframe must contain theta, truth, target_estimate, or target_mean.")
    target_se_column = (
        "target_standard_error"
        if "target_standard_error" in truth_df.columns
        else "target_se"
        if "target_se" in truth_df.columns
        else None
    )
    if target_se_column is None:
        target_se = np.zeros(len(truth_df), dtype=np.float64)
    else:
        target_se = np.maximum(truth_df[target_se_column].fillna(0.0).to_numpy(dtype=np.float64), 0.0)
    truth_index = truth_df.loc[:, ["item_id", target_column]].copy()
    truth_index["target_value"] = truth_index[target_column].to_numpy(dtype=np.float64)
    truth_index["target_standard_error_value"] = target_se
    merged = posterior_df.loc[:, ["item_id", "posterior_mean", "posterior_var"]].merge(
        truth_index.loc[:, ["item_id", "target_value", "target_standard_error_value"]],
        on="item_id",
        how="inner",
        validate="one_to_one",
    )
    if merged.shape[0] != posterior_df.shape[0]:
        raise ValueError("posterior and truth item_id values do not align one-to-one.")
    mean = merged["posterior_mean"].to_numpy(dtype=np.float64)
    variance = np.maximum(merged["posterior_var"].to_numpy(dtype=np.float64), 1e-18)
    target = merged["target_value"].to_numpy(dtype=np.float64)
    target_se = merged["target_standard_error_value"].to_numpy(dtype=np.float64)
    return mean, variance, target, target_se


def compute_target_aware_z_scores(
    posterior_df: pd.DataFrame,
    truth_df: pd.DataFrame,
) -> pd.DataFrame:
    mean, variance, target, target_se = _target_arrays(posterior_df, truth_df)
    naive_sd = np.sqrt(variance)
    target_aware_sd = np.sqrt(variance + np.square(target_se))
    return pd.DataFrame(
        {
            "z_naive": (mean - target) / np.maximum(naive_sd, 1e-18),
            "z_target_aware": (mean - target) / np.maximum(target_aware_sd, 1e-18),
            "target_standard_error": target_se,
        }
    )


def compute_target_aware_coverage(
    posterior_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    *,
    level: float = 0.90,
) -> dict[str, float]:
    mean, variance, target, target_se = _target_arrays(posterior_df, truth_df)
    z_value = float(norm.ppf(0.5 + level / 2.0))
    naive_sd = np.sqrt(variance)
    target_aware_sd = np.sqrt(variance + np.square(target_se))
    naive_lower = mean - z_value * naive_sd
    naive_upper = mean + z_value * naive_sd
    target_lower = mean - z_value * target_aware_sd
    target_upper = mean + z_value * target_aware_sd
    return {
        f"coverage_{int(level * 100)}": float(np.mean((target >= naive_lower) & (target <= naive_upper))),
        f"coverage_{int(level * 100)}_target_aware": float(
            np.mean((target >= target_lower) & (target <= target_upper))
        ),
    }


def evaluate_against_exact_truth(
    posterior_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    *,
    level: float = 0.90,
) -> dict[str, float]:
    point = evaluate_point_metrics(posterior_df, truth_df)
    interval = evaluate_interval_metrics(posterior_df, truth_df, level=level)
    return {**point, **interval}


def _interval_score(mean: np.ndarray, sd: np.ndarray, target: np.ndarray, *, level: float) -> np.ndarray:
    z_value = float(norm.ppf(0.5 + level / 2.0))
    lower = mean - z_value * sd
    upper = mean + z_value * sd
    width = upper - lower
    alpha = 1.0 - level
    lower_miss = np.maximum(lower - target, 0.0)
    upper_miss = np.maximum(target - upper, 0.0)
    return width + (2.0 / alpha) * lower_miss + (2.0 / alpha) * upper_miss


def evaluate_against_noisy_target(
    posterior_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    *,
    level: float = 0.90,
) -> dict[str, float]:
    mean, variance, target, target_se = _target_arrays(posterior_df, truth_df)
    error = mean - target
    naive_sd = np.sqrt(variance)
    target_aware_sd = np.sqrt(variance + np.square(target_se))
    z_naive = error / np.maximum(naive_sd, 1e-18)
    z_target = error / np.maximum(target_aware_sd, 1e-18)
    coverage = compute_target_aware_coverage(posterior_df, truth_df, level=level)
    naive_score = _interval_score(mean, naive_sd, target, level=level)
    target_score = _interval_score(mean, target_aware_sd, target, level=level)
    return {
        "rmse_to_target_estimate": float(np.sqrt(np.mean(np.square(error)))),
        "normalized_rmse_target_aware": float(np.sqrt(np.mean(np.square(error) / np.maximum(variance + np.square(target_se), 1e-18)))),
        "z_score_mean": float(np.mean(z_naive)),
        "z_score_sd": float(np.std(z_naive)),
        "z_score_mean_target_aware": float(np.mean(z_target)),
        "z_score_sd_target_aware": float(np.std(z_target)),
        "interval_score_90_naive": float(np.mean(naive_score)),
        "interval_score_90_target_aware": float(np.mean(target_score)),
        "target_se_mean": float(np.mean(target_se)),
        "target_se_median": float(np.median(target_se)),
        "target_aware_metrics_used": bool(np.any(target_se > 0.0)),
        **coverage,
    }


def compute_pairwise_gap_probabilities(
    posterior_df: pd.DataFrame,
    truth_df: pd.DataFrame,
) -> pd.DataFrame:
    target_column = (
        "theta"
        if "theta" in truth_df.columns
        else "target_estimate"
        if "target_estimate" in truth_df.columns
        else "target_mean"
        if "target_mean" in truth_df.columns
        else "truth"
        if "truth" in truth_df.columns
        else None
    )
    if target_column is None:
        raise ValueError("truth dataframe must contain theta, truth, target_estimate, or target_mean.")
    target_se_column = (
        "target_standard_error"
        if "target_standard_error" in truth_df.columns
        else "target_se"
        if "target_se" in truth_df.columns
        else None
    )
    truth = truth_df.loc[:, ["item_id", target_column]].rename(columns={target_column: "target_estimate"}).copy()
    truth["target_standard_error"] = (
        truth_df[target_se_column].to_numpy(dtype=np.float64)
        if target_se_column is not None
        else np.zeros(len(truth_df), dtype=np.float64)
    )
    posterior_columns = ["item_id", "posterior_mean"]
    if "posterior_sd" in posterior_df.columns:
        posterior_columns.append("posterior_sd")
    posterior = posterior_df.loc[:, posterior_columns].copy()
    if "posterior_sd" not in posterior.columns and "posterior_var" in posterior_df.columns:
        posterior["posterior_sd"] = np.sqrt(np.maximum(posterior_df["posterior_var"].to_numpy(dtype=np.float64), 1e-18))
    if "posterior_sd" not in posterior.columns:
        raise ValueError("posterior dataframe must contain posterior_sd or posterior_var.")
    merged = posterior.merge(truth, on="item_id", how="inner", validate="one_to_one")
    if merged.shape[0] != posterior_df.shape[0]:
        raise ValueError("posterior and truth item_id values do not align one-to-one.")
    sd = np.maximum(merged["posterior_sd"].to_numpy(dtype=np.float64), 1e-18)
    mean = merged["posterior_mean"].to_numpy(dtype=np.float64)
    target = merged["target_estimate"].to_numpy(dtype=np.float64)
    target_se = np.maximum(merged["target_standard_error"].fillna(0.0).to_numpy(dtype=np.float64), 0.0)
    p_a = norm.cdf(mean / sd)
    out = merged.loc[:, ["item_id", "posterior_mean", "posterior_sd", "target_estimate", "target_standard_error"]].copy()
    out["posterior_prob_model_a_better"] = p_a
    out["predicted_sign"] = np.where(p_a >= 0.5, 1, -1)
    out["confidence"] = np.maximum(p_a, 1.0 - p_a)
    out["target_sign"] = np.where(target > 0.0, 1, -1)
    out["target_z"] = np.divide(
        np.abs(target),
        target_se,
        out=np.where(np.abs(target) > 0.0, np.inf, 0.0),
        where=target_se > 0.0,
    )
    out["correct_sign"] = out["predicted_sign"].to_numpy(dtype=np.int64) == out["target_sign"].to_numpy(dtype=np.int64)
    return out


def evaluate_decidable_gap_subsets(probability_df: pd.DataFrame) -> dict[str, float]:
    target_z = probability_df["target_z"].to_numpy(dtype=np.float64)
    return {
        "decidable_pair_fraction_90": float(np.mean(target_z >= 1.645)) if len(probability_df) else np.nan,
        "decidable_pair_fraction_95": float(np.mean(target_z >= 1.96)) if len(probability_df) else np.nan,
        "ambiguous_pair_fraction_90": float(np.mean(target_z < 1.645)) if len(probability_df) else np.nan,
        "n_total_pairs": int(len(probability_df)),
        "n_decidable_pairs_90": int(np.sum(target_z >= 1.645)),
        "n_decidable_pairs_95": int(np.sum(target_z >= 1.96)),
        "n_ambiguous_pairs_90": int(np.sum(target_z < 1.645)),
    }


def evaluate_selective_sign_decisions(
    posterior_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = (0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    probabilities = compute_pairwise_gap_probabilities(posterior_df, truth_df)
    subset_masks = {
        "all_pairs": np.ones(len(probabilities), dtype=bool),
        "decidable_pairs_90": probabilities["target_z"].to_numpy(dtype=np.float64) >= 1.645,
        "decidable_pairs_95": probabilities["target_z"].to_numpy(dtype=np.float64) >= 1.96,
        "ambiguous_pairs_90": probabilities["target_z"].to_numpy(dtype=np.float64) < 1.645,
    }
    rows: list[dict[str, object]] = []
    for subset_name, subset_mask in subset_masks.items():
        sub = probabilities.loc[subset_mask].copy()
        n_total = int(len(sub))
        for threshold in thresholds:
            declared = sub["confidence"].to_numpy(dtype=np.float64) >= float(threshold)
            n_declared = int(np.sum(declared))
            if n_declared == 0:
                false_sign = np.nan
                accuracy = np.nan
            else:
                false_sign = float(
                    np.mean(
                        sub.loc[declared, "predicted_sign"].to_numpy(dtype=np.int64)
                        != sub.loc[declared, "target_sign"].to_numpy(dtype=np.int64)
                    )
                )
                accuracy = float(1.0 - false_sign)
            declaration_rate = float(n_declared / n_total) if n_total > 0 else np.nan
            rows.append(
                {
                    "subset": subset_name,
                    "threshold": float(threshold),
                    "declaration_rate": declaration_rate,
                    "abstention_rate": np.nan if n_total == 0 else float(1.0 - declaration_rate),
                    "false_sign_rate_declared": false_sign,
                    "selective_accuracy": accuracy,
                    "n_declared": n_declared,
                    "n_total": n_total,
                }
            )
    curves = pd.DataFrame(rows)
    summary_row: dict[str, object] = {"subset": "all_pairs", **evaluate_decidable_gap_subsets(probabilities)}
    all_pairs = curves[curves["subset"] == "all_pairs"]
    for threshold, suffix in [(0.90, "90"), (0.95, "95")]:
        selected = all_pairs[np.isclose(all_pairs["threshold"].to_numpy(dtype=np.float64), threshold)]
        if selected.empty:
            summary_row[f"false_sign_rate_at_{suffix}conf"] = np.nan
            summary_row[f"declaration_rate_at_{suffix}conf"] = np.nan
            summary_row[f"selective_accuracy_at_{suffix}conf"] = np.nan
        else:
            row = selected.iloc[0]
            summary_row[f"false_sign_rate_at_{suffix}conf"] = row["false_sign_rate_declared"]
            summary_row[f"declaration_rate_at_{suffix}conf"] = row["declaration_rate"]
            summary_row[f"selective_accuracy_at_{suffix}conf"] = row["selective_accuracy"]
    summary = pd.DataFrame([summary_row])
    return curves, summary


def evaluate_pairwise_calibration(
    posterior_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    *,
    bins: int = 10,
    max_pairs: int | None = None,
    random_state: int = 0,
    pair_subset: str = "all_pairs",
    close_gap_quantile: float = 0.30,
    top_fraction: float = 0.10,
    epsilon: float = 1e-8,
) -> tuple[pd.DataFrame, float]:
    truth = truth_df.set_index("item_id")
    theta = truth["theta"] if "theta" in truth.columns else truth["truth"]
    pairs = pairwise_win_probabilities(posterior_df, max_pairs=max_pairs, random_state=random_state)
    if pairs.empty:
        empty = pd.DataFrame(columns=["bin_left", "bin_right", "mean_pred", "empirical_win_rate", "n_pairs"])
        return empty, 0.0
    truth_i = theta.loc[pairs["item_i"].to_numpy()].to_numpy(dtype=np.float64)
    truth_k = theta.loc[pairs["item_k"].to_numpy()].to_numpy(dtype=np.float64)
    pairs = pairs.assign(observed_win=(truth_i > truth_k).astype(float))
    gap = np.abs(truth_i - truth_k)
    if pair_subset == "close_pairs":
        nonzero = gap > 1e-12
        if np.any(nonzero):
            threshold = float(np.quantile(gap[nonzero], close_gap_quantile))
            pairs = pairs.loc[nonzero & (gap <= threshold)].copy()
        else:
            pairs = pairs.copy()
    elif pair_subset == "top_decile_pairs":
        ranks = theta.rank(method="first", ascending=False)
        cutoff = max(1, int(np.ceil(top_fraction * len(theta))))
        top_ids = set(ranks.loc[ranks <= cutoff].index.to_numpy().tolist())
        top_mask = np.array(
            [(item_i in top_ids) or (item_k in top_ids) for item_i, item_k in zip(pairs["item_i"], pairs["item_k"], strict=True)],
            dtype=bool,
        )
        pairs = pairs.loc[top_mask].copy()
    elif pair_subset != "all_pairs":
        raise ValueError("pair_subset must be all_pairs, close_pairs, or top_decile_pairs.")
    if pairs.empty:
        empty = pd.DataFrame(columns=["pair_subset", "bin_left", "bin_right", "mean_pred", "empirical_win_rate", "n_pairs"])
        return empty, 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    total = len(pairs)
    probs = np.clip(pairs["pairwise_win_probability"].to_numpy(dtype=np.float64), epsilon, 1.0 - epsilon)
    observed = pairs["observed_win"].to_numpy(dtype=np.float64)
    for idx in range(bins):
        left = edges[idx]
        right = edges[idx + 1]
        if idx == bins - 1:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        n_pairs = int(np.sum(mask))
        if n_pairs == 0:
            mean_pred = np.nan
            empirical = np.nan
        else:
            mean_pred = float(np.mean(probs[mask]))
            empirical = float(np.mean(observed[mask]))
            ece += (n_pairs / total) * abs(mean_pred - empirical)
        rows.append(
            {
                "bin_left": left,
                "bin_right": right,
                "mean_pred": mean_pred,
                "empirical_win_rate": empirical,
                "n_pairs": n_pairs,
                "pair_subset": pair_subset,
            }
        )
    return pd.DataFrame(rows), float(ece)


def evaluate_pairwise_scores(
    posterior_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    *,
    max_pairs: int | None = None,
    random_state: int = 0,
    close_gap_quantile: float = 0.30,
    epsilon: float = 1e-8,
) -> dict[str, float]:
    truth = truth_df.set_index("item_id")
    theta = truth["theta"] if "theta" in truth.columns else truth["truth"]
    pairs = pairwise_win_probabilities(posterior_df, max_pairs=max_pairs, random_state=random_state)
    if pairs.empty:
        return {
            "pairwise_brier_score": np.nan,
            "pairwise_log_score": np.nan,
            "all_pair_ece": np.nan,
            "close_pair_ece": np.nan,
            "close_pair_brier_score": np.nan,
            "close_pair_log_score": np.nan,
            "close_pair_n_pairs": 0,
            "all_pair_n_pairs": 0,
            "top_decile_pair_ece": np.nan,
        }
    truth_i = theta.loc[pairs["item_i"].to_numpy()].to_numpy(dtype=np.float64)
    truth_k = theta.loc[pairs["item_k"].to_numpy()].to_numpy(dtype=np.float64)
    observed = (truth_i > truth_k).astype(float)
    prob = np.clip(pairs["pairwise_win_probability"].to_numpy(dtype=np.float64), epsilon, 1.0 - epsilon)
    gap = np.abs(truth_i - truth_k)
    nonzero = gap > 1e-12
    if np.any(nonzero):
        close_threshold = float(np.quantile(gap[nonzero], close_gap_quantile))
        close = nonzero & (gap <= close_threshold)
    else:
        close = np.ones_like(gap, dtype=bool)
    if not np.any(close):
        close = np.ones_like(gap, dtype=bool)
    ranks = theta.rank(method="first", ascending=False)
    cutoff = max(1, int(np.ceil(0.10 * len(theta))))
    top_ids = set(ranks.loc[ranks <= cutoff].index.to_numpy().tolist())
    top_mask = np.array(
        [(item_i in top_ids) or (item_k in top_ids) for item_i, item_k in zip(pairs["item_i"], pairs["item_k"], strict=True)],
        dtype=bool,
    )

    def log_score(mask: np.ndarray) -> float:
        return float(-np.mean(observed[mask] * np.log(prob[mask]) + (1.0 - observed[mask]) * np.log(1.0 - prob[mask])))

    return {
        "pairwise_brier_score": float(np.mean(np.square(prob - observed))),
        "pairwise_log_score": log_score(np.ones_like(observed, dtype=bool)),
        "all_pair_ece": float(np.mean(np.abs(prob - observed))),
        "close_pair_ece": float(np.mean(np.abs(prob[close] - observed[close]))),
        "close_pair_brier_score": float(np.mean(np.square(prob[close] - observed[close]))),
        "close_pair_log_score": log_score(close),
        "close_pair_n_pairs": int(np.sum(close)),
        "all_pair_n_pairs": int(prob.size),
        "top_decile_pair_ece": float(np.mean(np.abs(prob[top_mask] - observed[top_mask]))) if np.any(top_mask) else np.nan,
    }


def _dcg(relevance: np.ndarray) -> float:
    discounts = 1.0 / np.log2(np.arange(2, relevance.size + 2))
    return float(np.sum(relevance * discounts))


def ndcg_at_k(truth: np.ndarray, score: np.ndarray, k: int) -> float:
    k = min(int(k), truth.size)
    if k <= 0:
        raise ValueError("k must be positive.")
    relevance = truth - float(np.min(truth))
    order = np.argsort(-score)[:k]
    ideal = np.argsort(-truth)[:k]
    ideal_dcg = _dcg(relevance[ideal])
    if ideal_dcg <= 0.0:
        return 1.0
    return _dcg(relevance[order]) / ideal_dcg


def topk_overlap_at_k(truth: np.ndarray, score: np.ndarray, k: int) -> float:
    k = min(int(k), truth.size)
    truth_top = set(np.argsort(-truth)[:k].tolist())
    score_top = set(np.argsort(-score)[:k].tolist())
    return float(len(truth_top & score_top) / k)


def evaluate_ranking_metrics(
    posterior_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    *,
    k_values: tuple[int, ...] = (1, 5, 10),
    n_samples: int = 5000,
    random_state: int = 0,
) -> dict[str, float]:
    theta = _truth_theta(truth_df)
    mean, _ = _posterior_arrays(posterior_df)
    metrics: dict[str, float] = {
        "posterior_rank_entropy": posterior_rank_entropy(posterior_df, n_samples=n_samples, random_state=random_state)
    }
    for k in k_values:
        clipped = min(int(k), theta.size)
        metrics[f"top_{k}_overlap"] = topk_overlap_at_k(theta, mean, clipped)
        metrics[f"ndcg_at_{k}"] = ndcg_at_k(theta, mean, clipped)
    return metrics


def evaluate_decision_metrics(
    posterior_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    *,
    threshold: float = 0.95,
    n_samples: int = 5000,
    random_state: int = 0,
) -> dict[str, float]:
    return {
        "false_best_rate": false_best_rate(
            posterior_df,
            truth_df,
            threshold=threshold,
            n_samples=n_samples,
            random_state=random_state,
        ),
        "probability_select_true_best": probability_select_true_best(
            posterior_df,
            truth_df,
            n_samples=n_samples,
            random_state=random_state,
        ),
    }


def evaluate_decision_metrics_multi(
    posterior_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = (0.75, 0.90, 0.95),
    k_values: tuple[int, ...] = (1, 3, 5),
    n_samples: int = 5000,
    random_state: int = 0,
) -> dict[str, float]:
    best = best_probabilities(posterior_df, n_samples=n_samples, random_state=random_state)
    truth = truth_df.set_index("item_id")
    theta = truth["theta"] if "theta" in truth.columns else truth["truth"]
    true_best = theta.idxmax()
    metrics: dict[str, float] = {}
    n_items = max(len(best), 1)
    for threshold in thresholds:
        suffix = str(threshold).replace(".", "")
        declared = best.loc[best["best_probability"] > threshold]
        metrics[f"false_best_rate_{suffix}"] = 0.0 if declared.empty else float(np.mean(declared["item_id"].to_numpy() != true_best))
        metrics[f"abstention_rate_{suffix}"] = float(1.0 - len(declared) / n_items)
        metrics[f"selective_accuracy_{suffix}"] = np.nan if declared.empty else float(np.mean(declared["item_id"].to_numpy() == true_best))
    mean = posterior_df.set_index("item_id")["posterior_mean"]
    ranked = mean.sort_values(ascending=False).index.to_list()
    for k in k_values:
        metrics[f"true_best_in_credible_top_{k}"] = float(true_best in set(ranked[: min(int(k), len(ranked))]))
    return metrics


def evaluate_stratified_coverage(
    posterior_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    *,
    view_count: np.ndarray | None = None,
    graph_component: np.ndarray | None = None,
    rank_bins: int = 5,
    quantile_bins: int = 4,
    level: float = 0.90,
) -> pd.DataFrame:
    theta = _truth_theta(truth_df)
    mean, variance = _posterior_arrays(posterior_df)
    sd = np.sqrt(variance)
    z_value = float(norm.ppf(0.5 + level / 2.0))
    covered = (theta >= mean - z_value * sd) & (theta <= mean + z_value * sd)

    def add_quantile_rows(rows: list[dict[str, object]], name: str, values: np.ndarray) -> None:
        if np.all(values == values[0]):
            labels = np.zeros(values.shape[0], dtype=np.int64)
        else:
            labels = pd.qcut(values, q=min(quantile_bins, len(np.unique(values))), labels=False, duplicates="drop")
            labels = np.asarray(labels, dtype=np.int64)
        for label in sorted(np.unique(labels)):
            mask = labels == label
            rows.append(
                {
                    "stratum_type": name,
                    "stratum": str(int(label)),
                    "n_items": int(np.sum(mask)),
                    "coverage": float(np.mean(covered[mask])),
                    "mean_posterior_var": float(np.mean(variance[mask])),
                    "mean_abs_truth": float(np.mean(np.abs(theta[mask]))),
                }
            )

    rows: list[dict[str, object]] = []
    if view_count is not None:
        add_quantile_rows(rows, "view_count_quantile", np.asarray(view_count, dtype=np.float64))
    add_quantile_rows(rows, "posterior_variance_quantile", variance)
    add_quantile_rows(rows, "effect_magnitude_quantile", np.abs(theta))
    true_rank = np.empty(theta.size, dtype=np.int64)
    true_rank[np.argsort(-theta)] = np.arange(theta.size)
    add_quantile_rows(rows, "true_rank_bin", true_rank.astype(np.float64))
    if graph_component is not None:
        graph_component = np.asarray(graph_component)
        for component in sorted(np.unique(graph_component).tolist()):
            mask = graph_component == component
            rows.append(
                {
                    "stratum_type": "graph_component",
                    "stratum": str(component),
                    "n_items": int(np.sum(mask)),
                    "coverage": float(np.mean(covered[mask])),
                    "mean_posterior_var": float(np.mean(variance[mask])),
                    "mean_abs_truth": float(np.mean(np.abs(theta[mask]))),
                }
            )
    return pd.DataFrame(rows)
