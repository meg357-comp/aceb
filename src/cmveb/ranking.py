from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _posterior_arrays(posterior_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    item_id = posterior_df["item_id"].to_numpy()
    mean = posterior_df["posterior_mean"].to_numpy(dtype=np.float64)
    variance = np.maximum(posterior_df["posterior_var"].to_numpy(dtype=np.float64), 1e-18)
    return item_id, mean, variance


def _sample_pair_indices(n_items: int, max_pairs: int | None, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    if n_items < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    n_pairs = n_items * (n_items - 1) // 2
    if max_pairs is None or n_items <= 2000 or max_pairs >= n_pairs:
        upper, lower = np.triu_indices(n_items, k=1)
        return upper.astype(np.int64), lower.astype(np.int64)
    rng = np.random.default_rng(random_state)
    seen: set[tuple[int, int]] = set()
    first = []
    second = []
    while len(first) < max_pairs:
        i = int(rng.integers(0, n_items))
        k = int(rng.integers(0, n_items - 1))
        if k >= i:
            k += 1
        j, l = sorted((i, k))
        pair = (j, l)
        if pair in seen:
            continue
        seen.add(pair)
        first.append(j)
        second.append(l)
    return np.asarray(first, dtype=np.int64), np.asarray(second, dtype=np.int64)


def pairwise_win_probabilities(
    posterior_df: pd.DataFrame,
    max_pairs: int | None = None,
    random_state: int = 0,
    covariance_lookup: dict[tuple[object, object], float] | None = None,
) -> pd.DataFrame:
    item_id, mean, variance = _posterior_arrays(posterior_df)
    first, second = _sample_pair_indices(len(item_id), max_pairs, random_state)
    denom_var = variance[first] + variance[second]
    if covariance_lookup is not None:
        cov = np.zeros(first.shape[0], dtype=np.float64)
        for idx, (i, k) in enumerate(zip(first, second, strict=True)):
            key = (item_id[i], item_id[k])
            cov[idx] = covariance_lookup.get(key, covariance_lookup.get((key[1], key[0]), 0.0))
        denom_var = np.maximum(denom_var - 2.0 * cov, 1e-18)
    probability = norm.cdf((mean[first] - mean[second]) / np.sqrt(np.maximum(denom_var, 1e-18)))
    return pd.DataFrame(
        {
            "item_i": item_id[first],
            "item_k": item_id[second],
            "idx_i": first,
            "idx_k": second,
            "pairwise_win_probability": probability,
        }
    )


def _posterior_samples(
    posterior_df: pd.DataFrame,
    n_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    item_id, mean, variance = _posterior_arrays(posterior_df)
    rng = np.random.default_rng(random_state)
    samples = rng.normal(mean[None, :], np.sqrt(variance)[None, :], size=(n_samples, len(mean)))
    return item_id, samples


def topk_probabilities_normal_approx(
    posterior_df: pd.DataFrame,
    k_values: list[int] | tuple[int, ...] | np.ndarray,
    n_samples: int = 5000,
    random_state: int = 0,
) -> pd.DataFrame:
    item_id, samples = _posterior_samples(posterior_df, n_samples, random_state)
    rows = []
    order = np.argsort(samples, axis=1)
    for k in k_values:
        k = int(k)
        if k < 1:
            raise ValueError("k_values must be positive.")
        clipped_k = min(k, samples.shape[1])
        counts = np.zeros(samples.shape[1], dtype=np.float64)
        top = order[:, -clipped_k:]
        np.add.at(counts, top.ravel(), 1.0)
        probability = counts / float(n_samples)
        rows.extend({"item_id": item_id[idx], "k": k, "topk_probability": probability[idx]} for idx in range(len(item_id)))
    return pd.DataFrame(rows)


def posterior_rank_entropy(
    posterior_df: pd.DataFrame,
    n_samples: int = 5000,
    random_state: int = 0,
) -> float:
    _, samples = _posterior_samples(posterior_df, n_samples, random_state)
    n_items = samples.shape[1]
    ranks = np.empty_like(np.argsort(samples, axis=1))
    order = np.argsort(-samples, axis=1)
    row_idx = np.arange(samples.shape[0])[:, None]
    ranks[row_idx, order] = np.arange(n_items)[None, :]
    entropy_by_item = []
    for item_idx in range(n_items):
        counts = np.bincount(ranks[:, item_idx], minlength=n_items).astype(np.float64)
        probabilities = counts[counts > 0.0] / float(n_samples)
        entropy_by_item.append(float(-np.sum(probabilities * np.log(probabilities))))
    return float(np.mean(entropy_by_item))


def best_probabilities(
    posterior_df: pd.DataFrame,
    n_samples: int = 5000,
    random_state: int = 0,
) -> pd.DataFrame:
    item_id, samples = _posterior_samples(posterior_df, n_samples, random_state)
    winners = np.argmax(samples, axis=1)
    counts = np.bincount(winners, minlength=len(item_id)).astype(np.float64)
    return pd.DataFrame({"item_id": item_id, "best_probability": counts / float(n_samples)})


def _truth_theta(truth_df: pd.DataFrame) -> pd.Series:
    if "theta" in truth_df.columns:
        return truth_df.set_index("item_id")["theta"]
    if "truth" in truth_df.columns:
        return truth_df.set_index("item_id")["truth"]
    raise ValueError("truth_df must contain a theta or truth column.")


def false_best_rate(
    posterior_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    threshold: float = 0.95,
    n_samples: int = 5000,
    random_state: int = 0,
) -> float:
    best = best_probabilities(posterior_df, n_samples=n_samples, random_state=random_state)
    declared = best.loc[best["best_probability"] > threshold]
    if declared.empty:
        return 0.0
    truth = _truth_theta(truth_df)
    true_best = truth.idxmax()
    return float(np.mean(declared["item_id"].to_numpy() != true_best))


def probability_select_true_best(
    posterior_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    n_samples: int = 5000,
    random_state: int = 0,
) -> float:
    best = best_probabilities(posterior_df, n_samples=n_samples, random_state=random_state)
    truth = _truth_theta(truth_df)
    true_best = truth.idxmax()
    matched = best.loc[best["item_id"] == true_best, "best_probability"]
    return float(matched.iloc[0]) if not matched.empty else 0.0
