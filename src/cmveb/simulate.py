from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_multiview_data(
    *,
    n_items: int = 100,
    n_views: int = 3,
    n_features: int = 2,
    seed: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    item_ids = np.array([f"item_{idx:04d}" for idx in range(n_items)])
    view_ids = np.array([f"view_{idx:02d}" for idx in range(n_views)])

    feature_matrix = rng.normal(size=(n_items, n_features))
    item_features = pd.DataFrame(feature_matrix, columns=[f"x{idx}" for idx in range(n_features)])
    item_features.insert(0, "item_id", item_ids)

    w_mu = rng.normal(scale=0.3, size=n_features)
    prior_mean = feature_matrix @ w_mu
    prior_variance = 0.5 + 0.2 * rng.random(n_items)
    theta = rng.normal(prior_mean, np.sqrt(prior_variance))
    view_scale = np.linspace(1.0, 0.5, n_views)
    extra_noise = np.linspace(0.02, 0.1, n_views)

    rows = []
    for item_idx, item_id in enumerate(item_ids):
        for view_idx, view_id in enumerate(view_ids):
            standard_error = float(rng.uniform(0.05, 0.25))
            variance = standard_error * standard_error + extra_noise[view_idx] ** 2
            estimate = rng.normal(view_scale[view_idx] * theta[item_idx], np.sqrt(variance))
            rows.append(
                {
                    "item_id": item_id,
                    "view_id": view_id,
                    "estimate": float(estimate),
                    "standard_error": standard_error,
                }
            )
    observations = pd.DataFrame(rows)
    view_metadata = pd.DataFrame({"view_id": view_ids, "is_reference_view": [idx == 0 for idx in range(n_views)]})
    truth = pd.DataFrame({"item_id": item_ids, "theta": theta, "prior_mean": prior_mean, "prior_variance": prior_variance})
    return observations, item_features, view_metadata, truth
