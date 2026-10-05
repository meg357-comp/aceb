from __future__ import annotations

import numpy as np
import pandas as pd

from cmveb.schemas import IndexedData, PreparedData
from cmveb.schemas import validate_item_features, validate_observations, validate_view_metadata


def prepare_data(
    observations: pd.DataFrame,
    item_features: pd.DataFrame,
    view_metadata: pd.DataFrame,
    *,
    standardize_features: bool = True,
    min_standard_error: float = 1e-8,
) -> PreparedData:
    validate_observations(observations)
    validate_item_features(item_features)
    validate_view_metadata(view_metadata)

    obs = observations.copy()
    features = item_features.copy()
    views = view_metadata.copy()
    obs["standard_error"] = obs["standard_error"].clip(lower=min_standard_error)

    feature_columns = [column for column in features.columns if column != "item_id"]
    if not feature_columns:
        features["intercept"] = 1.0
        feature_columns = ["intercept"]
        scaler = {"intercept": {"center": 0.0, "scale": 1.0}}
    else:
        if "intercept" not in feature_columns:
            features.insert(1, "intercept", 1.0)
            feature_columns = ["intercept"] + feature_columns
        scaler = {}
        if standardize_features:
            for column in feature_columns:
                if column == "intercept":
                    scaler[column] = {"center": 0.0, "scale": 1.0}
                    continue
                center = float(features[column].mean())
                scale = float(features[column].std(ddof=0))
                if not np.isfinite(scale) or scale <= 0.0:
                    scale = 1.0
                features[column] = (features[column] - center) / scale
                scaler[column] = {"center": center, "scale": scale}
        else:
            scaler = {column: {"center": 0.0, "scale": 1.0} for column in feature_columns}

    obs = obs.loc[
        obs["item_id"].isin(features["item_id"]) & obs["view_id"].isin(views["view_id"])
    ].reset_index(drop=True)
    return PreparedData(obs, features, views, feature_columns, scaler)


def center_estimates_by_view(
    data: IndexedData,
    reference_view: int | str | None = None,
) -> tuple[IndexedData, pd.DataFrame]:
    if reference_view is None:
        reference_idx = None
    elif isinstance(reference_view, str):
        matches = np.flatnonzero(data.view_ids == reference_view)
        if matches.size == 0:
            raise ValueError(f"Unknown reference view: {reference_view}")
        reference_idx = int(matches[0])
    else:
        reference_idx = int(reference_view)
        if not (0 <= reference_idx < data.n_views):
            raise ValueError("reference_view must be a valid view index.")

    view_mean = np.zeros(data.n_views, dtype=np.float64)
    for view in range(data.n_views):
        rows = data.view_idx == view
        view_mean[view] = float(np.mean(data.estimate[rows])) if np.any(rows) else 0.0
    offsets = view_mean.copy()
    if reference_idx is not None:
        offsets = offsets - view_mean[reference_idx]
    centered_estimate = data.estimate - offsets[data.view_idx]
    centered = IndexedData(
        item_ids=data.item_ids,
        view_ids=data.view_ids,
        item_features=data.item_features,
        feature_columns=data.feature_columns,
        scaler=data.scaler,
        item_idx=data.item_idx,
        view_idx=data.view_idx,
        estimate=np.ascontiguousarray(centered_estimate, dtype=np.float64),
        standard_error=data.standard_error,
        is_reference_view=data.is_reference_view,
        item_offsets=data.item_offsets,
        row_order=data.row_order,
    )
    offset_table = pd.DataFrame(
        {
            "view_id": data.view_ids,
            "view_idx": np.arange(data.n_views, dtype=np.int64),
            "view_bias": offsets,
        }
    )
    return centered, offset_table
