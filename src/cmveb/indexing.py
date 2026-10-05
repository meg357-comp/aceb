from __future__ import annotations

import numpy as np
import pandas as pd

from cmveb.schemas import IndexedData, PreparedData


def _encode(series: pd.Series, categories: np.ndarray) -> np.ndarray:
    lookup = {value: idx for idx, value in enumerate(categories.tolist())}
    encoded = series.map(lookup)
    if encoded.isna().any():
        raise ValueError("Found identifiers that are absent from metadata.")
    return encoded.to_numpy(dtype=np.int64)


def build_indexed_data(prepared: PreparedData) -> IndexedData:
    observations = prepared.observations.copy()
    item_features = prepared.item_features.copy()
    view_metadata = prepared.view_metadata.copy()

    item_ids = item_features["item_id"].to_numpy()
    view_ids = view_metadata["view_id"].drop_duplicates().to_numpy()

    observations = observations.loc[
        observations["item_id"].isin(item_ids) & observations["view_id"].isin(view_ids)
    ].reset_index(drop=True)

    observations["item_idx"] = _encode(observations["item_id"], item_ids)
    observations["view_idx"] = _encode(observations["view_id"], view_ids)
    observations["row_order"] = np.arange(len(observations), dtype=np.int64)
    observations = observations.sort_values(["item_idx", "view_idx"]).reset_index(drop=True)

    row_item_idx = observations["item_idx"].to_numpy(dtype=np.int64)
    counts = np.bincount(row_item_idx, minlength=len(item_ids))
    item_offsets = np.zeros(len(item_ids) + 1, dtype=np.int64)
    item_offsets[1:] = np.cumsum(counts)

    view_metadata = view_metadata.drop_duplicates("view_id").set_index("view_id")
    is_reference_view = view_metadata.loc[view_ids, "is_reference_view"].astype(bool).to_numpy()
    feature_matrix = item_features.loc[:, prepared.feature_columns].to_numpy(dtype=np.float64, copy=True)

    return IndexedData(
        item_ids=item_ids,
        view_ids=view_ids,
        item_features=np.ascontiguousarray(feature_matrix, dtype=np.float64),
        feature_columns=prepared.feature_columns,
        scaler=prepared.scaler,
        item_idx=np.ascontiguousarray(observations["item_idx"].to_numpy(dtype=np.int64)),
        view_idx=np.ascontiguousarray(observations["view_idx"].to_numpy(dtype=np.int64)),
        estimate=np.ascontiguousarray(observations["estimate"].to_numpy(dtype=np.float64)),
        standard_error=np.ascontiguousarray(observations["standard_error"].to_numpy(dtype=np.float64)),
        is_reference_view=np.ascontiguousarray(is_reference_view),
        item_offsets=item_offsets,
        row_order=np.ascontiguousarray(observations["row_order"].to_numpy(dtype=np.int64)),
    )
