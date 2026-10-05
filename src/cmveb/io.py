from __future__ import annotations

from pathlib import Path

import pandas as pd

from cmveb.schemas import PreparedData
from cmveb.schemas import (
    ITEM_FEATURE_REQUIRED_COLUMNS,
    OBSERVATION_REQUIRED_COLUMNS,
    VIEW_METADATA_REQUIRED_COLUMNS,
)
from cmveb.schemas import validate_item_features, validate_observations, validate_view_metadata


def read_parquet_inputs(
    observations_path: str | Path,
    item_features_path: str | Path,
    view_metadata_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observations = pd.read_parquet(observations_path)
    item_features = pd.read_parquet(item_features_path)
    view_metadata = pd.read_parquet(view_metadata_path)
    return observations, item_features, view_metadata


def load_prepared_data(
    observations_path: str | Path,
    item_features_path: str | Path,
    view_metadata_path: str | Path,
    *,
    standardize_features: bool = True,
) -> PreparedData:
    from cmveb.preprocess import prepare_data

    observations, item_features, view_metadata = read_parquet_inputs(
        observations_path,
        item_features_path,
        view_metadata_path,
    )
    validate_observations(observations)
    validate_item_features(item_features)
    validate_view_metadata(view_metadata)
    return prepare_data(
        observations,
        item_features,
        view_metadata,
        standardize_features=standardize_features,
    )


def required_columns() -> dict[str, list[str]]:
    return {
        "observations.parquet": OBSERVATION_REQUIRED_COLUMNS.copy(),
        "item_features.parquet": ITEM_FEATURE_REQUIRED_COLUMNS.copy(),
        "view_metadata.parquet": VIEW_METADATA_REQUIRED_COLUMNS.copy(),
    }
