from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


OBSERVATION_REQUIRED_COLUMNS = ["item_id", "view_id", "estimate", "standard_error"]
ITEM_FEATURE_REQUIRED_COLUMNS = ["item_id"]
VIEW_METADATA_REQUIRED_COLUMNS = ["view_id", "is_reference_view"]


@dataclass(slots=True)
class PreparedData:
    observations: pd.DataFrame
    item_features: pd.DataFrame
    view_metadata: pd.DataFrame
    feature_columns: list[str]
    scaler: dict[str, dict[str, float]]


@dataclass(slots=True)
class IndexedData:
    item_ids: np.ndarray
    view_ids: np.ndarray
    item_features: np.ndarray
    feature_columns: list[str]
    scaler: dict[str, dict[str, float]]
    item_idx: np.ndarray
    view_idx: np.ndarray
    estimate: np.ndarray
    standard_error: np.ndarray
    is_reference_view: np.ndarray
    item_offsets: np.ndarray
    row_order: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.estimate.shape[0])

    @property
    def n_items(self) -> int:
        return int(self.item_ids.shape[0])

    @property
    def n_views(self) -> int:
        return int(self.view_ids.shape[0])


@dataclass(slots=True)
class SufficientStatistics:
    precision_addend: np.ndarray
    natural_addend: np.ndarray
    second_moment_weight: np.ndarray


@dataclass(slots=True)
class PosteriorState:
    mean: np.ndarray
    variance: np.ndarray
    second_moment: np.ndarray
    prior_mean: np.ndarray
    prior_variance: np.ndarray


@dataclass(slots=True)
class FitResult:
    posterior: PosteriorState
    view_scale: np.ndarray
    extra_noise: np.ndarray
    w_mu: np.ndarray
    w_v: np.ndarray
    history: list[dict[str, float]]
    diagnostics: dict[str, object] | None = None
    view_bias: np.ndarray | None = None


@dataclass(slots=True)
class ParquetInput:
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)


def _require_columns(frame: pd.DataFrame, required: list[str], table_name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def validate_observations(observations: pd.DataFrame) -> None:
    _require_columns(observations, OBSERVATION_REQUIRED_COLUMNS, "observations")
    if observations[["item_id", "view_id"]].isna().any().any():
        raise ValueError("observations contains missing item_id or view_id values.")
    numeric = observations[["estimate", "standard_error"]]
    if not all(pd.api.types.is_numeric_dtype(numeric[column]) for column in numeric.columns):
        raise TypeError("estimate and standard_error must be numeric.")
    if (observations["standard_error"] <= 0).any():
        raise ValueError("standard_error must be strictly positive.")
    if observations.duplicated(["item_id", "view_id"]).any():
        raise ValueError("observations contains duplicate item_id/view_id rows.")


def validate_item_features(item_features: pd.DataFrame) -> None:
    _require_columns(item_features, ITEM_FEATURE_REQUIRED_COLUMNS, "item_features")
    if item_features["item_id"].isna().any():
        raise ValueError("item_features contains missing item_id values.")
    if item_features["item_id"].duplicated().any():
        raise ValueError("item_features contains duplicate item_id values.")
    feature_columns = [column for column in item_features.columns if column != "item_id"]
    bad = [column for column in feature_columns if not pd.api.types.is_numeric_dtype(item_features[column])]
    if bad:
        raise TypeError(f"item feature columns must be numeric: {bad}")


def validate_view_metadata(view_metadata: pd.DataFrame) -> None:
    _require_columns(view_metadata, VIEW_METADATA_REQUIRED_COLUMNS, "view_metadata")
    if view_metadata["view_id"].isna().any():
        raise ValueError("view_metadata contains missing view_id values.")
    if view_metadata["view_id"].duplicated().any():
        raise ValueError("view_metadata contains duplicate view_id values.")
    if not pd.api.types.is_bool_dtype(view_metadata["is_reference_view"]):
        raise TypeError("is_reference_view must be boolean.")
    if int(view_metadata["is_reference_view"].sum()) != 1:
        raise ValueError("view_metadata must mark exactly one reference view.")
