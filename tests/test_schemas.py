from __future__ import annotations

import pandas as pd
import pytest

from cmveb.schemas import validate_item_features, validate_observations, validate_view_metadata


def test_schema_validation_accepts_required_tables() -> None:
    validate_observations(pd.DataFrame({"item_id": ["a"], "view_id": ["v0"], "estimate": [0.1], "standard_error": [0.2]}))
    validate_item_features(pd.DataFrame({"item_id": ["a"], "x": [1.0]}))
    validate_view_metadata(pd.DataFrame({"view_id": ["v0"], "is_reference_view": [True]}))


def test_schema_validation_rejects_missing_columns() -> None:
    with pytest.raises(ValueError):
        validate_observations(pd.DataFrame({"item_id": ["a"], "view_id": ["v0"], "estimate": [0.1]}))


def test_schema_validation_requires_single_reference_view() -> None:
    with pytest.raises(ValueError):
        validate_view_metadata(pd.DataFrame({"view_id": ["v0", "v1"], "is_reference_view": [True, True]}))
