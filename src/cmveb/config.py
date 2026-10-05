from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(slots=True)
class DataConfig:
    observations: Path
    item_features: Path
    view_metadata: Path
    output_dir: Path = Path("outputs")


@dataclass(slots=True)
class PreprocessConfig:
    duplicate_policy: Literal["reject"] = "reject"
    feature_standardize: bool = True
    min_standard_error: float = 1e-8
    eps: float = 1e-8


@dataclass(slots=True)
class ModelConfig:
    eps: float = 1e-8
    reference_view_scale: float = 1.0
    initial_non_reference_view_scale: float = 0.5
    initial_extra_noise: float = 1e-4
    extra_noise_lower: float = 0.0
    extra_noise_upper: float = 10.0
    extra_noise_shrinkage: float = 1.0
    l2_prior: float = 1e-4
    l2_view_scale: float = 1e-4
    residual_calibration: Literal["leave_view", "in_sample", "unit_crossfit"] = "leave_view"
    n_crossfit_folds: int = 5
    random_state: int = 0
    crossfit_max_iter: int = 50
    fit_view_bias: bool = False
    center_reference: bool = False


@dataclass(slots=True)
class FitConfig:
    seed: int = 1
    mode: Literal["staged_calibrated"] = "staged_calibrated"
    max_iter: int = 200


@dataclass(slots=True)
class CMVEBConfig:
    data: DataConfig
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fit: FitConfig = field(default_factory=FitConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CMVEBConfig":
        import yaml

        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        data_payload = payload["data"]
        data = DataConfig(
            observations=Path(data_payload["observations"]),
            item_features=Path(data_payload["item_features"]),
            view_metadata=Path(data_payload["view_metadata"]),
            output_dir=Path(data_payload.get("output_dir", "outputs")),
        )
        preprocess = PreprocessConfig(**payload.get("preprocess", {}))
        model = ModelConfig(**payload.get("model", {}))
        fit = FitConfig(**payload.get("fit", {}))
        return cls(data=data, preprocess=preprocess, model=model, fit=fit)

    def to_dict(self) -> dict:
        def convert(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, list):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
