from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from cmveb.config import CMVEBConfig
from cmveb.indexing import build_indexed_data
from cmveb.io import load_prepared_data
from cmveb.models.anchored_eb import fit_staged_calibrated_eb
from cmveb.plotting import posterior_frame

app = typer.Typer(help="Calibrated multiview empirical Bayes.")


@app.command()
def fit(config: Path) -> None:
    cfg = CMVEBConfig.from_yaml(config)
    prepared = load_prepared_data(
        cfg.data.observations,
        cfg.data.item_features,
        cfg.data.view_metadata,
        standardize_features=cfg.preprocess.feature_standardize,
    )
    data = build_indexed_data(prepared)
    result = fit_staged_calibrated_eb(data, cfg.model, max_iter=cfg.fit.max_iter)
    cfg.data.output_dir.mkdir(parents=True, exist_ok=True)
    posterior_frame(data.item_ids, result.posterior).to_parquet(cfg.data.output_dir / "posterior.parquet", index=False)
    pd.DataFrame({"view_id": data.view_ids, "view_scale": result.view_scale, "extra_noise": result.extra_noise}).to_csv(
        cfg.data.output_dir / "view_calibration.csv",
        index=False,
    )
