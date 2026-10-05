from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(7)
    n_items = 30
    item_ids = np.array([f"item_{idx:03d}" for idx in range(n_items)])
    view_ids = np.array(["reference", "view_b", "view_c"])

    feature = rng.normal(size=n_items)
    theta = 0.5 * feature + rng.normal(scale=0.8, size=n_items)
    view_scale = np.array([1.0, 0.75, 1.2])
    residual_noise = np.array([0.05, 0.10, 0.15])

    rows: list[dict[str, object]] = []
    for item_idx, item_id in enumerate(item_ids):
        for view_idx, view_id in enumerate(view_ids):
            standard_error = float(rng.uniform(0.15, 0.25))
            noise_sd = np.sqrt(standard_error**2 + residual_noise[view_idx] ** 2)
            estimate = rng.normal(view_scale[view_idx] * theta[item_idx], noise_sd)
            rows.append(
                {
                    "item_id": item_id,
                    "view_id": view_id,
                    "estimate": float(estimate),
                    "standard_error": standard_error,
                }
            )

    output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_dir / "observations.parquet", index=False)
    pd.DataFrame({"item_id": item_ids, "feature": feature}).to_parquet(
        output_dir / "item_features.parquet", index=False
    )
    pd.DataFrame(
        {
            "view_id": view_ids,
            "is_reference_view": [True, False, False],
        }
    ).to_parquet(output_dir / "view_metadata.parquet", index=False)
    print(f"Wrote tiny example data to {output_dir}")


if __name__ == "__main__":
    main()
