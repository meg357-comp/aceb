from __future__ import annotations

import pandas as pd

from cmveb.schemas import PosteriorState


def posterior_frame(item_ids, posterior: PosteriorState) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": item_ids,
            "posterior_mean": posterior.mean,
            "posterior_standard_deviation": posterior.variance ** 0.5,
            "prior_mean": posterior.prior_mean,
            "prior_standard_deviation": posterior.prior_variance ** 0.5,
        }
    )
