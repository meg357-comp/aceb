from __future__ import annotations


from cmveb.models.baseline import adaptive_shrinkage_like


class AdaptiveShrinkageModel:
    """Small mixture-normal empirical-Bayes shrinkage wrapper."""

    def fit(self, data, *args, **kwargs):
        return adaptive_shrinkage_like(data, *args, **kwargs)


__all__ = ["AdaptiveShrinkageModel", "adaptive_shrinkage_like"]
