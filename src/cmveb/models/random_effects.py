from __future__ import annotations


from cmveb.models.baseline import random_effects_meta_analysis


class RandomEffectsModel:
    """DerSimonian-Laird random-effects meta-analysis wrapper."""

    def fit(self, data, *args, **kwargs):
        return random_effects_meta_analysis(data, *args, **kwargs)


__all__ = ["RandomEffectsModel", "random_effects_meta_analysis"]
