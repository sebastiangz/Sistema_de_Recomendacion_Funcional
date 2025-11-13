"""
pipeline.py
Define pipelines composables de recomendación funcional.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class RecommendationPipeline:
    model: object

    def fit(self, *args, **kwargs):
        return self.model  # inmutable, no entrenamiento destructivo

    def recommend(self, user_id: int, n_items: int = 10):
        return self.model.recommend(user_id, n_items)


def create_recommendation_pipeline(collaborative, content_based, weights=(0.7, 0.3)):
    """Crea un sistema híbrido funcionalmente."""
    from .hybrid import HybridRecommender
    return RecommendationPipeline(
        HybridRecommender(collaborative, content_based, weights)
    )
