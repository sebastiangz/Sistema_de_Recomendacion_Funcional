"""
hybrid.py
Sistema híbrido composable (colaborativo + contenido).
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass(frozen=True)
class HybridRecommender:
    collaborative: object
    content_based: object
    weights: Tuple[float, float] = (0.7, 0.3)

    def recommend(self, user_idx: int, n: int = 10) -> List[Tuple[int, float]]:
        """Genera recomendaciones híbridas."""
        collab_recs = dict(self.collaborative.recommend(user_idx, n * 2))
        w_c, w_cb = self.weights

        final_scores = {}
        for item_id, c_score in collab_recs.items():
            # Diversificación basada en contenido
            content_bonus = np.mean([
                s for _, s in self.content_based.find_similar_items(item_id, k=3)
            ])
            final_scores[item_id] = w_c * c_score + w_cb * content_bonus

        return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:n]

