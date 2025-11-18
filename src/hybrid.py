"""
hybrid.py 
Composición de recomendaciones colaborativas + contenido.
"""

from typing import List, Tuple, Callable, Dict
import numpy as np


def hybrid_scores(
    collab_scores: Dict[object, float],
    content_score_fn: Callable[[object], float],
    w_collab: float = 0.7,
    w_content: float = 0.3
) -> Dict[object, float]:
    """
    Combina puntajes colaborativos y de contenido de manera funcional.
    No modifica los argumentos originales.

    Parámetros:
    -----------
    collab_scores : dict[item -> score]
    content_score_fn : función que recibe item_id y regresa score float
    w_collab, w_content : pesos del modelo

    Retorna:
    --------
    dict : {item: score_hibrido}
    """
    final_scores = {}
    for item_id, cscore in collab_scores.items():
        final_scores[item_id] = (
            w_collab * float(cscore)
            + w_content * float(content_score_fn(item_id))
        )
    return final_scores


def hybrid_recommend(
    collab_recommender: Callable[[object, int], List[Tuple[object, float]]],
    content_sim_fn: Callable[[object], List[Tuple[object, float]]],
    user_id: object,
    n: int = 10,
    w_collab: float = 0.7,
    w_content: float = 0.3
) -> List[Tuple[object, float]]:
    """
    Genera una recomendación híbrida puramente funcional.

    Parámetros:
    -----------
    collab_recommender : función user -> recomendaciones colaborativas
    content_sim_fn : función item_id -> lista de (item, score) similares
    user_id : usuario objetivo
    n : número de recomendaciones
    w_collab, w_content : pesos

    Retorna:
    --------
    Lista ordenada de (item, score)
    """

    collab = dict(collab_recommender(user_id, n * 2))

    def content_score(item_id):
        sims = content_sim_fn(item_id)
        if not sims:
            return 0.0
        return float(np.mean([s for _, s in sims]))

    final_scores = hybrid_scores(collab, content_score, w_collab, w_content)
    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:n]
