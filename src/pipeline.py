"""
pipeline.py 
No utiliza clases ni estado mutante.
"""

from typing import Callable, Any, List, Tuple
from .hybrid import hybrid_recommend


def create_pipeline(
    collab_fn: Callable[[Any, int], List[Tuple[Any, float]]],
    content_fn: Callable[[Any], List[Tuple[Any, float]]],
    weights: tuple[float, float] = (0.7, 0.3)
):
    """
    Crea una función de recomendación híbrida lista para usar.

    Retorna:
        recommend_fn(user_id, n_items) -> recomendaciones híbridas
    """
    w_collab, w_content = weights

    def recommend_fn(user_id: Any, n_items: int = 10):
        return hybrid_recommend(
            collab_recommender=collab_fn,
            content_sim_fn=content_fn,
            user_id=user_id,
            n=n_items,
            w_collab=w_collab,
            w_content=w_content
        )

    return recommend_fn
