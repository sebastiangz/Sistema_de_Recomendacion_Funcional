"""
recommender.py - Sistema de recomendación completamente funcional (sin POO).
"""

from typing import Callable, Dict, List, Tuple
import pandas as pd

from .collaborative import (
    build_user_item_matrix,
    user_based_recommend,
    svd_recommend
)
from .content_based import create_content_model
from .hybrid import hybrid_recommend


# ============================================================
# FÁBRICAS FUNCIONALES DE RECOMENDADORES
# ============================================================

def create_collaborative_filter(method: str = "user"):
    """
    Retorna una función fit(df, user_col, item_col) que produce
    una función recommend(user_id, n_items).
    """
    def fit(df: pd.DataFrame, user_col: str, item_col: str):

        mat, _, _ = build_user_item_matrix(df, user_col, item_col)

        def recommend(user_id, n_items=5):
            if method == "svd":
                return svd_recommend(mat, user_id, n_recs=n_items)
            return user_based_recommend(mat, user_id, n_recs=n_items)

        return recommend  # función pura, sin estados mutables

    return fit


def create_content_based(feature_columns: List[str]):
    """
    Retorna una función fit(items_df) -> recommend(item_id, k)
    """
    def fit(items_df: pd.DataFrame):
        model = create_content_model(items_df, feature_columns)

        def recommend(item_id, k=5):
            return model.find_similar_items(item_id, k)

        return recommend

    return fit


def create_hybrid_recommender(
    collab_recommend_fn: Callable,
    content_recommend_fn: Callable,
    weights: Tuple[float, float] = (0.7, 0.3)
):
    """
    Retorna una función recommend(user_id, n_items) combinando
    recomendaciones colaborativas y basadas en contenido.
    """

    def recommend(user_id, n_items=10):

        def content_sim(item_id):
            sims = content_recommend_fn(item_id, k=3)
            if not sims:
                return 0.0
            return float(sum(score for _, score in sims) / len(sims))

        return hybrid_recommend(
            collab_recommender=collab_recommend_fn,
            content_sim_fn=content_sim,
            user_id=user_id,
            n=n_items,
            w_collab=weights[0],
            w_content=weights[1],
        )

    return recommend
