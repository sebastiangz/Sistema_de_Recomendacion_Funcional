"""
recommender.py (versión funcional)
Sistema de recomendación completamente funcional (sin POO).

Expone fábricas de recomendadores:
- create_collaborative_filter()
- create_content_based()
- create_hybrid_recommender()
"""

from typing import Callable, Dict, Tuple, List
import pandas as pd

from .collaborative import (
    build_user_item_matrix,
    user_based_recommend,
    svd_recommend
)

from .content_based import create_content_model
from .hybrid import HybridRecommender


# ============================================================
# FABRICAS FUNCIONALES DE MODELOS DE RECOMENDACIÓN
# ============================================================

def create_collaborative_filter(method: str = "user") -> Dict[str, Callable]:
    """
    Fábrica de recomendador colaborativo funcional.

    Retorna un diccionario inmutable con una función "fit" interna.
    """
    def fit(df: pd.DataFrame, user_col: str, item_col: str):
        mat, _, _ = build_user_item_matrix(df, user_col, item_col)

        def recommend(user_id, n_items=5):
            if method == "svd":
                return svd_recommend(mat, user_id, n_recs=n_items)
            return user_based_recommend(mat, user_id, n_recs=n_items)

        return {"recommend": recommend}

    return {"fit": fit}


def create_content_based(feature_columns: List[str]) -> Dict[str, Callable]:
    """
    Fábrica de filtrado basado en contenido (funcional).
    """
    def fit(items_df: pd.DataFrame):
        model = create_content_model(items_df, feature_columns)

        def recommend(item_id, k=5):
            return model.find_similar_items(item_id, k)

        return {"recommend": recommend, "model": model}

    return {"fit": fit}


def create_hybrid_recommender(
    collaborative_model: Dict[str, Callable],
    content_model: Dict[str, Callable],
    weights: Tuple[float, float] = (0.7, 0.3)
) -> Dict[str, Callable]:
    """
    Fábrica de recomendador híbrido funcional.
    El resultado combina resultados colaborativos y de contenido.
    """
    hybrid = HybridRecommender(
        collaborative=collaborative_model,
        content_based=content_model.get("model"),
        weights=weights
    )

    def recommend(user_id, n_items=10):
        return hybrid.recommend(user_id, n_items)

    return {"recommend": recommend}
