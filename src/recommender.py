"""
recommender.py

Interfaz principal del Sistema de Recomendación Funcional.
Combina filtrado colaborativo, basado en contenido y métodos híbridos.
"""

from typing import List, Tuple
import pandas as pd
from .collaborative import (
    build_user_item_matrix,
    user_based_recommend,
    item_based_recommend,
    svd_recommend
)
from .content_based import create_content_model
from .hybrid import HybridRecommender


class CollaborativeFilter:
    """
    Recomendador basado en filtrado colaborativo.
    Soporta: user-based, item-based, y SVD.
    """
    def __init__(self, method: str = "user"):
        self.method = method
        self.mat = None

    def fit(self, df: pd.DataFrame, user_col: str, item_col: str, rating_col: str = None):
        self.mat, _, _ = build_user_item_matrix(df, user_col, item_col, rating_col)
        return self

    def recommend(self, user_id, n_items: int = 5) -> List[Tuple[object, float]]:
        if self.method == "user":
            return user_based_recommend(self.mat, user_id, n_recs=n_items)
        elif self.method == "item":
            return item_based_recommend(self.mat, user_id, n_recs=n_items)
        elif self.method == "svd":
            return svd_recommend(self.mat, user_id, n_recs=n_items)
        else:
            raise ValueError(f"Método desconocido: {self.method}")


class ContentBased:
    """
    Recomendador basado en contenido (TF-IDF + similitud de coseno).
    """
    def __init__(self, text_columns: List[str]):
        self.text_columns = text_columns
        self.model = None

    def fit(self, items_df: pd.DataFrame):
        self.model = create_content_model(items_df, self.text_columns)
        return self

    def find_similar_items(self, item_id, k: int = 10):
        if self.model is None:
            raise RuntimeError("El modelo no ha sido entrenado con fit().")
        return self.model.find_similar_items(item_id, k=k)


class HybridSystem:
    """
    Recomendador híbrido que combina filtrado colaborativo y contenido.
    """
    def __init__(self, collaborative: CollaborativeFilter, content: ContentBased, weights=(0.7, 0.3)):
        self.hybrid = HybridRecommender(collaborative, content, weights)

    def recommend(self, user_id, n_items: int = 10):
        return self.hybrid.recommend(user_id, n=n_items)
