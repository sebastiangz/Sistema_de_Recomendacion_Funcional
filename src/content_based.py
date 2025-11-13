"""
content_based.py
Filtrado basado en contenido funcional.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from .similarity import cosine_similarity


@dataclass(frozen=True)
class ContentBasedModel:
    """Modelo inmutable content-based."""
    features: np.ndarray
    item_ids: tuple

    def find_similar_items(self, item_id: int, k: int = 10):
        """Encuentra los ítems más similares."""
        idx = self.item_ids.index(item_id)
        target = self.features[idx]
        sims = [
            (iid, cosine_similarity(target, self.features[i]))
            for i, iid in enumerate(self.item_ids)
            if iid != item_id
        ]
        return sorted(sims, key=lambda x: x[1], reverse=True)[:k]


def extract_features(items: pd.DataFrame, text_cols: list[str]) -> np.ndarray:
    """Extrae características TF-IDF combinadas de varias columnas."""
    combined = items[text_cols].apply(lambda row: " ".join(row.values.astype(str)), axis=1)
    vectorizer = TfidfVectorizer(max_features=200)
    return vectorizer.fit_transform(combined).toarray()


def create_content_model(items: pd.DataFrame, text_cols: list[str]) -> ContentBasedModel:
    """Crea el modelo content-based funcionalmente."""
    features = extract_features(items, text_cols)
    return ContentBasedModel(features=features, item_ids=tuple(items.index))
