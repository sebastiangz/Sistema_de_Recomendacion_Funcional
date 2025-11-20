import numpy as np
from typing import List, Dict, Set, Any

def precision_at_k(recs: List[Any], relevant: Set[Any], k: int) -> float:
    return len(set(recs[:k]) & relevant) / k if k > 0 else 0.0

def recall_at_k(recs: List[Any], relevant: Set[Any], k: int) -> float:
    return len(set(recs[:k]) & relevant) / len(relevant) if relevant else 0.0

def ndcg_at_k(recs: List[Any], scores: Dict[Any, float], k: int) -> float:
    def dcg(vals):
        return sum((2 ** v - 1) / np.log2(i + 2) for i, v in enumerate(vals))
    actual = [scores.get(i, 0.0) for i in recs[:k]]
    ideal = sorted(scores.values(), reverse=True)[:k]
    return dcg(actual) / dcg(ideal) if sum(ideal) > 0 else 0.0


def _get_recommend_fn(model):
    """Devuelve una función recommend(user_id, k)."""
    # dict-style: {"recommend": fn}
    if isinstance(model, dict) and "recommend" in model and callable(model["recommend"]):
        return model["recommend"]
    # function itself: fn(user_id, k)
    if callable(model):
        return model
    # object with method recommend
    if hasattr(model, "recommend") and callable(getattr(model, "recommend")):
        return getattr(model, "recommend")
    raise TypeError("Modelo inválido: espere un dict{'recommend':fn}, una función o un objeto con método .recommend()")


def _normalize_recs(recs_raw):
    """
    Normaliza la salida de recommend(...) a una lista de items.
    Acepta:
      - [(item, score), ...] -> [item, ...]
      - [item, item, ...] -> [item, ...]
      - None -> []
    """
    if recs_raw is None:
        return []
    # if generator -> list
    if not isinstance(recs_raw, (list, tuple)):
        try:
            recs_raw = list(recs_raw)
        except Exception:
            return []
    if not recs_raw:
        return []
    first = recs_raw[0]
    # lista de tuplas
    if isinstance(first, (list, tuple)) and len(first) >= 1:
        return [r[0] for r in recs_raw]
    # lista plana de ids
    return list(recs_raw)


def evaluate_model(model, test_df, k: int = 10) -> Dict[str, float]:
    """
    Evaluación robusta compatible con:
      - model = {"recommend": fn}
      - model = fn(user_id, k)
      - model = object with method .recommend(user_id, k)
    test_df debe tener columnas: user_id, item_id, rating
    """
    recommend_fn = _get_recommend_fn(model)

    metrics = {'precision': [], 'recall': [], 'ndcg': []}

    # Validaciones mínimas sobre test_df
    if 'user_id' not in test_df.columns or 'item_id' not in test_df.columns or 'rating' not in test_df.columns:
        raise ValueError("test_df debe contener columnas 'user_id', 'item_id' y 'rating'")

    # Convertir rating a float cuando sea posible (no modificar original)
    ratings_col = test_df['rating'].astype(float, copy=False)

    # Agrupar por usuario y evaluar
    users = test_df['user_id'].unique()
    for uid in users:
        try:
            recs_raw = recommend_fn(uid, k)
        except Exception:
            # si falla recomendar para un usuario -> tratamos como sin recomendaciones
            recs_raw = []
        recs = _normalize_recs(recs_raw)

        # ítems relevantes y scores reales para el usuario
        user_mask = test_df['user_id'] == uid
        relevant_set = set(test_df.loc[user_mask, 'item_id'].values.tolist())
        # construir mapping item->rating (última aparición si hay duplicados)
        user_pairs = test_df.loc[user_mask, ['item_id', 'rating']].values.tolist()
        scores = {}
        for item, r in user_pairs:
            try:
                scores[item] = float(r)
            except Exception:
                scores[item] = 0.0

        # calcular métricas para este usuario
        metrics['precision'].append(precision_at_k(recs, relevant_set, k))
        metrics['recall'].append(recall_at_k(recs, relevant_set, k))
        metrics['ndcg'].append(ndcg_at_k(recs, scores, k))

    # evitar mean de lista vacía
    result = {}
    for m, vals in metrics.items():
        result[m] = float(np.mean(vals)) if vals else 0.0

    return result
