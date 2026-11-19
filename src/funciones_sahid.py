
"""
Equipo 9 - Sistema de Recomendacion Funcional
Avance: dos funciones adicionales
"""

# ==========================================
# FUNCION 1: ELEMENTOS FAVORITOS
# ==========================================

def get_user_top_items(matrix, user_id, k=5):
    """
    Obtiene los elementos mejor calificados
    por un usuario.
    """
    ratings = matrix.get_user_vector(user_id)

    items = [
        (item_id, float(rating))
        for item_id, rating in zip(matrix.item_ids, ratings)
        if rating > 0
    ]

    return sorted(
        items,
        key=lambda x: x[1],
        reverse=True
    )[:k]


# ==========================================
# FUNCION 2: RECOMENDAR ELEMENTOS NUEVOS
# ==========================================

def recommend_unrated_items(
    matrix, user_id, n=5, k=10,
    predict_fn=None
):
    """
    Recomienda elementos que un usuario
    todavia no ha calificado.

    predict_fn recibe la funcion de prediccion
    del sistema de filtrado colaborativo.
    """
    ratings = matrix.get_user_vector(user_id)

    unrated = [
        item_id
        for item_id, rating in zip(matrix.item_ids, ratings)
        if rating == 0
    ]

    if predict_fn is None:
        raise ValueError(
            "Debes proporcionar una funcion de prediccion"
        )

    predictions = [
        (
            item_id,
            predict_fn(matrix, user_id, item_id, k)
        )
        for item_id in unrated
    ]

    return sorted(
        predictions,
        key=lambda x: x[1],
        reverse=True
    )[:n]
