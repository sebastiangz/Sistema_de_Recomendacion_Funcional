"""
run_demo.py - Ejecución de demostración para el sistema de recomendación.
Este script muestra cómo utilizar los recomendadores colaborativo,
basado en contenido y el híbrido mediante los módulos ya implementados.
"""

import pandas as pd
from src.recommender import (
    create_collaborative_filter,
    create_content_based,
    create_hybrid_recommender,
)
from src.pipeline import create_pipeline


def main():
    # -------------------------------------------------------------
    # Datos de ejemplo
    # -------------------------------------------------------------
    ratings_df = pd.DataFrame({
        "user": ["U1", "U1", "U2", "U3"],
        "item": ["A", "B", "A", "C"],
        "rating": [5, 3, 4, 2]
    })

    items_df = pd.DataFrame({
        "title": ["Python Dev", "Data Scientist", "Java Engineer"],
        "desc": [
            "backend python developer",
            "machine learning python data",
            "java spring backend"
        ]
    }, index=["A", "B", "C"])

    # -------------------------------------------------------------
    # Crear recomendadores individuales
    # -------------------------------------------------------------
    # Filtro colaborativo (usuario-usuario o SVD)
    collab_fit = create_collaborative_filter(method="user")
    collab_recommend = collab_fit(ratings_df, "user", "item")

    # Basado en contenido
    content_fit = create_content_based(["title", "desc"])
    content_recommend = content_fit(items_df)

    # Híbrido
    hybrid = create_hybrid_recommender(
        collab_recommend,
        content_recommend,
        weights=(0.7, 0.3)
    )

    # -------------------------------------------------------------
    # Crear pipeline unificado
    # -------------------------------------------------------------
    pipeline = create_pipeline(
        collab_recommend,
        content_recommend,
        weights=(0.6, 0.4)
    )

    # -------------------------------------------------------------
    # Demostración
    # -------------------------------------------------------------
    user_id = "U1"

    print("===== DEMO DEL SISTEMA DE RECOMENDACIÓN =====")
    print(f"Usuario objetivo: {user_id}\n")

    print("--- Recomendaciones colaborativas ---")
    print(collab_recommend(user_id, 3))

    print("\n--- Recomendaciones basadas en contenido (item 'A') ---")
    print(content_recommend("A", 3))

    print("\n--- Recomendaciones híbridas ---")
    print(hybrid(user_id, 3))

    print("\n--- Pipeline unificado ---")
    print(pipeline(user_id, 3))


if __name__ == "__main__":
    main()
