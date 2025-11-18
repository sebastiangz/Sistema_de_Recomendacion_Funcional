"""
run_demo.py - DEMO del Sistema de Recomendación Funcional
Adaptado a la implementación REAL de collaborative.py (sin objetos, sin 'fit').
"""

import pandas as pd

from src.collaborative import (
    build_user_item_matrix,
    user_based_recommend,
    item_based_recommend
)

from src.matrix_factorization import svd_recommend
from src.content_based import create_content_model


# =========================================================
# Configuración del dataset del proyecto
# =========================================================
DATASET_PATH = "HR_Data_MNC_Data Science Lovers.csv"

user_col = "Employee_ID"
item_col = "Job_Title"
text_cols = ["Location", "Performance_Rating", "Experience_Years", "Status", "Work_Mode"]


print("\n===== DEMO DEL SISTEMA DE RECOMENDACIÓN =====\n")

# =========================================================
# 1) Cargar datos
# =========================================================
df = pd.read_csv(DATASET_PATH)

df = df[[user_col, item_col] + text_cols].dropna()

example_user = df[user_col].iloc[0]
print(f"Usuario objetivo: {example_user}\n")

# =========================================================
# 2) Construir matriz colaborativa
# =========================================================
mat, user_map, item_map = build_user_item_matrix(df, user_col, item_col)

# =========================================================
# 3) Recomendación colaborativa basada en usuarios
# =========================================================
print("--- Recomendaciones colaborativas (User-Based) ---")
print(user_based_recommend(mat, example_user, n_recs=5))

# =========================================================
# 4) Recomendación por SVD
# =========================================================
print("\n--- Recomendaciones colaborativas (SVD) ---")
print(svd_recommend(mat, example_user, n_recs=5))

# =========================================================
# 5) Recomendación basada en contenido
# =========================================================
items_df = df.set_index(item_col)[text_cols]
content_model = create_content_model(items_df, text_cols)

print("\n--- Recomendaciones basadas en contenido ---")
print(content_model.find_similar_items(items_df.index[0], k=5))

# =========================================================
# 6) Recomendación híbrida manual (sencilla)
# =========================================================
# Fusión simple: promedio entre SVD y contenido
print("\n--- Recomendaciones híbridas (promedio simple) ---")

svd_recs = dict(svd_recommend(mat, example_user, n_recs=10))
content_recs = dict(content_model.find_similar_items(items_df.index[0], k=10))

hybrid_scores = {}

for item in set(list(svd_recs.keys()) + list(content_recs.keys())):
    hybrid_scores[item] = (
        svd_recs.get(item, 0) * 0.7 +
        content_recs.get(item, 0) * 0.3
    )

hybrid_sorted = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
print(hybrid_sorted[:5])

print("\n===== FIN DEMO =====\n")
