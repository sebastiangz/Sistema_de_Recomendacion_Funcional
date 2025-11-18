"""
run_demo.py - DEMO del Sistema de Recomendación Funcional
Usa el dataset real HR_Data_MNC_Data Science Lovers.csv
"""

import pandas as pd
from src.recommender import (
    create_collaborative_filter,
    create_content_based,
    create_hybrid_recommender
)

# =========================================================
# Configuración del dataset del proyecto
# =========================================================
DATASET_PATH = "HR_Data_MNC_Data Science Lovers.csv"

user_col = "Employee_ID"
item_col = "Job_Title"
text_cols = ["Location", "Performance_Rating", "Experience_Years", "Status", "Work_Mode"]

# =========================================================
# Cargar datos
# =========================================================
print("\n===== DEMO DEL SISTEMA DE RECOMENDACIÓN =====\n")

df = pd.read_csv(DATASET_PATH)

# Filtrar solo registros donde existan usuario e ítem
df = df[[user_col, item_col] + text_cols].dropna()

# Elegir un usuario para demo
example_user = df[user_col].iloc[0]
print(f"Usuario objetivo: {example_user}\n")

# =========================================================
# 1) Recomendación Colaborativa (SVD)
# =========================================================
collab = create_collaborative_filter(method="svd")["fit"](df, user_col, item_col)
print("--- Recomendaciones colaborativas (SVD) ---")
print(collab["recommend"](example_user, 5))

# =========================================================
# 2) Recomendación Basada en Contenido
# =========================================================
# Creamos un DataFrame de items (puesto → atributos)
items = df.set_index(item_col)[text_cols]

content = create_content_based(text_cols)["fit"](items)
print("\n--- Recomendaciones basadas en contenido ---")
print(content["recommend"](items.index[0], 5))

# =========================================================
# 3) Recomendación Híbrida
# =========================================================
hybrid = create_hybrid_recommender(collab, content)
print("\n--- Recomendaciones híbridas ---")
print(hybrid["recommend"](example_user, 5))

print("\n===== FIN DEMO =====\n")
