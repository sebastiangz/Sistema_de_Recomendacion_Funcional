"""
AB_testing.py
Prueba A/B entre:
A = Content-Based
B = Híbrido (Colaborativo + Contenido)
"""

import pandas as pd
from src.recommender import (
    create_collaborative_filter,
    create_content_based,
    create_hybrid_recommender
)
from src.evaluation import evaluate_model

# =============================
# CARGA DEL DATASET
# =============================
DATASET_PATH = "HR_Data_MNC_Data Science Lovers.csv"
df = pd.read_csv(DATASET_PATH).dropna()

user_col = "Employee_ID"
item_col = "Job_Title"

text_cols = [
    "Department",
    "Job_Title",
    "Location",
    "Experience_Years",
    "Status",
    "Work_Mode"
]

# =============================
# MODELO A: CONTENT-BASED
# =============================

# columnas válidas para contenido (sin el item_col)
cb_text_cols = [c for c in text_cols if c != item_col]

content_factory = create_content_based(cb_text_cols)

# DataFrame indexado por Job_Title
items_df = df.set_index(item_col)[cb_text_cols]

content_model = {
    "recommend": content_factory(items_df)
}

print("\nModelo A listo (Content-Based)\n")

# =============================
# MODELO COLABORATIVO
# =============================
collab_factory = create_collaborative_filter("svd")
collab_model = collab_factory(df, user_col, item_col)

# =============================
# MODELO B: HÍBRIDO
# =============================
hybrid_model = create_hybrid_recommender(
    collab_model,
    content_model["recommend"],   # solo la función recommend
    weights=(0.7, 0.3)
)

hybrid_model = {"recommend": hybrid_model}

print("Modelo B listo (Híbrido)\n")

# =============================
# TEST SET
# =============================
test_df = pd.DataFrame({
    "user_id": df[user_col],
    "item_id": df[item_col],
    "rating": df["Performance_Rating"]
})

# =============================
# A/B TESTING
# =============================

print("\n============================")
print("       🔥 A/B TESTING")
print("============================\n")

print("Evaluando Modelo A (Content-Based)...")
results_A = evaluate_model(content_model, test_df, k=5)
print("Resultados A:", results_A)

print("\nEvaluando Modelo B (Híbrido)...")
results_B = evaluate_model(hybrid_model, test_df, k=5)
print("Resultados B:", results_B)

# =============================
# RESULTADO FINAL
# =============================
print("\n============================")
print(" 📊 RESULTADO FINAL (A vs B)")
print("============================\n")

for metric in results_A:
    winner = "A" if results_A[metric] > results_B[metric] else "B"
    print(f"{metric.upper():<10} → ganador: Modelo {winner}")
