"""
run_ab_test.py
A/B TESTING entre dos recomendadores:
A = Content-Based
B = Híbrido (Content + SVD)
"""

import pandas as pd
from src.recommender import (
    create_collaborative_filter,
    create_content_based,
    create_hybrid_recommender
)
from src.evaluation import evaluate_model

DATASET_PATH = "HR_Data_MNC_Data Science Lovers.csv"

df = pd.read_csv(DATASET_PATH).dropna()

user_col = "Employee_ID"
item_col = "Job_Title"
text_cols = ["Department","Job_Title","Location","Experience_Years","Status","Work_Mode"]

# =============================
# Entrenamiento de Modelos A y B
# =============================

content_model = create_content_based(text_cols)["fit"](df.set_index(item_col)[text_cols])
collab_model  = create_collaborative_filter("svd")["fit"](df, user_col, item_col)
hybrid_model  = create_hybrid_recommender(collab_model, content_model)

# Crear estructura de test
test_df = pd.DataFrame({
    "user_id": df[user_col],
    "item_id": df[item_col],
    "rating": df["Performance_Rating"]
})

print("\n===== A/B TESTING =====\n")

print("Modelo A: Content-Based")
results_A = evaluate_model(content_model, test_df, k=5)
print(results_A)

print("\nModelo B: Híbrido")
results_B = evaluate_model(hybrid_model, test_df, k=5)
print(results_B)

print("\n===== RESULTADO FINAL =====\n")
for metric in results_A:
    winner = "A" if results_A[metric] > results_B[metric] else "B"
    print(f"{metric.upper()}: ganador → Modelo {winner}")
