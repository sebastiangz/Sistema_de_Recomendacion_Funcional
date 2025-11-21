"""
DEMO: Recomendación de empleados similares usando análisis basado en contenido
Versión extendida con:
- df.head()
- ejemplo de la columna de características
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.similarity import cosine_similarity

# ============================================================
# 1. Cargar dataset
# ============================================================

df = pd.read_csv("HR_Data_MNC_Data Science Lovers.csv")

# Limpieza mínima de ID
df["Employee_ID"] = df["Employee_ID"].astype(str).str.strip()

print("\n===== DATASET CARGADO =====\n")
print(df.head(), "\n")   # ★ MOSTRAR HEAD DEL DATASET

# ============================================================
# 2. Construcción de características de texto
# ============================================================

text_cols = [
    "Full_Name",
    "Department",
    "Job_Title",
    "Location",
    "Performance_Rating",
    "Experience_Years",
    "Status",
    "Work_Mode",
]

# Crear columna "features"
df["features"] = df[text_cols].astype(str).agg(" ".join, axis=1)

print("===== EJEMPLO DE LA COLUMNA DE CARACTERÍSTICAS =====\n")
print(df[["Employee_ID", "features"]].head(), "\n")   # ★ MOSTRAR FEATURES

# ============================================================
# 3. Vectorizar todo el dataset
# ============================================================

vectorizer = TfidfVectorizer(max_features=400)
tfidf_matrix = vectorizer.fit_transform(df["features"])
tfidf_array = tfidf_matrix.toarray()

print("Vectorización completada. Dimensión TF-IDF:", tfidf_array.shape, "\n")

# ============================================================
# 4. Elegir empleado objetivo
# ============================================================

target_id = df.iloc[0]["Employee_ID"]  # Puedes cambiarlo
print(f"===== EMPLEADO OBJETIVO =====\nID: {target_id}\n")

target_features = df[df["Employee_ID"] == target_id]["features"].iloc[0]
target_vector = vectorizer.transform([target_features]).toarray()[0]

# ============================================================
# 5. Calcular similitud con todos los empleados
# ============================================================

similarities = []
for emp_id, vec in zip(df["Employee_ID"], tfidf_array):
    score = cosine_similarity(target_vector, vec)
    similarities.append((emp_id, score))

# Ordenar descendente
similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

# Remover a sí mismo
similarities = [pair for pair in similarities if pair[0] != target_id]

# Top K
top_k = similarities[:5]

# ============================================================
# 6. Mostrar resultados
# ============================================================

print("===== EMPLEADOS MÁS SIMILARES =====\n")
for emp_id, score in top_k:
    row = df[df["Employee_ID"] == emp_id].iloc[0]
    print(f"- {row['Full_Name']} | {row['Job_Title']} | Score={score:.3f}")

print("\n===== FIN DEL DEMO =====\n")
