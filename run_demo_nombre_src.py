"""
DEMO REAL (versión interactiva CON TEMPORIZADORES):
Recomendar empleados similares a un NOMBRE ingresado por el usuario.

Dataset: HR_Data_MNC_Data Science Lovers.csv
"""

import time
import pandas as pd
from src.content_based import create_content_model
from src.similarity import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ===============================
# CONFIGURACIÓN
# ===============================
DATASET_PATH = "HR_Data_MNC_Data Science Lovers.csv"

user_col = "Employee_ID"
name_col = "Full_Name"

# ===============================
# CARGA DE DATOS
# ===============================
t0 = time.time()
df = pd.read_csv(DATASET_PATH)
t1 = time.time()

df[name_col] = df[name_col].astype(str).str.strip()
df[user_col] = df[user_col].astype(str).str.strip()

print("\n===== Sistema de Recomendación por Nombre =====\n")
print(f"⏱ Tiempo en cargar dataset: {t1 - t0:.4f} segundos\n")

print("Vista previa del dataset:\n")
print(df[[user_col, name_col]].head(), "\n")

# ===============================
# ENTRADA DEL USUARIO
# ===============================
query_name = input("🧍 Ingresa el nombre a buscar (o parte del nombre): ").strip()

print(f"\n===== DEMO: Buscando empleados similares a '{query_name}' =====\n")

# ===============================
# MODELO CONTENT-BASED (solo nombres)
# ===============================
t2 = time.time()
items = df.set_index(user_col)[[name_col]]
content_model = create_content_model(items, [name_col])
t3 = time.time()

print(f"⏱ Tiempo en crear modelo de contenido: {t3 - t2:.4f} segundos\n")

# ===============================
# Encontrar coincidencias directas
# ===============================
t4 = time.time()
matching = df[df[name_col].str.contains(query_name, case=False, na=False)]

if matching.empty:
    print(f"⚠ No se encontraron nombres similares directamente. Se usará búsqueda semántica.\n")
    profile_vector_text = query_name
else:
    example = matching.iloc[0]
    profile_vector_text = example[name_col]
t5 = time.time()

print(f"⏱ Tiempo en búsqueda literal del nombre: {t5 - t4:.4f} segundos")
print(f"Nombre de referencia: {profile_vector_text}\n")

# ===============================
# TF-IDF
# ===============================
t6 = time.time()

vectorizer = TfidfVectorizer(max_features=300)
tfidf_matrix = vectorizer.fit_transform(items[name_col])
target_vector = vectorizer.transform([profile_vector_text]).toarray()[0]

t7 = time.time()
print(f"⏱ Tiempo en vectorizar nombres con TF-IDF: {t7 - t6:.4f} segundos\n")

# ===============================
# CALCULAR SIMILITUD
# ===============================
t8 = time.time()
similarities = []

for emp_id, vec in zip(items.index, tfidf_matrix.toarray()):
    score = cosine_similarity(target_vector, vec)
    similarities.append((emp_id, score))

ranked = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]
t9 = time.time()

print(f"⏱ Tiempo en calcular similitudes: {t9 - t8:.4f} segundos\n")

# ===============================
# MOSTRAR RESULTADOS
# ===============================
print(f"Top empleados con nombre similar a '{query_name}':\n")

for emp_id, score in ranked:
    row = df.loc[df[user_col] == emp_id].iloc[0]
    print(f"- {row['Full_Name']} | Score={score:.3f}")

print("\n===== FIN DEL DEMO REAL =====\n")

print("===== RESUMEN DE TIEMPOS =====")
print(f"📌 Carga de datos:            {t1 - t0:.4f} s")
print(f"📌 Creación del modelo:       {t3 - t2:.4f} s")
print(f"📌 Búsqueda literal:          {t5 - t4:.4f} s")
print(f"📌 Vectorización TF-IDF:      {t7 - t6:.4f} s")
print(f"📌 Cálculo de similitud:      {t9 - t8:.4f} s")
print("================================\n")
