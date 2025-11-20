"""
DEMO REAL (versión interactiva):
Recomendar empleados similares a un NOMBRE ingresado por el usuario.

Dataset: HR_Data_MNC_Data Science Lovers.csv
"""

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
df = pd.read_csv(DATASET_PATH)

# Limpieza básica
df[name_col] = df[name_col].astype(str).str.strip()
df[user_col] = df[user_col].astype(str).str.strip()

print("\n===== Sistema de Recomendación por Nombre =====\n")
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
items = df.set_index(user_col)[[name_col]]

# Crear modelo real con tus funciones
content_model = create_content_model(items, [name_col])

# ===============================
# Encontrar empleados cuyo nombre contiene lo buscado
# ===============================
matching = df[df[name_col].str.contains(query_name, case=False, na=False)]

if matching.empty:
    print(f"⚠ No se encontraron nombres similares directamente. Se usará búsqueda semántica.\n")
    # Creamos un "perfil" basado SOLO en el nombre ingresado
    profile_vector_text = query_name
else:
    # Usamos el primer empleado encontrado como referencia
    example = matching.iloc[0]
    profile_vector_text = example[name_col]

print(f"Nombre de referencia: {profile_vector_text}\n")

# ===============================
# TF-IDF
# ===============================
vectorizer = TfidfVectorizer(max_features=300)

# Vectorizamos únicamente la columna "Full_Name"
tfidf_matrix = vectorizer.fit_transform(items[name_col])

target_vector = vectorizer.transform([profile_vector_text]).toarray()[0]

# ===============================
# CALCULAR SIMILITUD
# ===============================
similarities = []
for emp_id, vec in zip(items.index, tfidf_matrix.toarray()):
    score = cosine_similarity(target_vector, vec)
    similarities.append((emp_id, score))

# Ordenar
ranked = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]

print(f"Top empleados con nombre similar a '{query_name}':\n")
for emp_id, score in ranked:
    row = df.loc[df[user_col] == emp_id].iloc[0]
    print(f"- {row['Full_Name']} | Score={score:.3f}")

print("\n===== FIN DEL DEMO REAL =====\n")
