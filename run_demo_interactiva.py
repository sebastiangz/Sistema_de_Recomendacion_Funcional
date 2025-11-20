"""
DEMO REAL (versión interactiva):
Recomendar empleados similares a un TRABAJO elegido por el usuario que vivan en un PAÍS elegido.

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
item_col = "Job_Title"

# Columnas para construir las características
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

# ===============================
# CARGA DE DATOS
# ===============================
df = pd.read_csv(DATASET_PATH)

# Limpieza básica
df = df.dropna(subset=[item_col, "Location"])
df[user_col] = df[user_col].astype(str).str.strip()

# ===============================
# ENTRADA DEL USUARIO
# ===============================

print("\n===== Sistema de Recomendación de Empleados =====\n")

target_country = input("🏳️ Ingrese el país donde buscar empleados: ").strip()
target_job = input("💼 Ingrese el trabajo a buscar (ej: Software Engineer): ").strip()

print(f"\n===== DEMO: Recomendar empleados similares a '{target_job}' en {target_country} =====\n")

# ===============================
# FILTRAR POR PAÍS
# ===============================

df_country = df[df["Location"].str.contains(target_country, case=False, na=False)]

if df_country.empty:
    raise ValueError(f"❌ No hay empleados registrados en {target_country}.")

# ===============================
# MODELO CONTENT-BASED
# ===============================
items = df_country.set_index(user_col)[text_cols]

content_model = create_content_model(items, text_cols)

# Buscar empleados del trabajo objetivo
matching_employees = df_country[df_country["Job_Title"].str.contains(target_job, case=False, na=False)]

# ===============================
# PERFIL OBJETIVO
# ===============================
if matching_employees.empty:
    print(f"⚠ No existen '{target_job}' en {target_country}. Buscando perfil global...\n")
    fallback = df[df["Job_Title"].str.contains(target_job, case=False, na=False)]

    if fallback.empty:
        raise ValueError(f"❌ No existe el trabajo '{target_job}' en ningún país del dataset.")

    example = fallback.iloc[0]
else:
    example = matching_employees.iloc[0]

profile_vector = " ".join(str(example[col]) for col in text_cols)

# ===============================
# TF-IDF
# ===============================
vectorizer = TfidfVectorizer(max_features=300)

tfidf_matrix = vectorizer.fit_transform(items.apply(lambda r: " ".join(r.values.astype(str)), axis=1))
target_vector = vectorizer.transform([profile_vector]).toarray()[0]

# ===============================
# CALCULAR SIMILITUD
# ===============================
similarities = []
for emp_id, vec in zip(items.index, tfidf_matrix.toarray()):
    score = cosine_similarity(target_vector, vec)
    similarities.append((emp_id, score))

# Ordenar
ranked = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]

print(f"Top candidatos similares a {target_job} en {target_country}:\n")
for emp_id, score in ranked:
    row = df.loc[df[user_col] == emp_id].iloc[0]
    print(f"- {row['Full_Name']} | {row['Job_Title']} | Score={score:.3f}")

print("\n===== FIN DEL DEMO REAL =====\n")
