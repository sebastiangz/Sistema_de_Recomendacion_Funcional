"""
run_demo.py
DEMO REAL: Recomendar empleados similares a Software Engineers que vivan en Mexico
Dataset: HR_Data_MNC_Data Science Lovers.csv
"""

import pandas as pd
from src.content_based import create_content_model
from src.similarity import cosine_similarity
import numpy as np

# ===============================
# CONFIGURACIÓN
# ===============================
DATASET_PATH = "HR_Data_MNC_Data Science Lovers.csv"

# Columnas del dataset
user_col = "Employee_ID"
item_col = "Job_Title"

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

# Foco en empleados de México
df_mexico = df[df["Location"].str.contains("Mexico", case=False, na=False)]

if df_mexico.empty:
    raise ValueError("❌ No hay empleados registrados en Mexico.")

print("\n===== DEMO: Recomendar empleados similares a 'Software Engineer' en Mexico =====\n")

# ===============================
# FILTRAR SOLO EMPLEADOS EN MÉXICO
# ===============================
target_job = "Software Engineer"

if target_job not in df_mexico["Job_Title"].unique():
    print(f"⚠ No existen '{target_job}' registrados en Mexico. Buscando candidatos similares...\n")

# ===============================
# MODELO CONTENT-BASED
# ===============================
items = df_mexico.set_index(user_col)[text_cols]

content_model = create_content_model(items, text_cols)

# Buscar un empleado que sea Software Engineer
software_engineers = df_mexico[df_mexico["Job_Title"] == target_job]

if software_engineers.empty:
    # Si no hay SE en México, tomar un SE global como referencia
    fallback = df[df["Job_Title"] == target_job].iloc[0]
    profile_vector = " ".join(str(fallback[col]) for col in text_cols)
    print("⚠ Usando perfil global de Software Engineer como referencia.\n")
else:
    example_employee = software_engineers.iloc[0]
    profile_vector = " ".join(str(example_employee[col]) for col in text_cols)

# Vectorizar perfil objetivo
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=300)
tfidf_matrix = vectorizer.fit_transform(items.apply(lambda r: " ".join(r.values.astype(str)), axis=1))

target_vector = vectorizer.transform([profile_vector]).toarray()[0]

# ===============================
# CALCULAR SIMILITUD CON TODOS
# ===============================
similarities = []
for emp_id, vec in zip(items.index, tfidf_matrix.toarray()):
    score = cosine_similarity(target_vector, vec)
    similarities.append((emp_id, score))

# Ordenar
ranked = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]

print("Top candidatos similares a Software Engineer en Mexico:\n")
for emp_id, score in ranked:
    row = df.loc[df[user_col] == emp_id].iloc[0]
    print(f"- {row['Full_Name']} | {row['Job_Title']} | Score={score:.3f}")

print("\n===== FIN DEL DEMO REAL =====\n")
