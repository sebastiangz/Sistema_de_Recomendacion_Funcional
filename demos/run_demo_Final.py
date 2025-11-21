"""
DEMO INTERACTIVO:
Recomendación de empleados usando filtrado basado en contenido (Content-Based)

El usuario puede seleccionar:
- País
- Trabajo (Job Title)
- Mínimo Performance Rating
- Mínimos Años de Experiencia
- Cantidad de resultados (Top-K)
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.similarity import cosine_similarity

# ===============================
# CONFIGURACIÓN
# ===============================
DATASET_PATH = "HR_Data_MNC_Data Science Lovers.csv"

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

user_col = "Employee_ID"
item_col = "Job_Title"


# ===============================
# CARGAR DATOS
# ===============================
df = pd.read_csv(DATASET_PATH)
df[user_col] = df[user_col].astype(str).str.strip()

# Limpieza básica
df = df.dropna(subset=[item_col, "Location"])

print("\n===== SISTEMA DE RECOMENDACIÓN DE EMPLEADOS =====\n")


# ===============================
# ENTRADAS DEL USUARIO
# ===============================

target_country = input("🏳️ País donde buscar empleados: ").strip()
target_job = input("💼 Puesto buscado (ej: Software Engineer): ").strip()

min_rating = int(input("⭐ Mínimo Performance Rating (1–5): ").strip())
min_experience = int(input("📚 Mínimos años de experiencia: ").strip())

top_k = int(input("🔝 Cantidad de recomendaciones (Top-K): ").strip())


print(f"\n===== FILTRANDO DATOS PARA {target_job} EN {target_country} =====\n")


# ===============================
# FILTRO POR PAÍS
# ===============================
df_filtered = df[df["Location"].str.contains(target_country, case=False, na=False)]

if df_filtered.empty:
    raise ValueError(f"❌ No existen empleados en '{target_country}'.")


# ===============================
# FILTRO POR PERFORMANCE Y EXPERIENCIA
# ===============================
df_filtered = df_filtered[
    (df_filtered["Performance_Rating"] >= min_rating) &
    (df_filtered["Experience_Years"] >= min_experience)
]

if df_filtered.empty:
    raise ValueError("❌ No hay empleados que cumplan con rating y experiencia mínimos.")


# ===============================
# PERFIL OBJETIVO PARA TF-IDF
# ===============================

matching = df_filtered[df_filtered["Job_Title"].str.contains(target_job, case=False, na=False)]

if matching.empty:
    print("⚠ No se encontraron empleados con ese Job Title en este país/criterios.")
    print("⚠ Usando un ejemplo global como referencia.\n")

    global_match = df[df["Job_Title"].str.contains(target_job, case=False, na=False)]
    if global_match.empty:
        raise ValueError(f"❌ No existe el Job Title '{target_job}' en el dataset.")
    
    example = global_match.iloc[0]
else:
    example = matching.iloc[0]

profile_vector_text = " ".join(str(example[col]) for col in text_cols)


# ===============================
# VECTORIZACIÓN TF-IDF
# ===============================
df_filtered["features"] = df_filtered[text_cols].astype(str).agg(" ".join, axis=1)

vectorizer = TfidfVectorizer(max_features=300)

tfidf_matrix = vectorizer.fit_transform(df_filtered["features"])
tfidf_array = tfidf_matrix.toarray()

target_vector = vectorizer.transform([profile_vector_text]).toarray()[0]


# ===============================
# CALCULAR SIMILITUD
# ===============================
similarities = []
for emp_id, vec in zip(df_filtered[user_col], tfidf_array):
    score = cosine_similarity(target_vector, vec)
    similarities.append((emp_id, score))

ranked = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]


# ===============================
# MOSTRAR RESULTADOS
# ===============================
print(f"\n===== TOP {top_k} EMPLEADOS SIMILARES =====\n")

for emp_id, score in ranked:
    row = df[df[user_col] == emp_id].iloc[0]
    print(f"- {row['Full_Name']} | {row['Job_Title']} | Rating: {row['Performance_Rating']} | "
          f"Años Exp: {row['Experience_Years']} | Similarity={score:.3f}")

print("\n===== FIN DEL DEMO =====\n")
