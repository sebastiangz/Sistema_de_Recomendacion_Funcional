"""
DEMO INTERACTIVO:
Buscar empleados por nombre dentro del dataset y mostrar TODOS sus datos.

Usa módulos del proyecto en src/ y mantiene el mismo formato
que las otras demos interactivas.

Dataset: HR_Data_MNC_Data Science Lovers.csv
"""

import pandas as pd
from src.similarity import cosine_similarity   
from src.content_based import create_content_model  
from src.recommender import create_content_based   

# ===============================
# CONFIGURACIÓN
# ===============================
DATASET_PATH = "HR_Data_MNC_Data Science Lovers.csv"

print("\n===== SISTEMA DE BÚSQUEDA DE EMPLEADOS POR NOMBRE =====\n")

# ===============================
# CARGA DE DATOS
# ===============================
df = pd.read_csv(DATASET_PATH)

# Limpieza mínima
df["Employee_ID"] = df["Employee_ID"].astype(str).str.strip()
df["Full_Name"] = df["Full_Name"].astype(str).str.strip()

print("Vista inicial del dataset:\n")
print(df.head(), "\n")

# ===============================
# ENTRADAS DEL USUARIO
# ===============================
nombre_buscar = input("🔎 Ingresa el nombre (o parte del nombre) del empleado: ").strip()

# Validación del límite
while True:
    try:
        limite = int(input("📌 ¿Cuántos empleados deseas mostrar? (ej: 1, 5, 10): "))
        if limite > 0:
            break
        else:
            print("El número debe ser mayor que 0.")
    except ValueError:
        print("Ingresa un número entero válido.")

print("\n===== BUSCANDO EMPLEADOS... =====\n")

# ===============================
# BÚSQUEDA DE COINCIDENCIAS
# ===============================

coincidencias = df[df["Full_Name"].str.contains(nombre_buscar, case=False, na=False)]

# Limitar resultados
coincidencias = coincidencias.head(limite)

# ===============================
# RESULTADOS
# ===============================
if coincidencias.empty:
    print("❌ No se encontraron empleados con ese nombre.\n")
else:
    print(f"✔ Se encontraron {len(coincidencias)} empleados.\n")
    print("===== RESULTADOS =====\n")

    for _, row in coincidencias.iterrows():
        print("--------------------------------------------------")
        print(f"Employee ID:        {row['Employee_ID']}")
        print(f"Nombre completo:     {row['Full_Name']}")
        print(f"Departamento:        {row['Department']}")
        print(f"Puesto:              {row['Job_Title']}")
        print(f"Fecha contratación:  {row['Hire_Date']}")
        print(f"Ubicación:           {row['Location']}")
        print(f"Rating:              {row['Performance_Rating']}")
        print(f"Años experiencia:    {row['Experience_Years']}")
        print(f"Estatus:             {row['Status']}")
        print(f"Modalidad trabajo:   {row['Work_Mode']}")
        print(f"Salario (INR):       {row['Salary_INR']}")
        print("--------------------------------------------------")

print("\n===== FIN DEL DEMO =====\n")
