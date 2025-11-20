"""
DEMO: Búsqueda de empleados por nombre dentro del dataset
Permite escribir un nombre (completo o parcial) y devuelve TODOS sus datos.
"""

import pandas as pd

# ===============================
# CONFIGURACIÓN
# ===============================
DATASET_PATH = "HR_Data_MNC_Data Science Lovers.csv"

# ===============================
# CARGA DE DATOS
# ===============================
df = pd.read_csv(DATASET_PATH)

print("\n===== BÚSQUEDA DE EMPLEADOS POR NOMBRE =====\n")

# Mostrar primeras filas para el usuario
print("Vista inicial del dataset:\n")
print(df.head(), "\n")

# ===============================
# BÚSQUEDA POR NOMBRE
# ===============================

# Pedir nombre al usuario
nombre = input("Ingresa el nombre (o parte del nombre) a buscar: ").strip()

# Filtrar coincidencias (case insensitive)
coincidencias = df[df["Full_Name"].str.contains(nombre, case=False, na=False)]

# Mostrar resultados
if coincidencias.empty:
    print("\n❌ No se encontró ningún empleado con ese nombre.")
else:
    print(f"\n✔ Empleados encontrados ({len(coincidencias)} resultados):\n")
    for idx, row in coincidencias.iterrows():
        print("--------------------------------------------------")
        print(f"Employee ID:       {row['Employee_ID']}")
        print(f"Nombre completo:    {row['Full_Name']}")
        print(f"Departamento:       {row['Department']}")
        print(f"Puesto:             {row['Job_Title']}")
        print(f"Fecha contratación: {row['Hire_Date']}")
        print(f"Ubicación:          {row['Location']}")
        print(f"Rating:             {row['Performance_Rating']}")
        print(f"Años experiencia:   {row['Experience_Years']}")
        print(f"Estatus:            {row['Status']}")
        print(f"Modo de trabajo:    {row['Work_Mode']}")
        print(f"Salario (INR):      {row['Salary_INR']}")
        print("--------------------------------------------------")

print("\n===== FIN DE LA BÚSQUEDA =====\n")
