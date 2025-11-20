"""
DEMO: Búsqueda de empleados por nombre dentro del dataset
Ahora el usuario puede elegir cuántos empleados mostrar.
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

# Mostrar primeras filas del dataset
print("Vista inicial del dataset:\n")
print(df.head(), "\n")

# ===============================
# BUSCAR POR NOMBRE
# ===============================

# Pedir nombre a buscar
nombre = input("Ingresa el nombre (o parte del nombre) a buscar: ").strip()

# Pedir cantidad máxima de resultados
while True:
    try:
        limite = int(input("¿Cuántos empleados deseas mostrar? (ej: 1, 5, 10): "))
        if limite > 0:
            break
        else:
            print("Por favor ingresa un número mayor que 0.")
    except ValueError:
        print("Valor inválido. Ingresa un número entero.")

# Buscar coincidencias (case insensitive)
coincidencias = df[df["Full_Name"].str.contains(nombre, case=False, na=False)]

# Limitar resultados
coincidencias = coincidencias.head(limite)

# ===============================
# MOSTRAR RESULTADOS
# ===============================

if coincidencias.empty:
    print("\n❌ No se encontró ningún empleado con ese nombre.")
else:
    print(f"\n✔ Empleados encontrados (mostrando máximo {limite}):\n")
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