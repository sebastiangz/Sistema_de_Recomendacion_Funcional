"""
DEMO INTERACTIVO:
Buscar empleados por nombre dentro del dataset y mostrar TODOS sus datos.

Incluye temporizadores para:
- Carga de datos
- Limpieza
- Búsqueda
- Tiempo total del demo
"""

import time
import pandas as pd

# ===============================
# CONFIGURACIÓN
# ===============================
DATASET_PATH = "HR_Data_MNC_Data Science Lovers.csv"

print("\n===== SISTEMA DE BÚSQUEDA DE EMPLEADOS POR NOMBRE =====\n")

# ===============================
# CRONÓMETRO TOTAL
# ===============================
t_total_start = time.perf_counter()

# ===============================
# CARGA DE DATOS
# ===============================
t_load = time.perf_counter()
df = pd.read_csv(DATASET_PATH)
t_load_end = time.perf_counter()

print(f"⏱ Tiempo carga del dataset: {t_load_end - t_load:.4f} s\n")

# ===============================
# LIMPIEZA DE DATOS
# ===============================
t_clean = time.perf_counter()

df["Employee_ID"] = df["Employee_ID"].astype(str).str.strip()
df["Full_Name"] = df["Full_Name"].astype(str).str.strip()

t_clean_end = time.perf_counter()
print(f"⏱ Tiempo limpieza de datos: {t_clean_end - t_clean:.4f} s\n")

print("Vista inicial del dataset:\n")
print(df.head(), "\n")

# ===============================
# ENTRADAS DEL USUARIO
# ===============================
nombre_buscar = input("🔎 Ingresa el nombre (o parte del nombre) del empleado: ").strip()

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
# BÚSQUEDA
# ===============================
t_search = time.perf_counter()

coincidencias = df[df["Full_Name"].str.contains(nombre_buscar, case=False, na=False)]
coincidencias = coincidencias.head(limite)

t_search_end = time.perf_counter()
print(f"⏱ Tiempo búsqueda: {t_search_end - t_search:.4f} s\n")

# ===============================
# RESULTADOS
# ===============================
if coincidencias.empty:
    print("❌ No se encontraron empleados con ese nombre.\n")
else:
    print(f"✔ Se encontraron {len(coincidencias)} empleados.\n")
    print("===== RESULTADOS =====\n")

    t_print = time.perf_counter()

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

    t_print_end = time.perf_counter()
    print(f"\n⏱ Tiempo impresión resultados: {t_print_end - t_print:.4f} s\n")

# ===============================
# TIEMPO TOTAL
# ===============================
t_total_end = time.perf_counter()
print(f"⏱ Tiempo total del demo: {t_total_end - t_total_start:.4f} s")

print("\n===== FIN DEL DEMO =====\n")
