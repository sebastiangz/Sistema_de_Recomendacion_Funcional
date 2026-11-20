import random
import time
from src.cargar_datos import load_ratings, build_matrix, predecir, MatrizUI
from src.funciones_sahid import recommend_unrated_items

N_USUARIOS = 10
N_RECS = 5
K_VECINOS = 10
UMBRAL = 4.0
random.seed(42)

ratings = load_ratings("data/processed/ratings.csv")
completa = build_matrix(ratings)
tabla_train = completa.tabla.copy()

# 1. ocultar 20% de las calificaciones de 10 usuarios
oculto = {}
for uid in completa.user_ids:
    if len(oculto) == N_USUARIOS:
        break
    fila = completa.tabla.loc[uid]
    calificadas = list(fila[fila > 0].index)
    if len(calificadas) < 20:
        continue
    ocultas = random.sample(calificadas, max(1, len(calificadas) // 5))
    oculto[uid] = {i: fila[i] for i in ocultas}
    for i in ocultas:
        tabla_train.loc[uid, i] = 0.0

matriz = MatrizUI(tabla_train)

# 2. referencia: peliculas mas populares
popularidad = (tabla_train > 0).sum(axis=0).sort_values(ascending=False)

# 3. evaluar
print("\nUsuario\tHits\tPrecision@5\tRecall\tHits_popular")
t0 = time.perf_counter()
suma_p = 0
suma_r = 0
suma_base = 0
usados = 0

for uid, ocultas in oculto.items():
    relevantes = [i for i, nota in ocultas.items() if nota >= UMBRAL]
    if not relevantes:
        continue

    recs = recommend_unrated_items(
        matriz, uid, n=N_RECS, k=K_VECINOS, predict_fn=predecir
    )
    ids_recs = [item for item, _ in recs]
    hits = len([i for i in ids_recs if i in relevantes])

    populares = []
    for item in popularidad.index:
        if tabla_train.at[uid, item] == 0:
            populares.append(item)
        if len(populares) == N_RECS:
            break
    hits_base = len([i for i in populares if i in relevantes])

    precision = hits / N_RECS
    recall = hits / len(relevantes)
    suma_p += precision
    suma_r += recall
    suma_base += hits_base / N_RECS
    usados += 1
    print(f"{uid}\t{hits}\t{precision:.2f}\t\t{recall:.2f}\t{hits_base}")

print("\n===== RESULTADOS =====")
print(f"Usuarios evaluados:\t{usados}")
print(f"Precision@{N_RECS} filtrado colaborativo:\t{suma_p / usados:.3f}")
print(f"Recall promedio:\t{suma_r / usados:.3f}")
print(f"Precision@{N_RECS} solo populares:\t{suma_base / usados:.3f}")
print(f"Tiempo:\t{time.perf_counter() - t0:.1f} s")
