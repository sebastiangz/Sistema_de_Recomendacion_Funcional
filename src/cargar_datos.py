import numpy as np
import pandas as pd
from src.collaborative import build_user_item_matrix, user_top_k_neighbors


class MatrizUI:
    """Envuelve la tabla usuario x item para las funciones de Sahid."""

    def __init__(self, tabla):
        self.tabla = tabla
        self.user_ids = tuple(int(u) for u in tabla.index)
        self.item_ids = tuple(int(i) for i in tabla.columns)
        self.cache = {}

    def get_user_vector(self, user_id):
        return self.tabla.loc[user_id].values


def load_ratings(ruta="data/processed/ratings.csv", min_votos=50):
    df = pd.read_csv(ruta)
    df = df.rename(columns={"userId": "user_id",
                            "movieId": "item_id"})

    # peliculas con suficientes votos, para que corra rapido
    votos = df.groupby("item_id")["rating"].count()
    df = df[df["item_id"].isin(votos[votos >= min_votos].index)]
    return df[["user_id", "item_id", "rating"]]


def load_items(ruta="data/processed/movies.csv"):
    df = pd.read_csv(ruta)
    return df.rename(columns={"movieId": "id",
                              "genres": "genre"})


def build_matrix(df):
    # usa la funcion del equipo (collaborative.py)
    tabla, _, _ = build_user_item_matrix(
        df, "user_id", "item_id", "rating"
    )
    return MatrizUI(tabla)


def predecir(matriz, user_id, item_id, k=10):
    """Prediccion con vecinos (usa user_top_k_neighbors del equipo)."""
    clave = (user_id, k)
    if clave not in matriz.cache:
        mi = matriz.get_user_vector(user_id)
        vecinos = user_top_k_neighbors(matriz.tabla, user_id, k=k)
        sims = []
        for v in vecinos:
            vv = matriz.get_user_vector(v)
            s = np.dot(mi, vv) / (np.linalg.norm(mi) * np.linalg.norm(vv) + 1e-9)
            sims.append(float(s))
        matriz.cache[clave] = (vecinos, sims)

    vecinos, sims = matriz.cache[clave]
    num = 0.0
    den = 0.0
    for v, s in zip(vecinos, sims):
        nota = matriz.tabla.at[v, item_id]
        if nota > 0:
            num += s * nota
            den += abs(s)
    return num / den if den > 0 else 0.0
