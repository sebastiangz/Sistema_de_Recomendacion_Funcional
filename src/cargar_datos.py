import pandas as pd

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
    tabla = df.pivot_table(index="user_id", columns="item_id",
                           values="rating", fill_value=0)
    user_ids = tuple(int(u) for u in tabla.index)
    item_ids = tuple(int(i) for i in tabla.columns)
    return tabla.values.astype(float), user_ids, item_ids
