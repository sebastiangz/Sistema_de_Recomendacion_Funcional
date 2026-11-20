import pytest
from src.funciones_sahid import (
    get_user_top_items,
    recommend_unrated_items
)
from src.cargar_datos import load_ratings, build_matrix, predecir


@pytest.fixture(scope="module")
def matriz():
    ratings = load_ratings("data/processed/ratings.csv")
    return build_matrix(ratings)


def test_elementos_favoritos(matriz):
    usuario = matriz.user_ids[0]
    k = 5

    resultado = get_user_top_items(matriz, usuario, k)

    assert 0 < len(resultado) <= k

    notas = [nota for _, nota in resultado]
    assert notas == sorted(notas, reverse=True)

    vector = matriz.get_user_vector(usuario)
    for item_id, nota in resultado:
        idx = matriz.item_ids.index(item_id)
        assert nota > 0
        assert nota == float(vector[idx])


def test_recomendaciones_nuevas(matriz):
    usuario = matriz.user_ids[0]

    resultado = recommend_unrated_items(
        matriz, usuario, n=5, k=10, predict_fn=predecir
    )

    assert len(resultado) == 5

    preds = [p for _, p in resultado]
    assert preds == sorted(preds, reverse=True)

    vector = matriz.get_user_vector(usuario)
    for item_id, pred in resultado:
        idx = matriz.item_ids.index(item_id)
        assert vector[idx] == 0
        assert 0 <= pred <= 5


def test_limite_recomendaciones(matriz):
    usuario = matriz.user_ids[0]

    resultado = recommend_unrated_items(
        matriz, usuario, n=1, k=10, predict_fn=predecir
    )

    assert len(resultado) == 1


def test_sin_predictor(matriz):
    usuario = matriz.user_ids[0]

    with pytest.raises(ValueError):
        recommend_unrated_items(matriz, usuario)
