
import pytest
from src.funciones_sahid import (
    get_user_top_items,
    recommend_unrated_items
)


# Matriz de ejemplo para realizar las pruebas
class MatrizPrueba:
    def __init__(self):
        self.user_ids = (1, 2, 3, 4)
        self.item_ids = (101, 102, 103, 104)
        self.data = [
            [5, 4, 0, 0],
            [5, 3, 4, 0],
            [4, 0, 5, 3],
            [0, 4, 0, 5]
        ]

    def get_user_vector(self, user_id):
        indice = self.user_ids.index(user_id)
        return self.data[indice]


# Funcion de prediccion para probar recomendaciones
def prediccion_prueba(matrix, user_id, item_id, k):
    calificaciones = {
        103: 4.35,
        104: 3.94
    }
    return calificaciones[item_id]


# PRUEBA 1: Obtener elementos favoritos
def test_elementos_favoritos():
    matriz = MatrizPrueba()

    resultado = get_user_top_items(matriz, 1)

    assert resultado == [(101, 5.0), (102, 4.0)]


# PRUEBA 2: Recomendar elementos no evaluados
def test_recomendaciones_nuevas():
    matriz = MatrizPrueba()

    resultado = recommend_unrated_items(
        matriz,
        user_id=1,
        n=5,
        predict_fn=prediccion_prueba
    )

    assert resultado == [(103, 4.35), (104, 3.94)]


# PRUEBA 3: Limitar cantidad de recomendaciones
def test_limite_recomendaciones():
    matriz = MatrizPrueba()

    resultado = recommend_unrated_items(
        matriz,
        user_id=1,
        n=1,
        predict_fn=prediccion_prueba
    )

    assert len(resultado) == 1
    assert resultado[0][0] == 103


# PRUEBA 4: Error cuando no hay predictor
def test_sin_predictor():
    matriz = MatrizPrueba()

    with pytest.raises(ValueError):
        recommend_unrated_items(matriz, user_id=1)
