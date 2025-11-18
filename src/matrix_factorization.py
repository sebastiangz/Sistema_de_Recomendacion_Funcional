# matrix_factorization.py

"""
matrix_factorization.py

Implementación funcional y pura de recomendación basada en factorización (SVD).
Diseñado para integrarse con collaborative_filtering.py del Proyecto 9.

Exporta:
- svd_recommend(mat: pd.DataFrame, user_id, n_components: int = 10, n_recs: int = 5)

Principios:
- Funciones puras e inmutables (no modifican los inputs).
- Manejo seguro de casos degenerados (filas o columnas nulas).
- Documentación en español.
"""

from typing import List, Tuple, Dict
import numpy as np
import pandas as pd


def _safe_svd(A: np.ndarray, full_matrices: bool = False):
    """
    Envoltura segura sobre np.linalg.svd.
    Devuelve U, s, Vt garantizando tipos float64.
    """
    # numpy ya lanza LinAlgError en caso de problema; no lo atrapamos para no ocultar errores.
    U, s, Vt = np.linalg.svd(A, full_matrices=full_matrices)
    return U.astype(float), s.astype(float), Vt.astype(float)


def _predict_from_svd(U: np.ndarray, s: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:
    """
    Reconstruye la aproximación de rango-k a la matriz A usando las k primeras componentes.
    A_hat = U[:, :k] * s[:k] @ Vt[:k, :]
    """
    if k <= 0:
        raise ValueError("k debe ser >= 1")
    k = min(k, s.shape[0])
    Uk = U[:, :k]
    sk = s[:k]
    Vtk = Vt[:k, :]
    # multiplicación segura: (Uk * sk) @ Vtk  -> Uk * sk (broadcast columna)
    return (Uk * sk) @ Vtk


def svd_predict_matrix(mat: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
    """
    Devuelve una matriz de predicción (DataFrame) reconstruida mediante SVD de rango n_components.
    - No modifica 'mat' (inmutabilidad): trabaja con copia de los valores.
    - Los valores originales se usan donde existían ratings; la predicción sirve para estimar ceros.
    """
    if not isinstance(mat, pd.DataFrame):
        raise TypeError("mat debe ser un pd.DataFrame")

    # Trabajar con copia inmutable de los valores
    A = mat.values.astype(float).copy()
    # SVD estable y segura
    U, s, Vt = _safe_svd(A, full_matrices=False)
    # Reconstrucción de rango reducido
    k = min(n_components, s.shape[0])
    A_hat = _predict_from_svd(U, s, Vt, k)
    pred_df = pd.DataFrame(A_hat, index=mat.index, columns=mat.columns)
    return pred_df


def svd_recommend(mat: pd.DataFrame, user_id, n_components: int = 10, n_recs: int = 5) -> List[Tuple[object, float]]:
    """
    Genera recomendaciones SVD para un usuario dado.
    Parámetros:
    - mat: pd.DataFrame (usuarios x ítems) con 0 indicando item no valorado.
    - user_id: índice (label) de la fila en 'mat' que corresponde al usuario.
    - n_components: número de componentes para la reconstrucción SVD.
    - n_recs: número de recomendaciones a devolver.
    Retorna:
    - Lista de tuplas (item_label, puntuación_estimada) ordenada descendentemente.
    """
    if not isinstance(mat, pd.DataFrame):
        raise TypeError("mat debe ser un pd.DataFrame")

    if user_id not in mat.index:
        # Si el usuario no existe, devolvemos lista vacía (coherente con otros métodos)
        return []

    # No modificamos 'mat' (inmutabilidad)
    user_row = mat.loc[user_id].astype(float).copy()
    # Predicción completa por SVD
    pred_df = svd_predict_matrix(mat, n_components=n_components)

    # Candidate items: aquellos que el usuario NO ha calificado (supuesto = 0)
    user_orig = user_row
    candidate_mask = user_orig == 0

    # Extraer scores para candidates
    scores: Dict[object, float] = {}
    for col in mat.columns[candidate_mask]:
        # Tomar la predicción estimada
        val = float(pred_df.at[user_id, col])
        scores[col] = val

    # Ordenar por score descendente
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # En caso de empate, sorted mantiene orden estable; devolvemos hasta n_recs
    return ranked[:n_recs]


# Opcional: función para evaluar RMSE entre matriz original y predicha (útil para tuning)
def svd_rmse(mat: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    """
    Calcula RMSE solo sobre valores observados (no sobre ceros implícitos).
    - mat: DataFrame original (usuarios x ítems)
    - pred_df: DataFrame con predicciones (misma index/columns)
    """
    if not isinstance(mat, pd.DataFrame) or not isinstance(pred_df, pd.DataFrame):
        raise TypeError("mat y pred_df deben ser pd.DataFrame")
    if not mat.index.equals(pred_df.index) or not mat.columns.equals(pred_df.columns):
        raise ValueError("mat y pred_df deben compartir índices y columnas")

    mask = mat.values != 0  # considerar únicamente ratings observados
    if not mask.any():
        return float('nan')
    diffs = (mat.values - pred_df.values)[mask]
    mse = (diffs ** 2).mean()
    return float(np.sqrt(mse))


