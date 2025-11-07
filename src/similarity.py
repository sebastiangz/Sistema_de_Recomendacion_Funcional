"""
similarity.py
Módulo funcional para calcular métricas de similitud reutilizables.

Principios aplicados:
- Funciones puras e inmutables
- Composición funcional (funciones como argumentos)
- Operaciones vectorizadas (numpy)
- Sin efectos secundarios

"""

from typing import Callable, Union
import numpy as np
from functools import partial

# ==========================
# Funciones auxiliares puras
# ==========================

def _safe_norm(v: np.ndarray) -> float:
    """Norma segura (evita división entre cero)."""
    norm = np.linalg.norm(v)
    return norm if norm != 0 else 1.0

# ==========================
# Métricas de similitud
# ==========================

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Similitud de coseno funcional."""
    return float(np.dot(v1, v2) / (_safe_norm(v1) * _safe_norm(v2)))


def euclidean_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Transforma distancia euclidiana en similitud."""
    dist = np.linalg.norm(v1 - v2)
    return float(1 / (1 + dist))


def manhattan_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Similitud basada en distancia Manhattan."""
    dist = np.sum(np.abs(v1 - v2))
    return float(1 / (1 + dist))


def pearson_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Coeficiente de correlación de Pearson funcional."""
    if np.std(v1) == 0 or np.std(v2) == 0:
        return 0.0
    return float(np.corrcoef(v1, v2)[0, 1])

# ==========================
# Similitud generalizable
# ==========================

def similarity_matrix(X: np.ndarray,
                      metric: Callable[[np.ndarray, np.ndarray], float] = cosine_similarity
                     ) -> np.ndarray:
    """
    Calcula una matriz de similitud entre todas las filas de X.

    Parámetros
    ----------
    X : np.ndarray
        Matriz (n_samples, n_features)
    metric : Callable
        Función de similitud (por defecto, coseno)

    Retorna
    -------
    np.ndarray : matriz (n_samples, n_samples)
    """
    n = X.shape[0]
    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            s = metric(X[i], X[j])
            sim[i, j] = sim[j, i] = s
    return sim

# ==========================
# Versiones parciales (funciones compuestas)
# ==========================

cosine_matrix = partial(similarity_matrix, metric=cosine_similarity)
pearson_matrix = partial(similarity_matrix, metric=pearson_similarity)
euclidean_matrix = partial(similarity_matrix, metric=euclidean_similarity)

# ==========================
# Función composable
# ==========================

def similarity_fn(metric_name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Devuelve una función de similitud según el nombre solicitado.

    Parámetros
    ----------
    metric_name : str
        Una de ['cosine', 'pearson', 'euclidean', 'manhattan']

    Retorna
    -------
    Callable : función de similitud correspondiente
    """
    mapping = {
        'cosine': cosine_similarity,
        'pearson': pearson_similarity,
        'euclidean': euclidean_similarity,
        'manhattan': manhattan_similarity
    }
    return mapping.get(metric_name.lower(), cosine_similarity)
