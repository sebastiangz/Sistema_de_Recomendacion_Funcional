import pytest
import numpy as np

from src.similarity import (
    cosine_similarity,
    euclidean_similarity,
    manhattan_similarity,
    pearson_similarity,
    similarity_matrix,
    cosine_matrix,
    pearson_matrix,
    euclidean_matrix,
    similarity_fn
)

# -----------------------
# Fixtures de datos base
# -----------------------

@pytest.fixture
def v1():
    return np.array([1.0, 2.0, 3.0])

@pytest.fixture
def v2():
    return np.array([2.0, 4.0, 6.0])  # proporcional a v1 → coseno=1, pearson=1


@pytest.fixture
def matrix():
    return np.array([
        [1.0, 0.0, 2.0],
        [2.0, 1.0, 0.0],
        [0.0, 1.0, 1.0]
    ])


# -----------------------
# Tests: funciones básicas
# -----------------------

def test_cosine_similarity_basic(v1, v2):
    result = cosine_similarity(v1, v2)
    assert pytest.approx(result, rel=1e-6) == 1.0


def test_cosine_similarity_with_zero_vector():
    v = np.array([0.0, 0.0, 0.0])
    result = cosine_similarity(v, v)
    assert isinstance(result, float)


def test_euclidean_similarity_range(v1, v2):
    result = euclidean_similarity(v1, v2)
    assert 0 < result <= 1


def test_manhattan_similarity_range(v1, v2):
    result = manhattan_similarity(v1, v2)
    assert 0 < result <= 1


def test_pearson_similarity_basic(v1, v2):
    result = pearson_similarity(v1, v2)
    assert pytest.approx(result, rel=1e-6) == 1.0


def test_pearson_similarity_zero_variance():
    v = np.array([1.0, 1.0, 1.0])
    result = pearson_similarity(v, v)
    assert result == 0.0


# -----------------------
# Tests: similarity_matrix
# -----------------------

def test_similarity_matrix_shape(matrix):
    sim = similarity_matrix(matrix)
    assert sim.shape == (matrix.shape[0], matrix.shape[0])


def test_similarity_matrix_diagonal_ones(matrix):
    sim = similarity_matrix(matrix, metric=cosine_similarity)
    diag = np.diag(sim)
    assert all(abs(x - 1.0) < 1e-6 for x in diag)


def test_similarity_matrix_symmetric(matrix):
    sim = similarity_matrix(matrix)
    assert np.allclose(sim, sim.T)


# -----------------------
# Tests: versiones parciales
# -----------------------

def test_cosine_matrix(matrix):
    sim = cosine_matrix(matrix)
    assert sim.shape == (3, 3)


def test_pearson_matrix(matrix):
    sim = pearson_matrix(matrix)
    assert sim.shape == (3, 3)


def test_euclidean_matrix(matrix):
    sim = euclidean_matrix(matrix)
    assert sim.shape == (3, 3)


# -----------------------
# Tests: similarity_fn
# -----------------------

def test_similarity_fn_cosine(v1, v2):
    fn = similarity_fn("cosine")
    assert fn(v1, v2) == pytest.approx(1.0, rel=1e-6)


def test_similarity_fn_default(v1, v2):
    fn = similarity_fn("unknown_metric")
    assert fn(v1, v2) == pytest.approx(1.0, rel=1e-6)

