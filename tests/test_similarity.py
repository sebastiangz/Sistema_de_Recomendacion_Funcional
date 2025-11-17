import numpy as np
from src.similarity import (
    cosine_similarity, pearson_similarity,
    jaccard_similarity, euclidean_similarity,
    pairwise_similarity
)

def test_cosine_similarity_basic():
    v1, v2 = np.array([1, 1]), np.array([1, 1])
    assert cosine_similarity(v1, v2) == 1.0

def test_pearson_similarity_zero_variance():
    v1, v2 = np.array([2, 2, 2]), np.array([1, 1, 1])
    assert pearson_similarity(v1, v2) == 0.0

def test_jaccard_similarity_binary():
    v1, v2 = np.array([1, 0, 1]), np.array([1, 1, 0])
    assert jaccard_similarity(v1, v2) == 1/3

def test_euclidean_similarity():
    v1, v2 = np.array([0, 0]), np.array([3, 4])
    assert euclidean_similarity(v1, v2) == 1 / (1 + 5)

def test_pairwise_similarity_shape():
    mat = np.array([[1,0],[0,1]])
    sim = pairwise_similarity(mat)
    assert sim.shape == (2,2)
