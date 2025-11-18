import numpy as np
from src.similarity import jaccard_similarity

def test_jaccard_similarity_zero():
    v1 = np.array([0, 0, 0])
    v2 = np.array([0, 0, 0])
    assert jaccard_similarity(v1, v2) == 0.0

def test_jaccard_similarity_full():
    v1 = np.array([1, 1, 1])
    v2 = np.array([1, 1, 1])
    assert jaccard_similarity(v1, v2) == 1.0

def test_jaccard_similarity_partial():
    v1 = np.array([1, 0, 1])
    v2 = np.array([1, 1, 0])
    val = jaccard_similarity(v1, v2)
    assert 0 < val < 1