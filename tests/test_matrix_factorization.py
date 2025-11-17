import numpy as np
from src.matrix_factorization import train_svd

def test_train_svd_shapes():
    mat = np.array([
        [5, 3, 0],
        [4, 0, 1],
        [1, 1, 0]
    ])
    model = train_svd(mat, n_factors=2)
    assert model.U.shape[0] == 3
    assert model.Vt.shape[1] == 3

def test_predict_returns_float():
    mat = np.array([
        [5, 3, 0],
        [4, 0, 1],
        [1, 1, 0]
    ])
    model = train_svd(mat, n_factors=2)
    assert isinstance(model.predict(0,0), float)

def test_recommend_output():
    mat = np.array([
        [5, 3, 0],
        [4, 0, 1],
        [1, 1, 0]
    ])
    model = train_svd(mat, n_factors=2)
    recs = model.recommend(0, n=2)
    assert len(recs) == 2
