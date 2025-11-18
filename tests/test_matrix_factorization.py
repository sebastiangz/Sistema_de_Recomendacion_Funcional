import numpy as np
from src.matrix_factorization import train_svd, svd_recommend, SVDModel

def test_train_svd_output_type():
    matrix = np.random.rand(5, 5)
    model = train_svd(matrix, n_factors=3)
    assert isinstance(model, SVDModel)

def test_train_svd_dimensions():
    matrix = np.random.rand(8, 6)
    model = train_svd(matrix, n_factors=4)
    assert model.U.shape == (8, 4)
    assert model.S.shape == (4, 4)
    assert model.Vt.shape == (4, 6)

def test_svd_reconstruct():
    matrix = np.random.rand(6, 6)
    model = train_svd(matrix, n_factors=3)
    reconstruction = svd_recommend(model)
    assert reconstruction.shape == matrix.shape