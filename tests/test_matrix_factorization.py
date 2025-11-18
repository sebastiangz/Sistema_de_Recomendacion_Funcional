import pytest
import pandas as pd
import numpy as np

from src.matrix_factorization import (
    svd_predict_matrix,
    svd_recommend,
    svd_rmse
)

# ---------------------------------------------------------------------
# FIXTURES DE DATOS DE PRUEBA
# ---------------------------------------------------------------------

@pytest.fixture
def small_matrix():
    # Matriz usuario-item pequeña, conocida y controlada
    data = {
        "ItemA": [5, 4, 0],
        "ItemB": [3, 0, 4],
        "ItemC": [0, 2, 3],
    }
    index = ["User1", "User2", "User3"]
    return pd.DataFrame(data, index=index)


@pytest.fixture
def full_matrix():
    # Matriz sin ceros: predicción debería aproximarse a la original
    data = {
        "ItemA": [5, 4, 3],
        "ItemB": [3, 5, 4],
        "ItemC": [4, 3, 5],
    }
    index = ["User1", "User2", "User3"]
    return pd.DataFrame(data, index=index)

# ---------------------------------------------------------------------
# TESTS PARA svd_predict_matrix
# ---------------------------------------------------------------------

def test_svd_predict_matrix_returns_dataframe(small_matrix):
    pred = svd_predict_matrix(small_matrix, n_components=2)
    assert isinstance(pred, pd.DataFrame)


def test_svd_predict_matrix_same_shape(small_matrix):
    pred = svd_predict_matrix(small_matrix, n_components=2)
    assert pred.shape == small_matrix.shape


def test_svd_predict_matrix_raises_typeerror():
    with pytest.raises(TypeError):
        svd_predict_matrix(mat=[1, 2, 3], n_components=2)

# ---------------------------------------------------------------------
# TESTS PARA svd_recommend
# ---------------------------------------------------------------------

def test_svd_recommend_user_not_found(small_matrix):
    recs = svd_recommend(small_matrix, user_id="UnknownUser")
    assert recs == []


def test_svd_recommend_returns_limited_results(small_matrix):
    recs = svd_recommend(small_matrix, user_id="User1", n_recs=2)
    assert len(recs) <= 2


def test_svd_recommend_recommends_unrated_items(small_matrix):
    recs = svd_recommend(small_matrix, "User1", n_components=2, n_recs=5)
    recommended_items = [item for item, _ in recs]
    # User1 tenía 0 en ItemC por lo tanto debe estar recomendado
    assert "ItemC" in recommended_items

# ---------------------------------------------------------------------
# TESTS PARA svd_rmse
# ---------------------------------------------------------------------

def test_svd_rmse_valid_result(full_matrix):
    pred = svd_predict_matrix(full_matrix, n_components=2)
    error = svd_rmse(full_matrix, pred)
    assert isinstance(error, float)


def test_svd_rmse_shape_mismatch_raises(full_matrix):
    wrong = full_matrix.copy().drop(columns=["ItemA"])
    with pytest.raises(ValueError):
        svd_rmse(full_matrix, wrong)


def test_svd_rmse_nan_on_no_observed_values():
    mat = pd.DataFrame([[0, 0], [0, 0]])
    pred = mat.copy()
    result = svd_rmse(mat, pred)
    assert np.isnan(result)

