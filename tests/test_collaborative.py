import pandas as pd
from src.collaborative import build_user_item_matrix, user_based_recommend

def sample_df():
    return pd.DataFrame({
        "user": ["A", "A", "B", "C"],
        "item": ["X", "Y", "X", "Y"],
        "rating": [5, 3, 4, 2]
    })

def test_matrix_build():
    mat, users, items = build_user_item_matrix(sample_df(), "user", "item", "rating")
    assert mat.shape == (3,2)

def test_recommend_user_exists():
    mat, _, _ = build_user_item_matrix(sample_df(), "user", "item", "rating")
    recs = user_based_recommend(mat, "A", n_recs=2)
    assert isinstance(recs, list)

def test_recommend_user_not_found():
    mat, _, _ = build_user_item_matrix(sample_df(), "user", "item", "rating")
    assert user_based_recommend(mat, "Z") == []
