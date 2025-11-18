import pytest
import pandas as pd
import numpy as np

from src.collaborative import (
    build_user_item_matrix,
    user_top_k_neighbors,
    user_based_recommend,
    item_based_recommend
)

# --------------------------------------------------------------------
# Fixtures de datos
# --------------------------------------------------------------------

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "user": ["U1", "U1", "U2", "U2", "U3"],
        "item": ["A", "B", "A", "C", "B"],
        "rating": [5, 3, 4, 2, 4]
    })

@pytest.fixture
def interaction_matrix():
    return pd.DataFrame(
        [
            [5, 3, 0],   # U1
            [4, 0, 2],   # U2
            [0, 4, 0]    # U3
        ],
        index=["U1", "U2", "U3"],
        columns=["A", "B", "C"]
    )

# --------------------------------------------------------------------
# Tests for build_user_item_matrix
# --------------------------------------------------------------------

def test_build_user_item_matrix_basic(sample_df):
    mat, users, items = build_user_item_matrix(
        sample_df, user_col="user", item_col="item", rating_col="rating"
    )
    assert isinstance(mat, pd.DataFrame)
    assert set(mat.index) == {"U1", "U2", "U3"}
    assert set(mat.columns) == {"A", "B", "C"}


def test_build_user_item_matrix_implicit_rating(sample_df):
    df = sample_df.drop(columns=["rating"])
    mat, users, items = build_user_item_matrix(df, "user", "item")
    assert mat.values.max() == 1.0  # ratings implícitos


# --------------------------------------------------------------------
# Tests for user_top_k_neighbors
# --------------------------------------------------------------------

def test_user_top_k_neighbors_valid(interaction_matrix):
    neighbors = user_top_k_neighbors(interaction_matrix, user_id="U1", k=2)
    assert isinstance(neighbors, list)
    assert len(neighbors) <= 2
    assert "U1" not in neighbors  # no debe incluirse a sí mismo


def test_user_top_k_neighbors_user_not_found(interaction_matrix):
    assert user_top_k_neighbors(interaction_matrix, "Unknown") == []


def test_user_top_k_neighbors_invalid_metric(interaction_matrix):
    with pytest.raises(ValueError):
        user_top_k_neighbors(interaction_matrix, "U1", metric="invalid")


# --------------------------------------------------------------------
# Tests for user_based_recommend
# --------------------------------------------------------------------

def test_user_based_recommend_basic(interaction_matrix):
    recs = user_based_recommend(interaction_matrix, "U1", n_recs=2)
    assert isinstance(recs, list)
    assert len(recs) <= 2
    # U1 no ha valorado C, debe ser candidato
    items = [item for item, score in recs]
    assert "C" in items


def test_user_based_recommend_no_user(interaction_matrix):
    assert user_based_recommend(interaction_matrix, "Unknown") == []


# --------------------------------------------------------------------
# Tests for item_based_recommend
# --------------------------------------------------------------------

def test_item_based_recommend_basic(interaction_matrix):
    recs = item_based_recommend(interaction_matrix, "U1", n_recs=2)
    assert isinstance(recs, list)
    assert len(recs) <= 2
    items = [item for item, score in recs]
    assert "C" in items  # ítem no valorado por U1


def test_item_based_recommend_user_not_found(interaction_matrix):
    assert item_based_recommend(interaction_matrix, "Unknown") == []

