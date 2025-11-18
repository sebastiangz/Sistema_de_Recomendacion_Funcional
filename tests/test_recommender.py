import pytest
import pandas as pd
from src.recommender import (
    create_collaborative_filter,
    create_content_based,
    create_hybrid_recommender,
)


# -------------------------------------------------------
# FIXTURES
# -------------------------------------------------------

@pytest.fixture
def ratings_df():
    return pd.DataFrame({
        "user": ["U1", "U1", "U2", "U3"],
        "item": ["A", "B", "A", "B"],
        "rating": [5, 3, 4, 2]
    })


@pytest.fixture
def items_df():
    return pd.DataFrame({
        "title": ["Python Dev", "Data Scientist", "Java Engineer"],
        "desc": [
            "backend python developer",
            "machine learning python data",
            "java spring backend"
        ]
    }, index=["A", "B", "C"])


# -------------------------------------------------------
# TESTS COLLABORATIVE
# -------------------------------------------------------

def test_create_collab_recommender_user_based(ratings_df):
    fit = create_collaborative_filter("user")
    recommend = fit(ratings_df, "user", "item")
    result = recommend("U1", 2)
    assert isinstance(result, list)


def test_create_collab_recommender_svd(ratings_df):
    fit = create_collaborative_filter("svd")
    recommend = fit(ratings_df, "user", "item")
    result = recommend("U1", 2)
    assert isinstance(result, list)


# -------------------------------------------------------
# TESTS CONTENT-BASED
# -------------------------------------------------------

def test_create_content_based(items_df):
    fit = create_content_based(["title", "desc"])
    recommend = fit(items_df)
    result = recommend("A", 2)
    assert isinstance(result, list)


# -------------------------------------------------------
# TESTS HYBRID
# -------------------------------------------------------

def test_create_hybrid_recommender(ratings_df, items_df):

    # Collaborative
    collab_fit = create_collaborative_filter("user")
    collab_recommend = collab_fit(ratings_df, "user", "item")

    # Content based
    content_fit = create_content_based(["title", "desc"])
    content_recommend = content_fit(items_df)

    hybrid = create_hybrid_recommender(
        collab_recommend, content_recommend, weights=(0.7, 0.3)
    )

    result = hybrid("U1", 3)
    assert isinstance(result, list)
    assert len(result) <= 3
