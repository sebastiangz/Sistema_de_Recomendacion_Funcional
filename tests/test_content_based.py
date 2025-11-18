import pytest
import numpy as np
import pandas as pd

from src.content_based import (
    extract_features,
    create_content_model,
    ContentBasedModel
)


# -------------------------------------------------------------------
# Fixtures de datos
# -------------------------------------------------------------------

@pytest.fixture
def items_df():
    return pd.DataFrame(
        {
            "title": ["Python Developer", "Data Scientist", "Java Engineer"],
            "description": [
                "Python programming and backend development",
                "Data science, Python, machine learning",
                "Java Spring microservices backend"
            ]
        },
        index=[101, 102, 103]
    )


# -------------------------------------------------------------------
# Tests extract_features
# -------------------------------------------------------------------

def test_extract_features_returns_numpy(items_df):
    feats = extract_features(items_df, ["title", "description"])
    assert isinstance(feats, np.ndarray)


def test_extract_features_shape(items_df):
    feats = extract_features(items_df, ["title", "description"])
    assert feats.shape[0] == len(items_df)  # tantas filas como items


def test_extract_features_different_columns(items_df):
    # Usar solo "description"
    feats1 = extract_features(items_df, ["description"])
    # Usar title + description
    feats2 = extract_features(items_df, ["title", "description"])
    # Deben diferir (más información = vector diferente)
    assert feats1.shape != feats2.shape


# -------------------------------------------------------------------
# Tests create_content_model
# -------------------------------------------------------------------

def test_create_content_model_returns_model(items_df):
    model = create_content_model(items_df, ["title", "description"])
    assert isinstance(model, ContentBasedModel)


def test_create_content_model_item_ids_match(items_df):
    model = create_content_model(items_df, ["title", "description"])
    assert model.item_ids == tuple(items_df.index)


def test_create_content_model_feature_count(items_df):
    model = create_content_model(items_df, ["title", "description"])
    assert model.features.shape[0] == len(items_df)


# -------------------------------------------------------------------
# Tests ContentBasedModel.find_similar_items
# -------------------------------------------------------------------

def test_find_similar_items_returns_sorted(items_df):
    model = create_content_model(items_df, ["title", "description"])
    sims = model.find_similar_items(101, k=2)
    assert len(sims) <= 2
    # Similitud debe estar en orden descendente
    if len(sims) > 1:
        assert sims[0][1] >= sims[1][1]


def test_find_similar_items_excludes_self(items_df):
    model = create_content_model(items_df, ["title", "description"])
    sims = model.find_similar_items(101, k=5)
    item_ids = [iid for iid, _ in sims]
    assert 101 not in item_ids


def test_find_similar_items_valid_ids(items_df):
    model = create_content_model(items_df, ["title", "description"])
    sims = model.find_similar_items(101, k=5)
    for iid, score in sims:
        assert iid in items_df.index
        assert isinstance(score, float)
