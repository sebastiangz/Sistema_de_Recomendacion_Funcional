import pytest
import pandas as pd
import numpy as np

from src.evaluation import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    evaluate_model
)

# ----------------------------------------------------------
# Tests para precision_at_k
# ----------------------------------------------------------

def test_precision_at_k_basic():
    recs = [1, 2, 3, 4]
    relevant = {2, 4}
    assert precision_at_k(recs, relevant, 2) == 0.5  # top-2 solo contiene {2}


def test_precision_at_k_zero_k():
    assert precision_at_k([1,2], {1}, 0) == 0


def test_precision_at_k_no_overlap():
    assert precision_at_k([1,2,3], {9,8}, 3) == 0


# ----------------------------------------------------------
# Tests para recall_at_k
# ----------------------------------------------------------

def test_recall_at_k_basic():
    recs = [1, 2, 3, 4]
    relevant = {2, 4}
    assert recall_at_k(recs, relevant, 4) == 1.0  # ambos relevantes dentro de los 4


def test_recall_at_k_empty_relevant():
    assert recall_at_k([1,2,3], set(), 3) == 0


# ----------------------------------------------------------
# Tests para ndcg_at_k
# ----------------------------------------------------------

def test_ndcg_at_k_basic():
    recs = [1, 2, 3]
    scores = {1: 3, 2: 2, 3: 1}
    val = ndcg_at_k(recs, scores, 3)
    assert 0 <= val <= 1
    assert pytest.approx(val, rel=1e-6) == 1.0  # perfecto orden


def test_ndcg_at_k_ideal_zero():
    recs = [1, 2, 3]
    scores = {1: 0, 2: 0, 3: 0}
    assert ndcg_at_k(recs, scores, 3) == 0


# ----------------------------------------------------------
# Test para evaluate_model (con mock funcional)
# ----------------------------------------------------------

@pytest.fixture
def mock_model():
    class M:
        def recommend(self, user_id, k):
            return [(101, 5), (102, 4), (103, 3)]
    return M()


@pytest.fixture
def test_df():
    return pd.DataFrame([
        {"user_id": "U1", "item_id": 101, "rating": 5},
        {"user_id": "U1", "item_id": 999, "rating": 1},  # irrelevante recomendado
        {"user_id": "U2", "item_id": 103, "rating": 3}
    ])


def test_evaluate_model_returns_dict(mock_model, test_df):
    metrics = evaluate_model(mock_model, test_df, k=3)
    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == {"precision", "recall", "ndcg"}


def test_evaluate_model_metric_ranges(mock_model, test_df):
    metrics = evaluate_model(mock_model, test_df, k=3)
    assert all(0 <= v <= 1 for v in metrics.values())
