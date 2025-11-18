import pytest

from src.hybrid import hybrid_recommend, hybrid_scores


# ------------------------
# Fixtures de funciones simuladas (mocks funcionales)
# ------------------------

@pytest.fixture
def collab_mock():
    def fn(user, n):
        return [("A", 0.9), ("B", 0.7), ("C", 0.4)]
    return fn


@pytest.fixture
def content_mock():
    def fn(item_id):
        sims = {
            "A": [("X", 0.5), ("Y", 0.4)],
            "B": [("X", 0.3)],
            "C": []  # sin similares
        }
        return sims.get(item_id, [])
    return fn


# ------------------------
# Tests hybrid_scores
# ------------------------

def test_hybrid_scores_basic():
    collab = {"A": 1.0, "B": 0.5}
    content_fn = lambda x: {"A": 0.2, "B": 0.4}[x]
    result = hybrid_scores(collab, content_fn, 0.7, 0.3)
    assert result["A"] == pytest.approx(0.7*1.0 + 0.3*0.2)
    assert result["B"] == pytest.approx(0.7*0.5 + 0.3*0.4)


# ------------------------
# Tests hybrid_recommend
# ------------------------

def test_hybrid_recommend_basic(collab_mock, content_mock):
    recs = hybrid_recommend(collab_mock, content_mock, user_id="U1", n=2)
    assert isinstance(recs, list)
    assert len(recs) == 2


def test_hybrid_recommend_sorted(collab_mock, content_mock):
    recs = hybrid_recommend(collab_mock, content_mock, user_id="U1", n=3)
    scores = [score for _, score in recs]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_recommend_items_exist(collab_mock, content_mock):
    recs = hybrid_recommend(collab_mock, content_mock, "U1", 3)
    ids = [item for item, _ in recs]
    assert set(ids).issubset({"A", "B", "C"})
