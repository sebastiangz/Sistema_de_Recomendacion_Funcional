import pytest
from src.pipeline import create_pipeline


# -------------------------
# Fixtures: mocks funcionales
# -------------------------

@pytest.fixture
def collab_mock():
    def fn(user, n):
        return [("A", 0.8), ("B", 0.6), ("C", 0.3)]
    return fn


@pytest.fixture
def content_mock():
    def fn(item):
        sims = {
            "A": [("X", 0.4), ("Y", 0.3)],
            "B": [("Y", 0.2)],
            "C": []
        }
        return sims.get(item, [])
    return fn


# -------------------------
# Tests
# -------------------------

def test_create_pipeline_returns_callable(collab_mock, content_mock):
    pipe = create_pipeline(collab_mock, content_mock)
    assert callable(pipe)


def test_pipeline_recommend_returns_list(collab_mock, content_mock):
    pipe = create_pipeline(collab_mock, content_mock)
    recs = pipe("U1", 2)
    assert isinstance(recs, list)
    assert len(recs) <= 2


def test_pipeline_recommend_sorted(collab_mock, content_mock):
    pipe = create_pipeline(collab_mock, content_mock)
    recs = pipe("U1", 3)
    scores = [score for _, score in recs]
    assert scores == sorted(scores, reverse=True)


def test_pipeline_custom_weights(collab_mock, content_mock):
    pipe = create_pipeline(collab_mock, content_mock, weights=(0.2, 0.8))
    recs = pipe("U1", 1)
    # Solo probamos que funciona sin errores y retorna resultado válido
    assert isinstance(recs, list)
