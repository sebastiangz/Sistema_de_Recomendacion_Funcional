from src.hybrid import HybridRecommender

class DummyModel:
    def recommend(self, uid, n):
        return [(1, 0.9), (2, 0.7)]

    def find_similar_items(self, item_id, k):
        return [(3, 0.2), (4, 0.1)]

def test_hybrid_recommend():
    hybrid = HybridRecommender(DummyModel(), DummyModel(), weights=(0.7, 0.3))
    recs = hybrid.recommend(0, n=2)
    assert len(recs) == 2

def test_hybrid_weights():
    hybrid = HybridRecommender(DummyModel(), DummyModel(), weights=(1.0, 0.0))
    recs = hybrid.recommend(0, n=1)
    assert len(recs) == 1

def test_hybrid_type():
    hybrid = HybridRecommender(DummyModel(), DummyModel())
    recs = hybrid.recommend(1, 1)
    assert isinstance(recs[0], tuple)
