import pandas as pd
from src.evaluation import precision_at_k, recall_at_k, ndcg_at_k, evaluate_model

def test_precision():
    assert precision_at_k([1,2,3], {2,3}, 3) == 2/3

def test_recall():
    assert recall_at_k([1,2,3], {3}, 3) == 1.0

def test_ndcg():
    recs = [1,2,3]
    scores = {1:3, 2:2, 3:1}
    assert ndcg_at_k(recs, scores, 3) > 0

def test_evaluate_model():
    class Dummy:
        def recommend(self, user, k): return [(1,1.0),(2,0.5)]
    df = pd.DataFrame({"user_id":[1,1],"item_id":[1,2],"rating":[5,4]})
    res = evaluate_model(Dummy(), df, 2)
    assert "precision" in res
