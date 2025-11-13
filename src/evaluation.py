"""
evaluation.py
Métricas de evaluación funcionales.
"""

import numpy as np
from typing import List, Dict, Set

def precision_at_k(recs: List[int], relevant: Set[int], k: int) -> float:
    return len(set(recs[:k]) & relevant) / k if k > 0 else 0

def recall_at_k(recs: List[int], relevant: Set[int], k: int) -> float:
    return len(set(recs[:k]) & relevant) / len(relevant) if relevant else 0

def ndcg_at_k(recs: List[int], scores: Dict[int, float], k: int) -> float:
    def dcg(vals): return sum((2**v - 1)/np.log2(i+2) for i, v in enumerate(vals))
    actual = [scores.get(i, 0) for i in recs[:k]]
    ideal = sorted(scores.values(), reverse=True)[:k]
    return dcg(actual)/dcg(ideal) if sum(ideal) > 0 else 0

def evaluate_model(model, test_df, k=10) -> Dict[str, float]:
    metrics = {'precision': [], 'recall': [], 'ndcg': []}
    for uid in test_df['user_id'].unique():
        recs = [r[0] for r in model.recommend(uid, k)]
        relevant = set(test_df[test_df['user_id']==uid]['item_id'])
        scores = dict(test_df[test_df['user_id']==uid][['item_id','rating']].values)
        metrics['precision'].append(precision_at_k(recs, relevant, k))
        metrics['recall'].append(recall_at_k(recs, relevant, k))
        metrics['ndcg'].append(ndcg_at_k(recs, scores, k))
    return {m: np.mean(v) for m, v in metrics.items()}
