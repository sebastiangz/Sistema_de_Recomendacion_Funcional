"""collaborative_filtering_v2.py
Memory-efficient CF utilities: compute similarities on-the-fly for a target user,
and provide item-based and SVD-based recommendations.
"""

from typing import Tuple, List, Callable, Optional, Dict
import numpy as np
import pandas as pd

def build_user_item_matrix(df: pd.DataFrame,
                           user_col: str,
                           item_col: str,
                           rating_col: Optional[str] = None,
                           fillna: float = 0.0) -> Tuple[pd.DataFrame, Dict[int, object], Dict[int, object]]:
    df_copy = df[[user_col, item_col]].copy()
    if rating_col is not None and rating_col in df.columns:
        df_copy[rating_col] = df[rating_col]
    else:
        rating_col = "_implicit_rating_"
        df_copy[rating_col] = 1.0

    pivot = df_copy.pivot_table(index=user_col, columns=item_col, values=rating_col, aggfunc='mean', fill_value=fillna)
    user_map = {i: uid for i, uid in enumerate(pivot.index)}
    item_map = {j: iid for j, iid in enumerate(pivot.columns)}
    return pivot, user_map, item_map

def _cosine_with_target(mat_arr: np.ndarray, target_vec: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between target_vec and all rows in mat_arr efficiently."""
    # mat_arr: (n_users, n_items), target_vec: (n_items,)
    target_norm = np.linalg.norm(target_vec)
    mat_norms = np.linalg.norm(mat_arr, axis=1)
    # avoid division by zero
    denom = mat_norms * (target_norm if target_norm != 0 else 1.0)
    # dot product
    dots = mat_arr.dot(target_vec)
    sim = np.zeros_like(dots, dtype=float)
    nonzero = denom != 0
    sim[nonzero] = dots[nonzero] / denom[nonzero]
    return sim

def user_top_k_neighbors(mat: pd.DataFrame, user_id, k: int = 5, metric: str = 'cosine') -> List[object]:
    """Return top-k neighbor user IDs for a given user, computed without building full matrix."""
    if user_id not in mat.index:
        return []
    arr = mat.values.astype(float)
    idx = mat.index.get_loc(user_id)
    target = arr[idx, :]
    if metric == 'cosine':
        sims = _cosine_with_target(arr, target)
    elif metric == 'pearson':
        # compute pearson similarity to target
        row_means = np.mean(arr, axis=1)
        demean = arr - row_means[:, None]
        target_demean = target - np.mean(target)
        sims = _cosine_with_target(demean, target_demean)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    sims[idx] = -1.0  # exclude self
    topk_idx = np.argpartition(-sims, range(min(k, len(sims)-1)))[:k]
    # sort those topk
    topk_sorted = topk_idx[np.argsort(-sims[topk_idx])]
    return [mat.index[i] for i in topk_sorted]

def user_based_recommend(mat: pd.DataFrame, user_id, k_neighbors: int = 5, n_recs: int = 5,
                         metric: str = 'cosine') -> List[Tuple[object, float]]:
    if user_id not in mat.index:
        return []
    neighbors = user_top_k_neighbors(mat, user_id, k=k_neighbors, metric=metric)
    if not neighbors:
        return []
    arr = mat.values.astype(float)
    idx = mat.index.get_loc(user_id)
    target_ratings = arr[idx, :]
    candidate_mask = target_ratings == 0
    scores = {}
    # compute similarity vector of target to neighbors only
    for nbr in neighbors:
        nbr_idx = mat.index.get_loc(nbr)
        if metric == 'cosine':
            w = _cosine_with_target(arr, arr[nbr_idx, :])[idx]  # similarity between user and neighbor
        else:
            # fallback compute dot of demeaned
            user_mean = target_ratings.mean()
            nbr_mean = arr[nbr_idx, :].mean()
            num = ((target_ratings - user_mean) * (arr[nbr_idx, :] - nbr_mean)).sum()
            den = np.linalg.norm(target_ratings - user_mean) * np.linalg.norm(arr[nbr_idx, :] - nbr_mean)
            w = num/den if den != 0 else 0.0
        # add neighbor contribution for candidate items
        for j, is_candidate in enumerate(candidate_mask):
            if not is_candidate:
                continue
            r = arr[nbr_idx, j]
            if r != 0:
                scores[j] = scores.get(j, 0.0) + w * r
    # normalize by sum of abs weights per item
    # compute denom per item
    denoms = {}
    for nbr in neighbors:
        nbr_idx = mat.index.get_loc(nbr)
        if metric == 'cosine':
            w = _cosine_with_target(arr, arr[nbr_idx, :])[idx]
        else:
            user_mean = target_ratings.mean()
            nbr_mean = arr[nbr_idx, :].mean()
            num = ((target_ratings - user_mean) * (arr[nbr_idx, :] - nbr_mean)).sum()
            den = np.linalg.norm(target_ratings - user_mean) * np.linalg.norm(arr[nbr_idx, :] - nbr_mean)
            w = num/den if den != 0 else 0.0
        for j, is_candidate in enumerate(candidate_mask):
            if not is_candidate:
                continue
            if arr[nbr_idx, j] != 0:
                denoms[j] = denoms.get(j, 0.0) + abs(w)
    final_scores = {}
    for j, val in scores.items():
        denom = denoms.get(j, 0.0)
        if denom > 0:
            final_scores[mat.columns[j]] = val / denom
    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:n_recs]

def item_based_recommend(mat: pd.DataFrame, user_id, n_recs: int = 5) -> List[Tuple[object, float]]:
    # item-based: compute item-item sims (items are only ~29 here, so full matrix is fine)
    item_arr = mat.T.values.astype(float)
    # cosine similarity between items
    norms = np.linalg.norm(item_arr, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    normalized = item_arr / norms
    item_sim = normalized.dot(normalized.T)
    item_sim_df = pd.DataFrame(item_sim, index=mat.columns, columns=mat.columns)
    # compute scores for items user hasn't rated
    if user_id not in mat.index:
        return []
    user_ratings = mat.loc[user_id]
    candidate_items = user_ratings[user_ratings == 0].index
    scores = {}
    for item in candidate_items:
        sims = item_sim_df.loc[item]
        rated_items = user_ratings[user_ratings != 0].index
        numer = 0.0
        denom = 0.0
        for ri in rated_items:
            w = sims.get(ri, 0.0)
            r = user_ratings[ri]
            numer += w * r
            denom += abs(w)
        if denom > 0:
            scores[item] = numer / denom
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:n_recs]

def svd_recommend(mat: pd.DataFrame, user_id, n_components: int = 10, n_recs: int = 5) -> List[Tuple[object, float]]:
    # For moderate item count, SVD on user-item is ok because items are few.
    A = mat.values.astype(float)
    # center by subtracting global mean can help
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    k = min(n_components, len(s))
    pred = (U[:, :k] * s[:k]) @ Vt[:k, :]
    pred_df = pd.DataFrame(pred, index=mat.index, columns=mat.columns)
    user_orig = mat.loc[user_id]
    candidate_items = user_orig[user_orig == 0].index
    scores = {item: float(pred_df.at[user_id, item]) for item in candidate_items}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:n_recs]

