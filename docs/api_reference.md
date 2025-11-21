# API Reference — Functional Recommendation System

Este documento describe la API completa del sistema de recomendación funcional.

---

# 1. collaborative.py

## build_user_item_matrix(df, user_col, item_col, rating_col=None)
→ Devuelve: `(pivot_df, user_map, item_map)`

## user_based_recommend(mat, user_id, k_neighbors=5, n_recs=5)
→ Lista de tuplas `(item, score)`

## item_based_recommend(mat, user_id, n_recs=5)
→ Basado en similitud entre ítems

## svg_recommend(mat, user_id, n_components=10, n_recs=5)
→ Recomendación usando SVD

---

# 2. content_based.py

## extract_features(items_df, text_cols)
→ Devuelve matriz TF-IDF

## create_content_model(items_df, text_cols)
→ Devuelve un `ContentBasedModel`

## ContentBasedModel.find_similar_items(item_id, k)
→ Retorna ítems similares con score

---

# 3. hybrid.py

## hybrid_recommend(collab_recommender, content_sim_fn, user_id, n, w_collab, w_content)
Genera recomendaciones híbridas combinando:

- señales colaborativas  
- similitudes basadas en contenido

→ Devuelve lista de tuplas `(item, score)`

---

# 4. recommender.py (API funcional)

## create_collaborative_filter(method="user")
→ Retorna un dict:  
`{"fit": fit_fn}`

**fit_fn(df, user_col, item_col)**  
→ Retorna `recommend_fn(user_id, n)`

## create_content_based(feature_columns)
→ Retorna un dict con:
`{"fit": fit_fn}`

**fit_fn(items_df)**  
→ Retorna `recommend(item_id, k)`

## create_hybrid_recommender(collab_fn, content_fn, weights)
→ Retorna `recommend(user_id, n)`

---

# 5. evaluation.py

## precision_at_k(recs, relevant, k)
## recall_at_k(recs, relevant, k)
## ndcg_at_k(recs, scores, k)

## evaluate_model(model, test_df, k)
Acepta:
- `model = {"recommend": fn}`
- `model = fn`
- `model = objeto con .recommend()`

→ Devuelve diccionario:
{
"precision": float,
"recall": float,
"ndcg": float
}

---

# 6. pipeline.py

## create_recommendation_pipeline(collab_model, content_model, weights)
→ Devuelve un pipeline híbrido

---

# Fin del documento

