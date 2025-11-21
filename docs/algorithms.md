# Algorithms Used in the Functional Recommendation System

Este documento describe de manera formal y técnica los algoritmos utilizados en el Sistema de Recomendación Funcional desarrollado para el Proyecto 9.

---

# 1. Collaborative Filtering (User-Based)

## 1.1. Construcción de la matriz usuario–ítem
**Función:** `build_user_item_matrix(df, user_col, item_col, rating_col=None)`

- Crea una matriz `M` donde:
  - Filas = usuarios
  - Columnas = ítems
  - Celdas = rating (o 1.0 si es implícito)
- Los ceros representan ítems no vistos.

Matemáticamente:

\[
M_{u,i} =
\begin{cases}
r_{u,i} & \text{si existe rating} \\
0       & \text{si no existe}
\end{cases}
\]

---

## 1.2. User-Based KNN Similarity

### Similitud del coseno
\[
\text{sim}(u, v) = \frac{u \cdot v}{||u|| \, ||v||}
\]

El sistema calcula la similitud del usuario objetivo contra todos los demás.

### Selección de vecinos
Se seleccionan los **k usuarios con mayor similitud**:

\[
N_k(u) = \text{TopK}(\text{sim}(u, v))
\]

### Predicción de ratings
Para un ítem no visto por el usuario:

\[
\hat{r}_{u,i} = \frac{\sum_{v \in N_k(u)} sim(u, v) \cdot r_{v,i}}
                    {\sum_{v \in N_k(u)} |sim(u, v)|}
\]

---

# 2. SVD Matrix Factorization

**Función:** `svd_recommend(mat, user_id, n_components, n_recs)`

La matriz usuario–ítem se aproxima como:

\[
M \approx U_k \Sigma_k V_k^T
\]

La predicción estimada:

\[
\hat{M} = U_k \Sigma_k V_k^T
\]

Luego:
- Para cada usuario se toman los ítems no vistos.
- Se ordenan por mayor score en `\hat{M}`.

Ventajas:
- Maneja matrices muy dispersas.
- Produce mejores recomendaciones globales que el KNN simple.

---

# 3. Content-Based Filtering (TF-IDF)

**Funciones:**
- `extract_features(items_df, text_cols)`
- `ContentBasedModel.find_similar_items(item_id)`

### 3.1. Construcción del documento por ítem
Los textos se combinan:

\[
T_i = \text{col1}_i + " " + \text{col2}_i + …
\]

### 3.2. Vectorización TF-IDF
Se usa:

\[
\text{tfidf}(t,i) = tf(t,i) \cdot \log \frac{N}{df(t)}
\]

Obtiene una matriz:

- Filas = ítems  
- Columnas = features TF-IDF

### 3.3. Similitud entre ítems

\[
\text{sim}(i,j) = \frac{v_i \cdot v_j}{||v_i|| \, ||v_j||}
\]

---

# 4. Hybrid Recommendation

**Función:** `hybrid_recommend(collab_recommender, content_sim_fn, user_id, n, w_collab, w_content)`

Combine dos señales:

\[
\text{score}_h(i) = w_c \cdot \text{score}_c(i) + w_{cb} \cdot \text{bonus}(i)
\]

Donde:

- `score_c(i)` → score colaborativo
- `bonus(i)` → promedio de similitud de los 3 ítems más parecidos

Beneficios:
- Mayor diversidad
- Mejor precisión en escenarios con pocos datos colaborativos
- Aprovecha semántica textual

---

# 5. Evaluation Algorithms

Métricas aplicadas:

### Precision@k
\[
P@k = \frac{|recs_k \cap relevantes|}{k}
\]

### Recall@k
\[
R@k = \frac{|recs_k \cap relevantes|}{|relevantes|}
\]

### NDCG@k
\[
NDCG@k = \frac{DCG@k}{IDCG@k}
\]

---

# 6. Flujo General del Sistema

1. Construcción de matrices (colaborativo y contenido)
2. Cálculo de similitudes
3. Scoring colaborativo / por contenido
4. Combinación híbrida
5. Evaluación con Precision/Recall/NDCG

---

# Fin del documento
