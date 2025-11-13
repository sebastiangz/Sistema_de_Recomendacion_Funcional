# 🎯 Proyecto 9: Sistema de Recomendación Funcional

## 📋 Descripción del Proyecto

Sistema de recomendación implementado con programación funcional, utilizando filtrado colaborativo, content-based filtering y algoritmos de ML funcionales para generar recomendaciones personalizadas.

Descripción de NUESTRO proyecto: El proyecto de nuestro equipo tiene como objetivo el servir como un sistema de 
recomendación, el cual servirá para encontrar y recomendar a trabajadores con base en ciertas características, como por ejemplo: Departmento, Título, Fecha de Contratación, Localización, Rating de desempeño, Años de experiencia, Status, Modo de trabajo, Salario INR.

Casos de uso específicos: Este Proyecto tiene la intención de poder ser usado por empresas que buscan a trabajadores con ciertas competencias; si quisieramos que el proyecto fuera a la viceversa, ocuparíamos tener un banco de datos diferente en el que en lugar de tener a trabajadores, tuvieramos datos de empresas.



**Universidad de Colima - Ingeniería en Computación Inteligente**  
**Materia**: Programación Funcional  
**Profesor**: Gonzalez Zepeda Sebastian  
**Semestre**: Agosto 2025 - Enero 2026

---

## 🎯 Objetivos

- Implementar **algoritmos de recomendación** con funciones puras
- Desarrollar **matriz factorization** funcional
- Aplicar **similarity metrics** composables
- Crear **pipelines de ML** funcionales
- Utilizar **lazy evaluation** para grandes datasets
- Practicar **composición de transformadores**

---

## 🛠️ Tecnologías Utilizadas

- **Lenguaje**: Python 3.11+
- **Paradigma**: Programación Funcional
- **Librerías**:
  - `numpy` - Álgebra lineal
  - `pandas` - DataFrames
  - `scipy` - Métricas de similitud
  - `toolz` - Utilidades funcionales
  - `scikit-learn` - Algoritmos ML

---

## 📦 Instalación

```bash
# Clonar el repositorio
git clone https://github.com/sebastiangz/Sistema_de_Recomendacion_Funcional.git
cd SISTEMA_DE_RECOMENDACION_FUNCIONAL

# Crear entorno virtual
python -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### requirements.txt
```
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
scikit-learn>=1.3.0
toolz>=0.12.0
implicit>=0.7.0
surprise>=1.1.0
```

---

## 🚀 Uso del Sistema

```python
from src.recommender import CollaborativeFilter, ContentBased
from src.pipeline import create_recommendation_pipeline

# Cargar datos
ratings = load_ratings('data/ratings.csv')
items = load_items('data/items.csv')

# Crear sistema híbrido
recommender = create_recommendation_pipeline(
    collaborative=CollaborativeFilter(method='svd'),
    content_based=ContentBased(features=['genre', 'director']),
    weights=(0.7, 0.3)
)

# Entrenar modelo
model = recommender.fit(ratings, items)

# Generar recomendaciones
recommendations = model.recommend(
    user_id=123,
    n_items=10,
    exclude_seen=True
)

print(recommendations)
# [(item_id, score), ...]
```

---

## 📂 Estructura del Proyecto

```
recommendation-system-functional/
├── src/
│   ├── __init__.py
│   ├── recommender.py      # Sistemas de recomendación
│   ├── collaborative.py    # Filtrado colaborativo
│   ├── content_based.py    # Content-based filtering
│   ├── hybrid.py           # Sistema híbrido
│   ├── similarity.py       # Métricas de similitud
│   ├── matrix_factorization.py  # SVD, ALS funcional
│   ├── evaluation.py       # Métricas de evaluación
│   └── pipeline.py         # Pipelines composables
├── tests/
│   ├── test_collaborative.py
│   ├── test_content.py
│   └── test_evaluation.py
├── data/
│   ├── raw/               # Datos crudos
│   └── processed/         # Datos procesados
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── model_comparison.ipynb
├── examples/
│   ├── movie_recommendations.py
│   └── product_recommendations.py
├── docs/
│   ├── algorithms.md
│   └── api_reference.md
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔑 Características Principales

### 1. Filtrado Colaborativo Funcional
```python
from typing import Callable, Dict, Tuple
from functools import reduce
import numpy as np

@dataclass(frozen=True)
class UserItemMatrix:
    """Matriz usuario-item inmutable"""
    data: np.ndarray
    user_ids: tuple
    item_ids: tuple
    
    def get_user_vector(self, user_id: int) -> np.ndarray:
        """Obtener vector de usuario"""
        idx = self.user_ids.index(user_id)
        return self.data[idx, :]
    
    def get_item_vector(self, item_id: int) -> np.ndarray:
        """Obtener vector de item"""
        idx = self.item_ids.index(item_id)
        return self.data[:, idx]

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Similitud de coseno funcional"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

def find_similar_users(
    matrix: UserItemMatrix,
    user_id: int,
    k: int = 10,
    similarity_fn: Callable = cosine_similarity
) -> List[Tuple[int, float]]:
    """Encontrar usuarios similares funcionalmente"""
    target_vector = matrix.get_user_vector(user_id)
    
    similarities = [
        (uid, similarity_fn(target_vector, matrix.get_user_vector(uid)))
        for uid in matrix.user_ids
        if uid != user_id
    ]
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

def collaborative_filter_predict(
    matrix: UserItemMatrix,
    user_id: int,
    item_id: int,
    k: int = 10
) -> float:
    """Predecir rating usando k-NN funcional"""
    similar_users = find_similar_users(matrix, user_id, k)
    
    if not similar_users:
        return matrix.data.mean()
    
    # Weighted average de ratings de usuarios similares
    numerator = sum(
        sim * matrix.get_user_vector(uid)[matrix.item_ids.index(item_id)]
        for uid, sim in similar_users
        if matrix.get_user_vector(uid)[matrix.item_ids.index(item_id)] > 0
    )
    
    denominator = sum(
        sim for uid, sim in similar_users
        if matrix.get_user_vector(uid)[matrix.item_ids.index(item_id)] > 0
    )
    
    return numerator / denominator if denominator > 0 else matrix.data.mean()
```

### 2. Matrix Factorization Funcional (SVD)
```python
from scipy.sparse.linalg import svds

@dataclass(frozen=True)
class SVDModel:
    """Modelo SVD inmutable"""
    U: np.ndarray  # User factors
    sigma: np.ndarray  # Singular values
    Vt: np.ndarray  # Item factors
    mean_rating: float
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predecir rating funcionalmente"""
        prediction = (
            self.U[user_idx, :] @ 
            np.diag(self.sigma) @ 
            self.Vt[:, item_idx]
        ) + self.mean_rating
        return np.clip(prediction, 1, 5)  # Rating range
    
    def recommend(self, user_idx: int, n: int = 10) -> List[int]:
        """Generar recomendaciones"""
        # Predecir todos los items
        predictions = [
            (item_idx, self.predict(user_idx, item_idx))
            for item_idx in range(self.Vt.shape[1])
        ]
        
        # Ordenar y tomar top-n
        top_items = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
        return [item_idx for item_idx, _ in top_items]

def train_svd(
    matrix: UserItemMatrix,
    n_factors: int = 50
) -> SVDModel:
    """Entrenar modelo SVD funcionalmente"""
    # Centrar datos
    mean_rating = matrix.data[matrix.data > 0].mean()
    centered_data = matrix.data.copy()
    centered_data[centered_data > 0] -= mean_rating
    
    # SVD
    U, sigma, Vt = svds(centered_data, k=n_factors)
    
    return SVDModel(U, sigma, Vt, mean_rating)
```

### 3. Content-Based Filtering Funcional
```python
from sklearn.feature_extraction.text import TfidfVectorizer

@dataclass(frozen=True)
class ContentBasedModel:
    """Modelo content-based inmutable"""
    item_features: np.ndarray
    item_ids: tuple
    
    def find_similar_items(
        self,
        item_id: int,
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """Encontrar items similares basado en contenido"""
        idx = self.item_ids.index(item_id)
        target_features = self.item_features[idx]
        
        similarities = [
            (iid, cosine_similarity(target_features, self.item_features[i]))
            for i, iid in enumerate(self.item_ids)
            if iid != item_id
        ]
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

def extract_features(
    items: pd.DataFrame,
    text_columns: List[str]
) -> np.ndarray:
    """Extraer features de items funcionalmente"""
    # Combinar columnas de texto
    combined_text = items[text_columns].apply(
        lambda row: ' '.join(row.values.astype(str)),
        axis=1
    )
    
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=100)
    features = vectorizer.fit_transform(combined_text)
    
    return features.toarray()

def create_content_model(
    items: pd.DataFrame,
    feature_columns: List[str]
) -> ContentBasedModel:
    """Crear modelo content-based funcionalmente"""
    features = extract_features(items, feature_columns)
    item_ids = tuple(items['id'].values)
    
    return ContentBasedModel(features, item_ids)
```

### 4. Sistema Híbrido Composable
```python
from toolz import compose

@dataclass(frozen=True)
class HybridRecommender:
    """Sistema de recomendación híbrido"""
    collaborative: SVDModel
    content_based: ContentBasedModel
    weights: Tuple[float, float]
    
    def recommend(
        self,
        user_id: int,
        n: int = 10,
        diversity_weight: float = 0.3
    ) -> List[Tuple[int, float]]:
        """Generar recomendaciones híbridas"""
        # Recomendaciones colaborativas
        collab_recs = self.collaborative.recommend(user_id, n * 2)
        collab_scores = {
            item_id: self.collaborative.predict(user_id, item_id)
            for item_id in collab_recs
        }
        
        # Recomendaciones content-based (para diversidad)
        # Basado en items que el usuario ya calificó alto
        user_top_items = get_user_top_items(user_id, k=5)
        content_recs = set()
        for item_id in user_top_items:
            similar = self.content_based.find_similar_items(item_id, k=5)
            content_recs.update(iid for iid, _ in similar)
        
        # Combinar scores
        w_collab, w_content = self.weights
        
        final_scores = []
        for item_id in set(collab_recs) | content_recs:
            collab_score = collab_scores.get(item_id, 0) * w_collab
            
            # Score de diversidad basado en contenido
            content_score = (
                sum(
                    sim for iid, sim in 
                    self.content_based.find_similar_items(item_id, 10)
                    if iid in user_top_items
                ) * w_content
            )
            
            total_score = collab_score + content_score * diversity_weight
            final_scores.append((item_id, total_score))
        
        return sorted(final_scores, key=lambda x: x[1], reverse=True)[:n]
```

### 5. Métricas de Evaluación Funcionales
```python
def precision_at_k(
    recommendations: List[int],
    relevant_items: Set[int],
    k: int = 10
) -> float:
    """Precision@K funcional"""
    top_k = set(recommendations[:k])
    hits = len(top_k & relevant_items)
    return hits / k if k > 0 else 0

def recall_at_k(
    recommendations: List[int],
    relevant_items: Set[int],
    k: int = 10
) -> float:
    """Recall@K funcional"""
    top_k = set(recommendations[:k])
    hits = len(top_k & relevant_items)
    return hits / len(relevant_items) if relevant_items else 0

def ndcg_at_k(
    recommendations: List[int],
    relevance_scores: Dict[int, float],
    k: int = 10
) -> float:
    """NDCG@K (Normalized Discounted Cumulative Gain)"""
    def dcg(scores: List[float]) -> float:
        return sum(
            (2**score - 1) / np.log2(idx + 2)
            for idx, score in enumerate(scores)
        )
    
    # DCG actual
    actual_scores = [
        relevance_scores.get(item_id, 0)
        for item_id in recommendations[:k]
    ]
    actual_dcg = dcg(actual_scores)
    
    # DCG ideal
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal_scores)
    
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0

def evaluate_model(
    model: HybridRecommender,
    test_set: pd.DataFrame,
    k: int = 10
) -> Dict[str, float]:
    """Evaluación completa del modelo"""
    metrics = {
        'precision': [],
        'recall': [],
        'ndcg': []
    }
    
    for user_id in test_set['user_id'].unique():
        recommendations = model.recommend(user_id, k)
        relevant = set(test_set[test_set['user_id'] == user_id]['item_id'])
        
        metrics['precision'].append(
            precision_at_k([r[0] for r in recommendations], relevant, k)
        )
        metrics['recall'].append(
            recall_at_k([r[0] for r in recommendations], relevant, k)
        )
        
        relevance_scores = dict(
            test_set[test_set['user_id'] == user_id][['item_id', 'rating']].values
        )
        metrics['ndcg'].append(
            ndcg_at_k([r[0] for r in recommendations], relevance_scores, k)
        )
    
    return {
        metric: np.mean(values)
        for metric, values in metrics.items()
    }
```

---

## 📊 Funcionalidades Implementadas

### Algoritmos
- ✅ User-based collaborative filtering
- ✅ Item-based collaborative filtering
- ✅ Matrix factorization (SVD, ALS)
- ✅ Content-based filtering
- ✅ Sistema híbrido

### Features
- ✅ Cold start handling
- ✅ Diversity optimization
- ✅ Serendipity scoring
- ✅ Explanation generation

### Evaluación
- ✅ Precision, Recall, F1
- ✅ NDCG, MAP
- ✅ Coverage, Diversity
- ✅ A/B testing framework

---

## 🧪 Testing

```bash
# Tests
pytest tests/ -v

# Tests con datasets reales
pytest tests/test_integration.py

# Benchmarks
pytest tests/test_performance.py
```

---

## 📈 Pipeline de Desarrollo

### Semana 1: Algoritmos Base (30 Oct - 5 Nov)
- Collaborative filtering
- Similarity metrics
- Base recommender

### Semana 2: Matrix Factorization (6 Nov - 12 Nov)
- SVD implementation
- Content-based filtering
- Hybrid system

### Semana 3: Evaluación y Optimización (13 Nov - 19 Nov)
- Métricas de evaluación
- A/B testing
- Performance tuning

---

## 💼 Componente de Emprendimiento

**Aplicación Real**: Motor de recomendaciones para e-commerce

**Propuesta de Valor**:
- Aumento del 30% en conversión
- Personalización en tiempo real
- API simple y escalable
- Explicabilidad de recomendaciones

**Casos de Uso**:
- E-commerce
- Streaming (música, video)
- News aggregation
- Social networks

---

## 📚 Referencias

-Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook (2nd ed.). Springer. 
 https://doi.org/10.1007/978-1-4899-7637-6
-Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. 
 Computer, 42(8), 30-37. https://doi.org/10.1109/MC.2009.263
-Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based collaborative filtering 
 recommendation algorithms. In Proceedings of the 10th International Conference on World Wide Web (pp. 285-295). ACM.

---

## 🏆 Criterios de Evaluación

- **Algoritmos (30%)**: Corrección, eficiencia
- **Sistema Híbrido (25%)**: Integración elegante
- **Evaluación (25%)**: Métricas completas
- **Performance (20%)**: Escalabilidad

---

## 👥 Autor

**Nombres**: 
David Isaac Avalos Díaz
Osmar Antonio Isais Virgen

**Email**: 
osmarisais57@gmail.com 
diegoalxdiaz@outlook.com  
davalos16@ucol.mx

**GitHub**: 
https://github.com/ninja-1-droid
https://github.com/Daxed64
https://github.com/osmar433


---

## 📄 Licencia

Proyecto académico - Universidad de Colima © 2025
