import pandas as pd
from src.content_based import create_content_model

def sample_items():
    return pd.DataFrame({
        "title": ["Data Scientist", "Backend Dev", "Data Engineer"],
        "skills": ["python sql stats", "java spring", "python spark sql"]
    })

def test_create_content_model():
    model = create_content_model(sample_items(), ["title", "skills"])
    assert len(model.item_ids) == 3

def test_find_similar_items():
    model = create_content_model(sample_items(), ["title", "skills"])
    sims = model.find_similar_items(0, k=2)
    assert len(sims) == 2

def test_feature_vector_size():
    model = create_content_model(sample_items(), ["title", "skills"])
    assert model.features.shape[0] == 3
