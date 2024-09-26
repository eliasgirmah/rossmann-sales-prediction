# tests/test_traditional_model.py

import pytest
import numpy as np
from src.traditional_model import train_traditional_model, save_traditional_model, load_traditional_model

def test_train_traditional_model():
    # Mock feature and target data
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    
    model = train_traditional_model(X, y)
    assert model is not None

def test_save_and_load_traditional_model():
    # Mock feature and target data
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    
    model = train_traditional_model(X, y)
    save_traditional_model(model, path='models/test_traditional_model.pkl')
    
    loaded_model = load_traditional_model(path='models/test_traditional_model.pkl')
    assert loaded_model is not None
