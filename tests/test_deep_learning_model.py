import pytest
import numpy as np
from src.deep_learning_model import create_lstm_model, train_lstm_model, save_lstm_model, load_lstm_model

def test_create_lstm_model():
    input_shape = (10, 1)  # Example shape
    model = create_lstm_model(input_shape)
    assert model is not None
    assert len(model.layers) == 3  # Ensure the model has the expected number of layers

def test_train_lstm_model():
    # Mock feature and target data
    X = np.random.rand(100, 10, 1)
    y = np.random.rand(100)
    
    model = train_lstm_model(X, y, epochs=1, batch_size=8)
    assert model is not None
    assert hasattr(model, 'fit')  # Check if the model has the fit method

def test_save_and_load_lstm_model():
    # Mock feature and target data
    X = np.random.rand(100, 10, 1)
    y = np.random.rand(100)
    
    model = train_lstm_model(X, y, epochs=1, batch_size=8)
    save_lstm_model(model, path='models/test_lstm_model.h5')
    
    loaded_model = load_lstm_model(path='models/test_lstm_model.h5')
    assert loaded_model is not None
    assert hasattr(loaded_model, 'predict')  # Check if the loaded model has the predict method

