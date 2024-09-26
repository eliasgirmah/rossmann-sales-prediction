# tests/test_feature_engineering.py

import pytest
import pandas as pd
from src.feature_engineering import preprocess_data, add_features

@pytest.fixture
def mock_data():
    data = {
        'Store': [1, 2, 3],
        'Date': ['2015-07-31', '2015-07-30', '2015-07-29'],
        'Sales': [5263, 5020, 4782],
        'Customers': [555, 546, 523],
        'CompetitionDistance': [100.0, None, 200.0],
        'Promo2SinceYear': [2010, None, 2011]
    }
    df = pd.DataFrame(data)
    return df

def test_preprocess_data(mock_data):
    df = preprocess_data(mock_data)
    assert df['CompetitionDistance'].isnull().sum() == 0
    assert df['Promo2SinceYear'].isnull().sum() == 0

def test_add_features(mock_data):
    df = add_features(mock_data)
    assert 'Year' in df.columns
    assert 'Month' in df.columns
    assert 'Day' in df.columns
    assert 'WeekOfYear' in df.columns
