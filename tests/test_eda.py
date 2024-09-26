# test_eda.py
import unittest
import pandas as pd
from src.eda import load_data, check_missing_values, handle_missing_values, feature_extraction, combine_data

class TestEDA(unittest.TestCase):

    def setUp(self):
        # Setting up sample data for testing
        self.store_data = {
            'Store': [1, 2, 3],
            'CompetitionDistance': [100.0, None, 200.0],
            'CompetitionOpenSinceMonth': [9, None, 12],
            'CompetitionOpenSinceYear': [2010, None, 2012],
            'Promo2SinceWeek': [1, None, 1],
            'Promo2SinceYear': [2011, None, 2013],
            'PromoInterval': ['Jan,Feb,Mar', None, 'Apr,May,Jun']
        }

        self.train_data = {
            'Store': [1, 2, 3],
            'Sales': [500, 600, 700],
            'Open': [1, 0, 1],
            'Promo': [1, 0, 1],
            'Date': ['2023-09-01', '2023-09-02', '2023-09-03']
        }

        self.test_data = {
            'Store': [1, 2, 3],
            'Open': [1, None, 1]
        }

        self.store = pd.DataFrame(self.store_data)
        self.train = pd.DataFrame(self.train_data)
        self.test = pd.DataFrame(self.test_data)

    def test_check_missing_values(self):
        # Test for correct missing value detection
        missing_values = check_missing_values(self.store)
        self.assertEqual(missing_values['CompetitionDistance'], 1)

    def test_handle_missing_values(self):
        # Test for handling missing values correctly
        store, test = handle_missing_values(self.store, self.test)
        self.assertEqual(store['CompetitionDistance'].isnull().sum(), 0)
        self.assertEqual(store['PromoInterval'].isnull().sum(), 0)
        self.assertEqual(test['Open'].isnull().sum(), 0)

    def test_feature_extraction(self):
        # Test feature extraction for date components
        train_with_features = feature_extraction(self.train)
        self.assertIn('year', train_with_features.columns)
        self.assertIn('month', train_with_features.columns)
        self.assertIn('day', train_with_features.columns)
        self.assertIn('weekday', train_with_features.columns)

    def test_combine_data(self):
        # Test data merging between train/test and store datasets
        combined_train, combined_test = combine_data(self.train, self.test, self.store)
        self.assertIn('CompetitionDistance', combined_train.columns)
        self.assertIn('CompetitionDistance', combined_test.columns)

if __name__ == '__main__':
    unittest.main()
