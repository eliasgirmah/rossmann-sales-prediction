import unittest
import pandas as pd
from src.data_preprocessing import clean_data

class TestDataPreprocessing(unittest.TestCase):

    def test_clean_data(self):
        # Create mock data
        data = pd.DataFrame({'Sales': [500, None, 1500]})
        cleaned_data = clean_data(data)
        
        # Assert that missing values are filled
        self.assertEqual(cleaned_data['Sales'].isnull().sum(), 0)

if __name__ == "__main__":
    unittest.main()
