import unittest
from sklearn.metrics import recall_score
import joblib
import pandas as pd

class TestModel(unittest.TestCase):
    model = None
    input_data = None
    model_path = './artifacts/model.joblib'   # adjust if your model path differs
    data_path = './data.csv'
    target_column = 'target'

    def setUp(self):
        """Load model and data before running tests"""
        try:
            self.model = joblib.load(self.model_path)
            self.input_data = pd.read_csv(self.data_path)
        except Exception as e:
            print(f"Setup failed: {e}")
            self.model = None
            self.input_data = None

    def test_data_integrity_check(self):
        """Ensure required columns exist in the dataset"""
        self.assertIsNotNone(self.input_data, "Data not loaded properly")

        required_features = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rolling_avg_10",
            "volume_sum_10",
        ]
        features_present = all(col in self.input_data.columns for col in required_features)
        self.assertTrue(features_present, "Required features missing from data")

    def test_model_accuracy(self):
        """Model should achieve a minimum recall on stock prediction dataset"""
        self.assertIsNotNone(self.model, "Model not loaded properly")
        self.assertIsNotNone(self.input_data, "Data not loaded properly")

        df_features = self.input_data.drop(self.target_column, axis=1, errors='ignore')

        # Drop non-numeric columns (e.g., timestamp, stock_name)
        df_features = df_features.select_dtypes(include=['number'])

        target_labels = self.input_data[self.target_column]

        # Ensure target has more than one class before testing recall
        if len(set(target_labels)) < 2:
            self.skipTest("Target column has only one unique class; recall cannot be computed.")

        predictions = self.model.predict(df_features)
        retrieval_rate = recall_score(target_labels, predictions, average='macro')
        self.assertGreater(retrieval_rate, 0.85, f"Model recall too low: {retrieval_rate}")

if __name__ == '__main__':
    unittest.main()
