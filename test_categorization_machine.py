"""
Unit tests for you counterfeit prediction machine!

Run with:
    python test_categorization_machine.py
"""

import time
import unittest
import warnings
from pathlib import Path
import pandas as pd
from app.counterfeit_prediction_machine import CounterfeitPredictionMachine


# Threshold for average prediction time (in seconds)
# In the real world we wound not want to wait for hours to detect if we have counterfeit products, right?
PREDICTION_TIME_THRESHOLD = 0.4


# Project root directory
ROOT_DIR = Path(__file__).resolve().parent


class TestCounterfeitPredictionMachine(unittest.TestCase):
    """Tests that your model must pass to be valid."""

    def setUp(self) -> None:
        train_path = ROOT_DIR / "data" / "products_train.csv"
        test_path = ROOT_DIR / "data" / "products_test.csv"

        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.machine = CounterfeitPredictionMachine()
        # Train the model once for all tests
        print("Training...")
        self.machine._train()

    def test_train_runs_without_error(self) -> None:
        """Training should complete without raising exceptions."""
        self.machine._train()

    def test_predict_returns_bool(self) -> None:
        """Prediction must return a boolean (True or False)."""
        single_row = self.test_df.drop(columns=["label"]).iloc[0]
        pred = self.machine.predict(single_row)
        self.assertIsInstance(pred, bool, f"Expected bool, got {type(pred)}")
        self.assertIn(pred, [True, False], f"Expected True or False, got {pred}")

    def test_predict_on_single_row(self) -> None:
        """Model should handle predicting a single row."""
        single_row = self.test_df.drop(columns=["label"]).iloc[0]
        pred = self.machine.predict(single_row)
        self.assertIn(pred, [True, False])

    def test_that_predict_works_on_multiple_rows(self) -> None:
        """Test predictions on counterfeit and genuine products."""
        counterfeit_df = self.test_df[self.test_df["label"] == 1].head(3)
        genuine_df = self.test_df[self.test_df["label"] == 0].head(3)

        incorrect_predictions = 0
        total_predictions = 0

        print("Prediction in multiple...")
        for _, row in counterfeit_df.iterrows():
            pred = self.machine.predict(row)
            self.assertIsInstance(pred, bool)
            total_predictions += 1
            if pred != True:
                incorrect_predictions += 1
                warnings.warn(f"Incorrect prediction: Expected True (counterfeit), got {pred}")

        for _, row in genuine_df.iterrows():
            pred = self.machine.predict(row)
            self.assertIsInstance(pred, bool)
            total_predictions += 1
            if pred != False:
                incorrect_predictions += 1
                warnings.warn(f"Incorrect prediction: Expected False (genuine), got {pred}")

        print(f"Accuracy: {total_predictions - incorrect_predictions}/{total_predictions} predictions correct")

        if incorrect_predictions > 0:
            warnings.warn(
                f"Accuracy: {total_predictions - incorrect_predictions}/{total_predictions} predictions correct"
            )

    def test_prediction_speed(self) -> None:
        """Test prediction speed on 10 products. Lets make sure we are not too slow!"""
        test_features = self.test_df.drop(columns=["label"]).head(10)
        num_products = len(test_features)

        start_time = time.perf_counter()
        for _, row in test_features.iterrows():
            pred = self.machine.predict(row)
            self.assertIsInstance(pred, bool)
        total_time = time.perf_counter() - start_time

        avg_time = total_time / num_products

        # Always print the timing info
        print(f"\nPrediction speed: {total_time:.2f}s total, {avg_time:.2f}s avg per prediction")

        # Warn if too slow
        if avg_time > PREDICTION_TIME_THRESHOLD:
            warnings.warn(
                f"Slow predictions: {avg_time:.2f}s average per prediction "
                f"(threshold: {PREDICTION_TIME_THRESHOLD}s)"
            )


if __name__ == "__main__":
    unittest.main()
