"""
Submit the results of your counterfeit prediction machine to the scoreboard!

Run with:
    python submit_results.py
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent

import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score

from app.counterfeit_prediction_machine import CounterfeitPredictionMachine


# ------- DONT CHANGE THESE VARIABLES -------
API_BASE = "https://score-board-737666003009.europe-north2.run.app"
# Maximum allowed average prediction time (in seconds)
# If exceeded, the script will abort
MAX_AVG_PREDICTION_TIME = 0.4
CHECK_INTERVAL = 10  # Check speed every N predictions
# ------- DONT CHANGE THESE VARIABLES -------


def load_data() -> pd.DataFrame:
    """Load the data."""
    return pd.read_csv(ROOT_DIR / "data" / "products_eval.csv")


def prepare_counterfeit_prediction_machine() -> CounterfeitPredictionMachine:
    """Prepare the counterfeit prediction machine.
    NOTE: You can change this function if your implementation requires it.
    """
    machine = CounterfeitPredictionMachine()
    machine._train()
    return machine


def submit_to_scoreboard(team_name: str, precision: float, recall: float, predict_time: float) -> int:
    """Submit results to the scoreboard."""
    print("\nSubmitting to scoreboard...")

    try:
        response = requests.post(
            f"{API_BASE}/submit_record",
            json={
                "team_name": team_name,
                "precision": precision,
                "recall": recall,
                "time": predict_time,
            },
            timeout=30,
        )
        print(f"Response: {response.status_code} - {response.text}")

        if response.status_code >= 400:
            print("Submission failed!")
            return 1

        print("Submission successful!")
        return 0

    except requests.RequestException as e:
        print(f"Error submitting results: {e}")
        return 1

 
def main() -> int:
    # Load environment variables
    load_dotenv(ROOT_DIR / ".env")

    # Get team name from environment variables
    team_name = os.getenv("TEAM_NAME")
    if not team_name:
        raise ValueError("TEAM_NAME environment variable is not set")

    print(f"Running predictions for team: {team_name}")

    # Get the counterfeit prediction machine
    print("Preparing counterfeit prediction machine...")
    prepare_start = time.perf_counter()
    machine = prepare_counterfeit_prediction_machine()
    prepare_time = time.perf_counter() - prepare_start
    print(f"Time taken to prepare counterfeit prediction machine: {prepare_time:.4f}s")

    # Load data
    eval_df = load_data()
    eval_features_df = eval_df.drop(columns=["label"])
    eval_labels_list = eval_df["label"].tolist()

    # Run predictions
    total_samples = len(eval_features_df)
    predictions = []
    predict_start = time.perf_counter()
    for i, (_, row) in enumerate(eval_features_df.iterrows()):
        pred = machine.predict(row)
        predictions.append(pred)

        # Check every N predictions if we're on track to exceed time limit
        if (i + 1) % CHECK_INTERVAL == 0:
            elapsed_time = time.perf_counter() - predict_start
            predictions_made = i + 1
            avg_time = elapsed_time / predictions_made
            estimated_total = avg_time * total_samples

            if avg_time > MAX_AVG_PREDICTION_TIME:
                print(f"\nError: Average prediction time ({avg_time:.3f}s) exceeds limit ({MAX_AVG_PREDICTION_TIME}s)")
                print(f"Checked after {predictions_made} predictions")
                print(f"Estimated total time: {estimated_total:.1f}s")
                print(
                    "Aborting to save time. Please optimize your solution. Maybe there is a better way to do things?..."
                )
                return 1

    predict_time = time.perf_counter() - predict_start

    # Validate predictions
    for i, pred in enumerate(predictions):
        if not isinstance(pred, bool):
            print(f"Error: Invalid prediction at index {i}: {pred} (must be True or False)")
            return 1

    # Convert bool predictions to int for sklearn metrics (True=1, False=0)
    predictions_int = [1 if p else 0 for p in predictions]

    # Calculate metrics
    accuracy = accuracy_score(eval_labels_list, predictions_int)
    precision = precision_score(eval_labels_list, predictions_int, zero_division=0)
    recall = recall_score(eval_labels_list, predictions_int, zero_division=0)

    print("\n" + "=" * 40)
    print(f"Team:       {team_name}")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"Time:       {predict_time:.4f}s")
    print("=" * 40)

    # Submit to API
    #return submit_to_scoreboard(team_name, precision, recall, predict_time)


if __name__ == "__main__":
    sys.exit(main())
