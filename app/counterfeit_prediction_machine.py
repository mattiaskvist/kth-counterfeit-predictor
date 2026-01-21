from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

from app.models import MODEL_BUILDERS

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CounterfeitPredictionMachine:
    """
    Detects counterfeit products using supervised ML classification.

    The model is trained on the provided training data and predicts one product at a time.
    """

    def __init__(self) -> None:
        self.train_df = pd.read_csv(PROJECT_ROOT / "data" / "products_train.csv")
        self.eval_df = pd.read_csv(PROJECT_ROOT / "data" / "products_eval.csv")
        self.model = None
        self.selected_model_name: str | None = None
        self.validation_score: float | None = None

    def _train(self) -> None:
        """Train and select the best model from the available pipelines."""
        features = self.train_df.drop(columns=["label"])
        labels = self.train_df["label"].astype(int)

        eval_features = self.eval_df.drop(columns=["label"])
        eval_labels = self.eval_df["label"].astype(int)

        best_name = None
        best_score = -1.0

        for name, builder in MODEL_BUILDERS.items():
            model = builder()
            model.fit(features, labels)
            preds = model.predict(eval_features)
            score = f1_score(eval_labels, preds)

            if score > best_score:
                best_score = score
                best_name = name

        if best_name is None:
            raise RuntimeError("No models are available for training.")

        self.model = MODEL_BUILDERS[best_name]()
        self.model.fit(features, labels)
        self.selected_model_name = best_name
        self.validation_score = best_score

    def predict(self, product: pd.Series) -> bool:
        """
        Predict whether a single product is counterfeit.

        Args:
            product: A single product row as a pandas Series (with or without the label column)

        Returns:
            True if the product is likely counterfeit, False if genuine
        """
        if self.model is None:
            self._train()

        if isinstance(product, pd.Series):
            features = product.to_frame().T
        elif isinstance(product, pd.DataFrame):
            features = product.copy()
        else:
            raise TypeError("product must be a pandas Series or DataFrame")

        if "label" in features.columns:
            features = features.drop(columns=["label"])

        pred = self.model.predict(features)[0]
        return bool(int(pred))
