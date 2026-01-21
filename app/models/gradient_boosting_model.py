from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor


def build_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=150,
                    max_depth=3,
                    subsample=0.8,
                    learning_rate=0.05,
                    random_state=random_state,
                ),
            ),
        ]
    )
