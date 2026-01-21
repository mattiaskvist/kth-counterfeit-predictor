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
                    random_state=random_state,
                ),
            ),
        ]
    )
