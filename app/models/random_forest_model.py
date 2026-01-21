from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor


def build_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=50,
                    max_depth=4,
                    min_samples_leaf=2,
                    min_samples_split=8,
                    class_weight={0: 1.0, 1: 3.0},
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
