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
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
