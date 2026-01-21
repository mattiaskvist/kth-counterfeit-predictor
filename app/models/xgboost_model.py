from __future__ import annotations

from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor

try:
    from xgboost import XGBClassifier

    _HAVE_XGBOOST = True
except ImportError:
    _HAVE_XGBOOST = False

from sklearn.ensemble import HistGradientBoostingClassifier


def build_model(random_state: int = 42) -> Pipeline:
    if _HAVE_XGBOOST:
        model = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        model = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.08,
            max_iter=300,
            random_state=random_state,
        )

    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", model),
        ]
    )
