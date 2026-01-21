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
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_lambda=1.5,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        model = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.06,
            max_iter=200,
            random_state=random_state,
        )

    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", model),
        ]
    )
