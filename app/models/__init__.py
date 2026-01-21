from __future__ import annotations

from typing import Callable, Dict

from sklearn.pipeline import Pipeline

from .gradient_boosting_model import build_model as build_gradient_boosting
from .random_forest_model import build_model as build_random_forest
from .xgboost_model import build_model as build_xgboost

MODEL_BUILDERS: Dict[str, Callable[[], Pipeline]] = {
    "random_forest": build_random_forest,
    "xgboost_like": build_xgboost,
    "gradient_boosting": build_gradient_boosting,
}
