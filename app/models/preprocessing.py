from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ID_COLS = ["product_id", "seller_id"]
CATEGORICAL_COLS = ["category", "brand", "seller_country", "shipping_origin"]
NUMERIC_COLS = [
    "price",
    "listing_views",
    "sales_count",
    "saved_count",
    "certification_badges",
    "warranty_months",
]
BOOL_COLS = [
    "return_policy_clear",
    "bulk_discount_available",
    "unusual_payment_patterns",
    "ip_location_mismatch",
]
DATE_COL = "listing_date"
DATE_FEATURES = [
    "listing_year",
    "listing_month",
    "listing_day",
    "listing_dayofweek",
    "listing_days_since_start",
]

_BOOL_MAPPING = {
    True: 1,
    False: 0,
    "True": 1,
    "False": 0,
    "true": 1,
    "false": 0,
    "1": 1,
    "0": 0,
    1: 1,
    0: 0,
}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        date_col: str = DATE_COL,
        id_cols: Iterable[str] | None = None,
        categorical_cols: Iterable[str] | None = None,
        numeric_cols: Iterable[str] | None = None,
        bool_cols: Iterable[str] | None = None,
    ) -> None:
        self.date_col = date_col
        self.id_cols = list(id_cols) if id_cols is not None else []
        self.categorical_cols = list(categorical_cols) if categorical_cols is not None else []
        self.numeric_cols = list(numeric_cols) if numeric_cols is not None else []
        self.bool_cols = list(bool_cols) if bool_cols is not None else []
        self.reference_date_: pd.Timestamp | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        X = X.copy()
        self._ensure_columns(X, [self.date_col])
        dates = pd.to_datetime(X[self.date_col], errors="coerce")
        if dates is not None and dates.notna().any():
            self.reference_date_ = dates.min()
        else:
            self.reference_date_ = pd.Timestamp("1970-01-01")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.drop(columns=["label"], errors="ignore")
        self._ensure_columns(X, self.categorical_cols + self.numeric_cols + self.bool_cols + [self.date_col])

        X = X.drop(columns=self.id_cols, errors="ignore")

        for col in self.bool_cols:
            X[col] = X[col].map(_BOOL_MAPPING).fillna(0).astype(int)

        for col in self.numeric_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        if self.date_col in X.columns:
            dates = pd.to_datetime(X[self.date_col], errors="coerce")
            # Add simple calendar features for seasonal patterns.
            X["listing_year"] = dates.dt.year.fillna(0).astype(int)
            X["listing_month"] = dates.dt.month.fillna(0).astype(int)
            X["listing_day"] = dates.dt.day.fillna(0).astype(int)
            X["listing_dayofweek"] = dates.dt.dayofweek.fillna(0).astype(int)
            ref = self.reference_date_ or pd.Timestamp("1970-01-01")
            X["listing_days_since_start"] = (dates - ref).dt.days.fillna(0).astype(int)
            X = X.drop(columns=[self.date_col], errors="ignore")

        self._ensure_columns(X, DATE_FEATURES)

        return X

    @staticmethod
    def _ensure_columns(X: pd.DataFrame, columns: Iterable[str]) -> None:
        for col in columns:
            if col not in X.columns:
                X[col] = None


def build_preprocessor() -> Pipeline:
    numeric_features = NUMERIC_COLS + BOOL_COLS + DATE_FEATURES

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, CATEGORICAL_COLS),
            ("numeric", numeric_transformer, numeric_features),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            (
                "feature_engineering",
                FeatureEngineer(
                    date_col=DATE_COL,
                    id_cols=ID_COLS,
                    categorical_cols=CATEGORICAL_COLS,
                    numeric_cols=NUMERIC_COLS,
                    bool_cols=BOOL_COLS,
                ),
            ),
            ("columns", column_transformer),
        ]
    )
