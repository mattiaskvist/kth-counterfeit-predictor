from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from app.models import MODEL_BUILDERS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "products_train.csv"


def _param_grid_for_model(model) -> dict:
    estimator = model.named_steps.get("model")
    if estimator is None:
        raise ValueError("Pipeline does not contain a 'model' step.")

    name = estimator.__class__.__name__
    if name == "RandomForestClassifier":
        return {
            "model__n_estimators": [5, 10, 17, 25, 50, 100, 150, 200, 250, 300, 350, 500, 700],
            "model__max_depth": [3, 4, 5, 6],
            "model__min_samples_leaf": [1, 2, 3, 4, 5],
            "model__min_samples_split": [1, 2, 3, 4, 5, 8],
        }
    if name == "GradientBoostingClassifier":
        return {
            "model__n_estimators": [5, 10, 17, 25, 50, 100, 150, 200, 250, 300, 350, 500, 700],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 4, 5, 6],
            "model__subsample": [0.7, 0.9],
        }
    if name == "XGBClassifier":
        return {
            "model__n_estimators": [5, 10, 17, 25, 50, 100, 150, 200, 250, 300, 350, 500, 700],
            "model__max_depth": [3, 4, 5, 6],
            "model__learning_rate": [0.03, 0.07],
            "model__subsample": [0.7, 0.9],
            "model__colsample_bytree": [0.7, 0.9],
        }

        # return {
        #     "model__n_estimators": [250],
        #     "model__max_depth": [3],
        #     "model__learning_rate": [0.05],
        #     "model__subsample": [0.8],
        #     "model__colsample_bytree": [0.7],
        # }
    if name == "HistGradientBoostingClassifier":
        return {
            "model__max_iter": [5, 10, 17, 25, 50, 100, 150, 200, 250, 300, 350, 500, 700],
            "model__max_depth": [3, 4, 5, 6],
            "model__learning_rate": [0.04, 0.08],
            "model__l2_regularization": [0.0, 1.0],
        }

    raise ValueError(f"No param grid configured for estimator: {name}")


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["label"])
    y = df["label"].astype(int)

    scorer = make_scorer(f1_score, pos_label=1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    best_overall = {
        "name": None,
        "score": -1.0,
        "params": None,
    }

    for name, builder in MODEL_BUILDERS.items():
        model = builder()
        param_grid = _param_grid_for_model(model)

        search = GridSearchCV(
            model,
            param_grid=param_grid,
            scoring=scorer,
            cv=cv,
            n_jobs=-1,
        )
        search.fit(X, y)

        print(f"\nModel: {name}")
        print(f"Best F1: {search.best_score_:.4f}")
        print(f"Best params: {search.best_params_}")

        if search.best_score_ > best_overall["score"]:
            best_overall = {
                "name": name,
                "score": search.best_score_,
                "params": search.best_params_,
            }

    print("\nBest overall")
    print(f"Model: {best_overall['name']}")
    print(f"Best F1: {best_overall['score']:.4f}")
    print(f"Best params: {best_overall['params']}")


if __name__ == "__main__":
    main()
