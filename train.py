import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

DATA_PATH = Path("data/loan_data_raw.csv")
MODEL_OUTPUT_PATH = Path("model.joblib")
METRICS_OUTPUT_PATH = Path("metrics.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
N_ITER = 30
MAX_AGE = 85

EDUCATION_ORDER = ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
CATEGORICAL_COLS = ["person_home_ownership", "loan_intent", "previous_loan_defaults_on_file"]
DROP_COLS = ["person_gender"]
TARGET_COL = "loan_status"

log = logging.getLogger("train")


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["person_age"] <= MAX_AGE].copy()
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.drop(columns=DROP_COLS)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].copy()
    return X, y


def build_pipeline() -> Pipeline:
    education_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OrdinalEncoder(categories=[EDUCATION_ORDER]),
    )
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )
    numeric_remainder = make_pipeline(SimpleImputer(strategy="median"))

    preprocessing = ColumnTransformer(
        [
            ("education", education_pipeline, ["person_education"]),
            ("cat", cat_pipeline, CATEGORICAL_COLS),
        ],
        remainder=numeric_remainder,
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("random_forest", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )


def tune(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
    param_distributions = {
        "random_forest__n_estimators": randint(100, 500),
        "random_forest__max_depth": randint(3, 20),
        "random_forest__min_samples_split": randint(2, 10),
        "random_forest__min_samples_leaf": randint(1, 5),
        "random_forest__max_features": randint(2, 10),
    }
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=N_ITER,
        cv=cv,
        scoring=scoring,
        refit="f1",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    y_pred = model.predict(X)
    return {
        "f1": float(f1_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred)),
    }


def cv_metrics(search: RandomizedSearchCV) -> dict:
    i = search.best_index_
    return {
        "f1": float(search.cv_results_["mean_test_f1"][i]),
        "precision": float(search.cv_results_["mean_test_precision"][i]),
        "recall": float(search.cv_results_["mean_test_recall"][i]),
    }


def get_feature_importances(model: Pipeline) -> pd.DataFrame:
    """Return a feature-importance table sorted descending."""
    rf = model.named_steps["random_forest"]
    names = model.named_steps["preprocessing"].get_feature_names_out()
    return (
        pd.DataFrame({"feature": names, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log.info("Loading data from %s", DATA_PATH)
    df = load_and_clean(DATA_PATH)
    log.info("Loaded %d rows, %d columns", df.shape[0], df.shape[1])

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    log.info("Split sizes: train=%d, test=%d", len(X_train), len(X_test))

    log.info("Tuning with RandomizedSearchCV (n_iter=%d, cv=%d)", N_ITER, CV_SPLITS)
    search = tune(build_pipeline(), X_train, y_train)
    log.info("Best CV F1: %.4f", search.best_score_)
    log.info("Best params: %s", search.best_params_)

    best_model = search.best_estimator_
    train_metrics = cv_metrics(search)
    test_metrics = evaluate(best_model, X_test, y_test)
    log.info("Train (CV) metrics: %s", train_metrics)
    log.info("Test metrics: %s", test_metrics)

    importances = get_feature_importances(best_model)

    joblib.dump(best_model, MODEL_OUTPUT_PATH, compress=3)
    log.info("Saved model to %s", MODEL_OUTPUT_PATH)

    metrics_payload = {
        "best_params": {k: int(v) if isinstance(v, (np.integer,)) else v
                        for k, v in search.best_params_.items()},
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "top_10_feature_importances": importances.head(10).to_dict(orient="records"),
        "sklearn_version": sklearn.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    METRICS_OUTPUT_PATH.write_text(json.dumps(metrics_payload, indent=2))
    log.info("Saved metrics to %s", METRICS_OUTPUT_PATH)


if __name__ == "__main__":
    main()
