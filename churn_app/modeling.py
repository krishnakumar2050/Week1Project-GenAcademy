from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CANDIDATE_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

RANDOM_SEED = 42


@dataclass
class TrainedModels:
    validation_results: pd.DataFrame
    selected_name: str
    selected_pipeline: Pipeline
    validation_probabilities: np.ndarray
    y_validation: pd.Series
    x_test: pd.DataFrame
    test_probabilities: np.ndarray
    y_test: pd.Series
    split_sizes: dict[str, int]


def _pipeline(model_name: str, seed: int) -> Pipeline:
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if model_name == "Logistic Regression":
        numeric_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(numeric_steps)
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )
    if model_name == "Logistic Regression":
        classifier = LogisticRegression(
            class_weight="balanced", max_iter=1500, random_state=seed
        )
    elif model_name == "Random Forest":
        classifier = RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])


def _metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = probabilities >= threshold
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "PR-AUC": average_precision_score(y_true, probabilities),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "Brier score": brier_score_loss(y_true, probabilities),
        "True negatives": int(matrix[0, 0]),
        "False positives": int(matrix[0, 1]),
        "False negatives": int(matrix[1, 0]),
        "True positives": int(matrix[1, 1]),
    }


def train_and_evaluate(
    frame: pd.DataFrame, target: pd.Series, seed: int = RANDOM_SEED
) -> TrainedModels:
    """Compare on validation, refit the selected model, and test only once."""
    if list(frame.columns) != list(CANDIDATE_FEATURES):
        frame = frame.loc[:, CANDIDATE_FEATURES]
    if len(frame) < 50:
        raise ValueError("At least 50 labeled records are required for a split.")
    if len(frame) != len(target):
        raise ValueError("Feature and target row counts do not match.")
    target = pd.Series(target.to_numpy(), index=frame.index, name="Exited")

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        frame,
        target,
        test_size=0.2,
        random_state=seed,
        stratify=target,
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.25,
        random_state=seed,
        stratify=y_train_val,
    )
    if min(y_train.value_counts().min(), y_validation.value_counts().min(), y_test.value_counts().min()) < 1:
        raise ValueError("Each split must contain both churn classes.")

    validation_rows = []
    candidates = {}
    for name in ("Logistic Regression", "Random Forest"):
        model = _pipeline(name, seed)
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_validation)[:, 1]
        metrics = _metrics(y_validation, probabilities, 0.5)
        validation_rows.append({"Model": name, **metrics})
        candidates[name] = model

    validation_results = pd.DataFrame(validation_rows).sort_values(
        "PR-AUC", ascending=False
    )
    selected_name = str(validation_results.iloc[0]["Model"])
    selected_validation_model = candidates[selected_name]
    validation_probabilities = selected_validation_model.predict_proba(
        x_validation
    )[:, 1]
    selected_pipeline = _pipeline(selected_name, seed)
    selected_pipeline.fit(x_train_val, y_train_val)
    test_probabilities = selected_pipeline.predict_proba(x_test)[:, 1]
    return TrainedModels(
        validation_results=validation_results.reset_index(drop=True),
        selected_name=selected_name,
        selected_pipeline=selected_pipeline,
        validation_probabilities=validation_probabilities,
        y_validation=y_validation.copy(),
        x_test=x_test.copy(),
        test_probabilities=test_probabilities,
        y_test=y_test.copy(),
        split_sizes={
            "Training": len(x_train),
            "Validation": len(x_validation),
            "Test": len(x_test),
        },
    )


def evaluate_threshold(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> dict:
    return _metrics(y_true, probabilities, threshold)


def threshold_curve(
    y_true: pd.Series, probabilities: np.ndarray
) -> pd.DataFrame:
    thresholds = np.linspace(0.05, 0.95, 37)
    return pd.DataFrame(
        [
            {
                "Threshold": threshold,
                "Precision": precision_score(
                    y_true, probabilities >= threshold, zero_division=0
                ),
                "Recall": recall_score(
                    y_true, probabilities >= threshold, zero_division=0
                ),
            }
            for threshold in thresholds
        ]
    )


def risk_band(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "Higher risk"
    if probability >= threshold * 0.6:
        return "Review"
    return "Lower risk"
