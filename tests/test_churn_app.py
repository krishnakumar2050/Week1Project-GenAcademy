from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app import retention_segment_profile
from churn_app.data import (
    CANDIDATE_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    prepare_training_frame,
    validate_training_data,
)
from churn_app.modeling import (
    evaluate_threshold,
    risk_band,
    train_and_evaluate,
)
from churn_app.scoring import prepare_batch, score_batch


def sample_data(rows: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(21)
    frame = pd.DataFrame(
        {
            "CreditScore": rng.integers(350, 851, rows),
            "Geography": rng.choice(["France", "Spain", "Germany"], rows),
            "Gender": rng.choice(["Female", "Male"], rows),
            "Age": rng.integers(18, 80, rows),
            "Tenure": rng.integers(0, 11, rows),
            "Balance": rng.uniform(0, 250_000, rows),
            "NumOfProducts": rng.choice([1, 2, 3], rows),
            "HasCrCard": rng.integers(0, 2, rows),
            "IsActiveMember": rng.integers(0, 2, rows),
            "EstimatedSalary": rng.uniform(0, 200_000, rows),
            "Exited": np.arange(rows) % 2,
            "CustomerId": np.arange(1_000_000, 1_000_000 + rows),
            "Surname": ["Example"] * rows,
        }
    )
    return frame


def test_validation_reports_missing_and_excludes_sensitive_source_columns():
    frame = sample_data()
    frame.loc[0, "Age"] = np.nan
    result = validate_training_data(frame)

    assert result.is_valid
    assert any("Age" in warning for warning in result.warnings)
    statuses = result.report.set_index("Column")["Model status"]
    assert statuses["CustomerId"] == "Excluded: identifier"
    assert statuses["Exited"] == "Target only"
    assert "CustomerId" not in CANDIDATE_FEATURES
    assert "Surname" not in CANDIDATE_FEATURES


def test_prepare_training_frame_reports_invalid_target_rows():
    frame = sample_data()
    frame.loc[0, "Exited"] = np.nan

    features, target, excluded = prepare_training_frame(frame)

    assert excluded == 1
    assert len(features) == len(target) == len(frame) - 1
    assert tuple(features.columns) == CANDIDATE_FEATURES


def test_training_selects_on_validation_and_returns_holdout_results():
    features, target, _ = prepare_training_frame(sample_data())
    result = train_and_evaluate(features, target)

    assert result.selected_name in ("Logistic Regression", "Random Forest")
    assert sum(result.split_sizes.values()) == len(features)
    assert result.split_sizes == {"Training": 180, "Validation": 60, "Test": 60}
    assert len(result.test_probabilities) == len(result.y_test)
    assert len(result.validation_probabilities) == len(result.y_validation)
    metrics = evaluate_threshold(result.y_test, result.test_probabilities, 0.5)
    assert 0 <= metrics["PR-AUC"] <= 1
    assert sum(metrics[key] for key in (
        "True negatives", "False positives", "False negatives", "True positives"
    )) == len(result.y_test)


def test_batch_validation_reports_invalid_rows_and_keeps_identifiers_out_of_features():
    frame = sample_data(3).drop(columns="Exited")
    frame["Age"] = frame["Age"].astype(object)
    frame.loc[1, "Age"] = "not-a-number"
    valid, issues, references = prepare_batch(frame)

    assert len(valid) == 2
    assert issues["Source row"].unique().tolist() == [2]
    assert "CustomerId" in references
    assert "CustomerId" not in valid


def test_batch_scoring_and_risk_bands():
    training = sample_data()
    features, target, _ = prepare_training_frame(training)
    result = train_and_evaluate(features, target)
    batch_features = features.iloc[:2].copy()
    scores = score_batch(result.selected_pipeline, batch_features, 0.5)

    assert len(scores) == 2
    assert scores["Churn probability"].between(0, 1).all()
    assert scores["Risk band"].isin(["Higher risk", "Review", "Lower risk"]).all()
    assert risk_band(0.7, 0.5) == "Higher risk"
    assert risk_band(0.3, 0.5) == "Review"
    assert risk_band(0.1, 0.5) == "Lower risk"
    with pytest.raises(ValueError):
        score_batch(result.selected_pipeline, batch_features, 1.0)


def test_batch_rejects_missing_feature_columns():
    frame = pd.DataFrame({feature: [1] for feature in NUMERIC_FEATURES})
    with pytest.raises(ValueError, match="Missing required scoring columns"):
        prepare_batch(frame)


def test_batch_rejects_numeric_values_outside_supported_ranges():
    frame = sample_data(2).drop(columns="Exited")
    frame["Age"] = frame["Age"].astype(object)
    frame.loc[0, "Age"] = 121

    valid, issues, _ = prepare_batch(frame)

    assert len(valid) == 1
    assert issues["Source row"].tolist() == [1]
    assert "outside the supported range" in issues["Issue"].iloc[0]


def test_retention_segment_profile_shows_denominators_and_overall_difference():
    frame = sample_data(100)
    profile = retention_segment_profile(frame, "IsActiveMember")

    assert set(profile["Segment"]) == {"Active", "Inactive"}
    assert profile["Customers"].sum() == len(frame)
    assert profile["Exited"].sum() == int(frame["Exited"].sum())
    assert profile["Churn_rate"].between(0, 1).all()
    assert np.isclose(
        np.average(profile["vs_overall_pp"], weights=profile["Customers"]),
        0,
        atol=1e-10,
    )


def test_retention_segment_profile_rejects_unapproved_factor():
    with pytest.raises(ValueError, match="Unsupported retention insight factor"):
        retention_segment_profile(sample_data(), "Surname")


def test_streamlit_app_starts_without_uploaded_data():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("app.py", default_timeout=30).run()

    assert not app.exception
    assert any(element.value == "Bank Customer Churn Intelligence" for element in app.title)
    assert len(app.get("file_uploader")) == 1
    assert any("Upload a training CSV above" in element.value for element in app.info)
