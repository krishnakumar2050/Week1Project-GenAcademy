from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO

import pandas as pd


TARGET = "Exited"
IDENTIFIER_COLUMNS = ("RowNumber", "CustomerId", "Surname")
PENDING_REVIEW_FEATURES = (
    "Complain",
    "Satisfaction Score",
    "Card Type",
    "Point Earned",
)
NUMERIC_FEATURES = (
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
)
CATEGORICAL_FEATURES = ("Geography", "Gender")
CANDIDATE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_TRAINING_COLUMNS = CANDIDATE_FEATURES + (TARGET,)
NUMERIC_COLUMNS = NUMERIC_FEATURES + (
    "RowNumber",
    "CustomerId",
    "Exited",
    "Complain",
    "Satisfaction Score",
    "Point Earned",
)
EXPECTED_RANGES = {
    "CreditScore": (0, 1000),
    "Age": (0, 120),
    "Tenure": (0, 100),
    "Balance": (0, None),
    "NumOfProducts": (1, 20),
    "HasCrCard": (0, 1),
    "IsActiveMember": (0, 1),
    "EstimatedSalary": (0, None),
    "Exited": (0, 1),
    "Complain": (0, 1),
    "Satisfaction Score": (1, 5),
    "Point Earned": (0, None),
}


@dataclass(frozen=True)
class DataValidation:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    report: pd.DataFrame

    @property
    def is_valid(self) -> bool:
        return not self.errors


def read_csv(source: str | BinaryIO) -> pd.DataFrame:
    """Read a CSV with normalized headers and fail clearly on invalid input."""
    try:
        frame = pd.read_csv(source)
    except (UnicodeDecodeError, pd.errors.ParserError, OSError, ValueError) as exc:
        raise ValueError(f"Could not read the CSV file: {exc}") from exc

    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"CSV contains duplicate column names: {duplicates}")
    return frame


def normalize_training_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert known numeric columns without dropping malformed input rows."""
    normalized = frame.copy()
    for column in NUMERIC_COLUMNS:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    for column in CATEGORICAL_FEATURES + ("Card Type",):
        if column in normalized.columns:
            normalized[column] = normalized[column].astype("string").str.strip()
            normalized[column] = normalized[column].replace("", pd.NA)
    return normalized


def validate_training_data(frame: pd.DataFrame) -> DataValidation:
    errors: list[str] = []
    warnings: list[str] = []
    missing_columns = [c for c in REQUIRED_TRAINING_COLUMNS if c not in frame]
    if missing_columns:
        errors.append(
            "Missing required training columns: " + ", ".join(missing_columns)
        )

    normalized = normalize_training_data(frame)
    rows: list[dict[str, object]] = []
    for column in frame.columns:
        values = normalized[column]
        missing_count = int(values.isna().sum())
        invalid_count = 0
        if column in NUMERIC_COLUMNS:
            invalid_count = int(frame[column].notna().sum() - values.notna().sum())
            if invalid_count:
                warnings.append(
                    f"{column}: {invalid_count} non-numeric value(s) will be treated as missing."
                )
            if column in EXPECTED_RANGES:
                low, high = EXPECTED_RANGES[column]
                numeric = values.dropna()
                outside = pd.Series(False, index=numeric.index)
                if low is not None:
                    outside |= numeric < low
                if high is not None:
                    outside |= numeric > high
                if outside.any():
                    warnings.append(
                        f"{column}: {int(outside.sum())} value(s) fall outside the broad validation range."
                    )
        rows.append(
            {
                "Column": column,
                "Inferred type": str(frame[column].dtype),
                "Missing values": missing_count,
                "Distinct values": int(values.nunique(dropna=True)),
                "Model status": _model_status(column),
            }
        )

    if TARGET in normalized.columns:
        target = normalized[TARGET]
        unexpected = sorted(set(target.dropna().unique()) - {0, 1})
        if unexpected:
            errors.append(f"{TARGET} must contain only 0/1 values; found {unexpected}.")
        if target.isna().any():
            warnings.append(
                f"{int(target.isna().sum())} row(s) have a missing or invalid target and will be excluded from model training."
            )
        if target.dropna().nunique() < 2:
            errors.append("The target must contain both churn classes (0 and 1).")

    for column in REQUIRED_TRAINING_COLUMNS:
        if column in normalized.columns and normalized[column].isna().any():
            if column != TARGET:
                warnings.append(
                    f"{column}: {int(normalized[column].isna().sum())} missing value(s); model pipelines will impute these."
                )

    duplicates = int(frame.duplicated().sum())
    if duplicates:
        warnings.append(f"{duplicates} duplicate row(s) found; none were removed.")
    for identifier in ("CustomerId", "RowNumber"):
        if identifier in frame.columns:
            duplicated_ids = int(frame[identifier].duplicated(keep=False).sum())
            if duplicated_ids:
                warnings.append(
                    f"{identifier}: {duplicated_ids} row(s) have a duplicate identifier."
                )

    if len(frame) < 100:
        warnings.append(
            "Fewer than 100 rows are available; model evaluation may be unstable."
        )
    if TARGET in normalized and normalized[TARGET].notna().any():
        positive_rate = float(normalized[TARGET].mean())
        if positive_rate < 0.05 or positive_rate > 0.95:
            warnings.append(
                f"Churn prevalence is {positive_rate:.1%}; stratified splitting may be unreliable."
            )

    return DataValidation(tuple(dict.fromkeys(errors)), tuple(dict.fromkeys(warnings)), pd.DataFrame(rows))


def prepare_training_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, int]:
    """Return approved candidate features and target, reporting excluded rows."""
    normalized = normalize_training_data(frame)
    if TARGET not in normalized:
        raise ValueError(f"Required target column '{TARGET}' is missing.")
    usable = normalized[TARGET].isin([0, 1])
    excluded = int((~usable).sum())
    selected = normalized.loc[usable]
    features = selected.loc[:, CANDIDATE_FEATURES].copy()
    target = selected[TARGET].astype(int).copy()
    if target.nunique() < 2:
        raise ValueError("Training requires at least one record in each target class.")
    return features, target, excluded


def _model_status(column: str) -> str:
    if column == TARGET:
        return "Target only"
    if column in IDENTIFIER_COLUMNS:
        return "Excluded: identifier"
    if column in PENDING_REVIEW_FEATURES:
        return "Held out: timing/governance review"
    if column in CANDIDATE_FEATURES:
        return "Candidate: governance approval required"
    return "Excluded: not in approved candidate schema"
