from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .data import (
    CANDIDATE_FEATURES,
    CATEGORICAL_FEATURES,
    EXPECTED_RANGES,
    NUMERIC_FEATURES,
)
from .modeling import risk_band

REFERENCE_COLUMNS = ("CustomerId", "RowNumber")


def prepare_batch(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return valid model rows, rejected-row issues, and optional references."""
    missing = [column for column in CANDIDATE_FEATURES if column not in frame]
    if missing:
        raise ValueError("Missing required scoring columns: " + ", ".join(missing))

    features = frame.loc[:, CANDIDATE_FEATURES].copy()
    issues: list[dict[str, object]] = []
    for column in NUMERIC_FEATURES:
        converted = pd.to_numeric(features[column], errors="coerce")
        invalid = features[column].notna() & converted.isna()
        for idx in features.index[invalid]:
            issues.append({"Source row": _row_number(frame, idx), "Issue": f"{column} is not numeric."})
        missing_values = converted.isna() & ~invalid
        for idx in features.index[missing_values]:
            issues.append({"Source row": _row_number(frame, idx), "Issue": f"{column} is missing."})
        low, high = EXPECTED_RANGES.get(column, (None, None))
        out_of_range = converted.notna()
        if low is not None:
            out_of_range &= converted < low
        if high is not None:
            out_of_range |= converted.notna() & (converted > high)
        for idx in features.index[out_of_range]:
            issues.append(
                {
                    "Source row": _row_number(frame, idx),
                    "Issue": f"{column} is outside the supported range ({low}, {high}).",
                }
            )
        features[column] = converted
    for column in CATEGORICAL_FEATURES:
        converted = features[column].astype("string").str.strip()
        missing_values = converted.isna() | converted.eq("")
        for idx in features.index[missing_values]:
            issues.append({"Source row": _row_number(frame, idx), "Issue": f"{column} is missing."})
        features[column] = converted.replace("", pd.NA)

    references = frame.loc[
        :, [column for column in REFERENCE_COLUMNS if column in frame.columns]
    ].copy()
    references.insert(0, "Source row", [_row_number(frame, idx) for idx in frame.index])
    issue_frame = pd.DataFrame(issues, columns=["Source row", "Issue"])
    invalid_rows = set(issue_frame["Source row"]) if not issue_frame.empty else set()
    valid_mask = pd.Series(
        [_row_number(frame, idx) not in invalid_rows for idx in frame.index],
        index=frame.index,
    )
    return (
        features.loc[valid_mask].reset_index(drop=True),
        issue_frame,
        references.loc[valid_mask].reset_index(drop=True),
    )


def score_batch(
    pipeline: Pipeline,
    features: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1.")
    if features.empty:
        return pd.DataFrame(
            columns=["Churn probability", "Risk band", "Threshold"]
        )
    probabilities = pipeline.predict_proba(features.loc[:, CANDIDATE_FEATURES])[:, 1]
    return pd.DataFrame(
        {
            "Churn probability": probabilities,
            "Risk band": [risk_band(float(p), threshold) for p in probabilities],
            "Threshold": threshold,
        }
    )


def _row_number(frame: pd.DataFrame, index: object) -> int:
    return int(frame.index.get_loc(index)) + 1
