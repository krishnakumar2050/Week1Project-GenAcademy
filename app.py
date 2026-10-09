from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn import __version__ as sklearn_version
from sklearn.calibration import calibration_curve

from churn_app.data import (
    CANDIDATE_FEATURES,
    CATEGORICAL_FEATURES,
    EXPECTED_RANGES,
    IDENTIFIER_COLUMNS,
    NUMERIC_FEATURES,
    PENDING_REVIEW_FEATURES,
    TARGET,
    normalize_training_data,
    prepare_training_frame,
    read_csv,
    validate_training_data,
)
from churn_app.modeling import (
    evaluate_threshold,
    risk_band,
    threshold_curve,
    train_and_evaluate,
)
from churn_app.scoring import prepare_batch, score_batch


st.set_page_config(page_title="Bank Churn Intelligence", page_icon="📊", layout="wide")

DISCLAIMER = (
    "Decision-support prototype only. Scores are estimates, not certainty or "
    "causal explanations. Do not use them for automated customer decisions. "
    "Obtain bank governance approval before operational use."
)


def main() -> None:
    st.title("Bank Customer Churn Intelligence")
    st.caption(
        "Explore historical churn, evaluate candidate models, and score future "
        "customers from an approved feature schema."
    )
    st.warning(DISCLAIMER)

    with st.sidebar:
        st.header("Workspace")
        page = st.radio(
            "Navigate",
            (
                "Overview",
                "Data Explorer",
                "Retention Insights",
                "Model Evaluation",
                "Score Customers",
                "Model Card",
            ),
        )
        st.divider()
        st.caption("Training and prediction data stay in memory for this session.")

    st.subheader("Training dataset")
    upload_col, source_col = st.columns([2, 1])
    with upload_col:
        uploaded = st.file_uploader(
            "Upload a training CSV",
            type=["csv"],
            key="training_dataset_upload",
            help=(
                "Choose the bank churn training dataset from your device. "
                "The file is processed in this session and is not saved by the app."
            ),
        )
    with source_col:
        st.caption("CSV only · Maximum upload size: 200 MB")
        if uploaded is None and os.environ.get("CHURN_DATA_PATH"):
            st.info("Using the locally configured training dataset.")
        elif uploaded is not None:
            st.success(f"Selected: {uploaded.name}")

    data, data_source, load_error = load_data(uploaded)
    if load_error:
        st.error(load_error)
        st.info(
            "Upload a compatible training CSV above, or set "
            "CHURN_DATA_PATH to a local CSV path."
        )
        return
    if data is None:
        st.info("Upload a training CSV above to load the dashboard.")
        return

    validation = validate_training_data(data)
    normalized = normalize_training_data(data)
    if validation.errors:
        for error in validation.errors:
            st.error(error)
        st.stop()
    if validation.warnings:
        with st.expander(f"Data validation warnings ({len(validation.warnings)})"):
            for warning in validation.warnings:
                st.warning(warning)

    st.session_state["training_data"] = normalized
    st.session_state["validation_report"] = validation
    st.session_state["data_source"] = data_source
    if page == "Overview":
        overview(normalized)
    elif page == "Data Explorer":
        data_explorer(normalized, validation)
    elif page == "Retention Insights":
        retention_insights(normalized)
    elif page == "Model Evaluation":
        model_evaluation(normalized)
    elif page == "Score Customers":
        customer_scoring(normalized)
    else:
        model_card(normalized)


def load_data(uploaded) -> tuple[pd.DataFrame | None, str, str | None]:
    if uploaded is not None:
        try:
            return read_csv(uploaded), uploaded.name, None
        except ValueError as exc:
            return None, uploaded.name, str(exc)
    path = os.environ.get("CHURN_DATA_PATH")
    if path:
        try:
            return read_csv(path), "Local file (CHURN_DATA_PATH)", None
        except ValueError as exc:
            return None, "Local file (CHURN_DATA_PATH)", str(exc)
    return None, "", None


def overview(data: pd.DataFrame) -> None:
    st.header("Executive Overview")
    target = data[TARGET]
    churn_rate = float(target.mean())
    latest = st.session_state.get("model_run")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers in training data", f"{len(data):,}")
    c2.metric("Historical churn rate", f"{churn_rate:.1%}")
    c3.metric(
        "Data validation",
        "Passed",
        delta=f"{data.isna().sum().sum():,} missing cell(s)",
        delta_color="off",
    )
    c4.metric(
        "Selected model",
        latest["selected_name"] if latest else "Not trained",
    )
    if latest:
        metric = evaluate_threshold(
            latest["y_test"], latest["test_probabilities"], 0.5
        )
        st.subheader("Held-out test snapshot")
        st.caption(
            "Final model evaluated once on a stratified random holdout; this is "
            "not a time-forward estimate of future performance."
        )
        cols = st.columns(4)
        cols[0].metric("PR-AUC", f"{metric['PR-AUC']:.3f}")
        cols[1].metric("Recall", f"{metric['Recall']:.3f}")
        cols[2].metric("Precision", f"{metric['Precision']:.3f}")
        cols[3].metric("High-risk flagged", f"{metric['False positives'] + metric['True positives']:,}")
    else:
        st.info("Train and evaluate models in **Model Evaluation** to activate scoring.")

    counts = target.value_counts().rename(index={0: "Stayed", 1: "Exited"}).rename_axis("Outcome").reset_index(name="Customers")
    fig = px.bar(counts, x="Outcome", y="Customers", color="Outcome", title="Training target balance")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "**Quick start:** inspect the data, review governance assumptions, then "
        "train candidate models before scoring customers."
    )


def data_explorer(data: pd.DataFrame, validation) -> None:
    st.header("Data Quality and Explorer")
    st.caption(f"Source: {st.session_state.get('data_source', 'Uploaded file')}")
    st.subheader("Schema and validation")
    st.dataframe(validation.report, use_container_width=True, hide_index=True)
    report_csv = validation.report.to_csv(index=False).encode("utf-8")
    st.download_button("Download schema report", report_csv, "data_schema_report.csv", "text/csv")

    normalized = normalize_training_data(data)
    st.subheader("Interactive exploration")
    left, right = st.columns(2)
    geographies = sorted(normalized["Geography"].dropna().unique().tolist())
    selected_geo = left.multiselect("Geography", geographies, default=geographies)
    ages = normalized["Age"].dropna()
    if ages.empty:
        st.info("No usable age values are available for age filtering.")
        min_age, max_age = 0.0, 120.0
    else:
        min_age, max_age = float(ages.min()), float(ages.max())
    age_range = right.slider(
        "Age range",
        min_value=int(min_age),
        max_value=max(int(min_age) + 1, int(max_age)),
        value=(int(min_age), int(max_age)),
    )
    lower, upper = st.columns(2)
    tenure_values = sorted(normalized["Tenure"].dropna().unique().tolist())
    tenure = lower.multiselect("Tenure (years)", tenure_values, default=tenure_values)
    products = sorted(normalized["NumOfProducts"].dropna().unique().tolist())
    product_count = upper.multiselect("Number of products", products, default=products)

    filtered = normalized[
        normalized["Geography"].isin(selected_geo)
        & normalized["Age"].between(*age_range, inclusive="both")
        & normalized["Tenure"].isin(tenure)
        & normalized["NumOfProducts"].isin(product_count)
    ]
    label = {0: "Stayed", 1: "Exited"}
    st.metric("Customers matching filters", f"{len(filtered):,}")
    st.caption("Group rates are observational associations, not evidence of cause.")
    if filtered.empty:
        st.warning("No records match the selected filters.")
        return

    numeric_views, category_views = st.tabs(["Churn rates", "Distributions"])
    with numeric_views:
        dimensions = ["Geography", "Gender", "IsActiveMember", "NumOfProducts", "Tenure"]
        dimension = st.selectbox("Compare churn by", dimensions)
        grouped = (
            filtered.groupby(dimension, dropna=False)[TARGET]
            .agg(Customers="size", Churn_rate="mean")
            .reset_index()
        )
        grouped["Churn rate"] = grouped["Churn_rate"].map(lambda value: f"{value:.1%}")
        grouped["Segment"] = grouped[dimension].map(
            lambda value: {0: "Inactive", 1: "Active"}.get(value, str(value))
        )
        fig = px.bar(
            grouped,
            x="Segment",
            y="Churn_rate",
            hover_data=["Customers", "Churn rate"],
            labels={"Churn_rate": "Churn rate"},
            title=f"Churn rate by {dimension} (denominators shown on hover)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(grouped[[dimension, "Customers", "Churn rate"]], hide_index=True, use_container_width=True)
    with category_views:
        numeric_column = st.selectbox(
            "Numeric distribution",
            ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"],
        )
        fig = px.histogram(
            filtered,
            x=numeric_column,
            color=TARGET,
            barmode="overlay",
            opacity=0.65,
            labels={TARGET: "Exited"},
        )
        st.plotly_chart(fig, use_container_width=True)
        outcome_counts = filtered[TARGET].value_counts().rename(index=label).rename_axis("Outcome").reset_index(name="Customers")
        st.dataframe(outcome_counts, hide_index=True, use_container_width=True)


RETENTION_FACTORS = {
    "Engagement status": "IsActiveMember",
    "Number of products": "NumOfProducts",
    "Tenure band": "Tenure",
    "Age band (sensitive; review only)": "Age",
    "Geography (review only)": "Geography",
    "Balance band (review only)": "Balance",
    "Complaint recorded (timing/leakage warning)": "Complain",
    "Satisfaction score (timing review)": "Satisfaction Score",
}


def retention_segment_profile(data: pd.DataFrame, factor: str) -> pd.DataFrame:
    """Summarize observed churn by a dataset feature, with denominators."""
    if factor not in RETENTION_FACTORS.values():
        raise ValueError(f"Unsupported retention insight factor: {factor}")
    if factor not in data or TARGET not in data:
        raise ValueError(f"Dataset must include '{factor}' and '{TARGET}'.")

    working = data[[factor, TARGET]].copy()
    if factor == "Age":
        working["Segment"] = pd.cut(
            working[factor],
            bins=[-np.inf, 29, 39, 49, 59, np.inf],
            labels=["Under 30", "30–39", "40–49", "50–59", "60+"],
        )
    elif factor == "Tenure":
        working["Segment"] = pd.cut(
            working[factor],
            bins=[-np.inf, 1, 4, 7, np.inf],
            labels=["0–1 years", "2–4 years", "5–7 years", "8+ years"],
        )
    elif factor == "Balance":
        working["Segment"] = pd.cut(
            working[factor],
            bins=[-np.inf, 0, 50_000, 100_000, 150_000, np.inf],
            labels=["Zero", "1–50k", "50–100k", "100–150k", "150k+"],
        )
    elif factor == "IsActiveMember":
        working["Segment"] = working[factor].map({0: "Inactive", 1: "Active"})
    elif factor == "HasCrCard":
        working["Segment"] = working[factor].map({0: "No card", 1: "Has card"})
    elif factor == "Complain":
        working["Segment"] = working[factor].map(
            {0: "No complaint recorded", 1: "Complaint recorded"}
        )
    elif factor == "NumOfProducts":
        working["Segment"] = working[factor].map(
            lambda value: f"{int(value)} product" if pd.notna(value) else "Unknown"
        )
    elif factor == "Satisfaction Score":
        working["Segment"] = working[factor].map(
            lambda value: f"Score {int(value)}" if pd.notna(value) else "Unknown"
        )
    else:
        working["Segment"] = working[factor].fillna("Unknown").astype(str)

    valid = working.dropna(subset=["Segment", TARGET])
    profile = (
        valid.groupby("Segment", observed=True)[TARGET]
        .agg(Customers="size", Exited="sum", Churn_rate="mean")
        .reset_index()
    )
    profile["Churn_rate_pct"] = profile["Churn_rate"] * 100
    profile["Exited"] = profile["Exited"].astype(int)
    profile["Segment"] = profile["Segment"].astype(str)
    overall_rate = float(valid[TARGET].mean()) if not valid.empty else 0.0
    profile["vs_overall_pp"] = (profile["Churn_rate"] - overall_rate) * 100
    return profile.sort_values("Churn_rate", ascending=False).reset_index(drop=True)


def retention_insights(data: pd.DataFrame) -> None:
    st.header("Retention Insights")
    st.write(
        "Compare historical churn patterns to help teams form and test "
        "customer-care campaign hypotheses."
    )
    st.warning(
        "These are retrospective group-level associations, not causes or "
        "individual churn predictions. Do not treat a segment as a customer "
        "target list. Validate signals with current, point-in-time data and "
        "human review before contacting customers."
    )
    overall_rate = float(data[TARGET].mean())
    metric_cols = st.columns(3)
    metric_cols[0].metric("Historical churn rate", f"{overall_rate:.1%}")
    metric_cols[1].metric("Customers represented", f"{len(data):,}")
    metric_cols[2].metric("Observed exits", f"{int(data[TARGET].sum()):,}")

    visual_factors = [
        "Engagement status",
        "Number of products",
        "Tenure band",
        "Age band (sensitive; review only)",
        "Geography (review only)",
        "Balance band (review only)",
    ]
    selected_label = st.selectbox(
        "Explore historical churn by factor",
        visual_factors,
        help="Age, geography and balance are shown for analysis only, not as campaign eligibility criteria.",
    )
    factor = RETENTION_FACTORS[selected_label]
    profile = retention_segment_profile(data, factor)
    minimum_size = st.number_input(
        "Minimum customers per segment to display",
        min_value=1,
        max_value=max(1, len(data)),
        value=min(100, max(1, len(data))),
        step=10,
        help="Small segment rates are volatile; increase this threshold for more stable views.",
    )
    visible = profile.loc[profile["Customers"] >= minimum_size].copy()
    if visible.empty:
        st.info("No segments meet the selected minimum group size.")
    else:
        if (visible["Customers"] < 300).any():
            st.info(
                "Some displayed segments have fewer than 300 customers. "
                "Treat those rates as less stable and avoid interpreting them "
                "without additional evidence."
            )
        visible["Rate label"] = visible["Churn_rate"].map(lambda rate: f"{rate:.1%}")
        st.caption(
            f"Overall dataset churn is {overall_rate:.1%}. Hover over the bars "
            "for segment size, observed exits and rate difference."
        )
        rate_chart = px.bar(
            visible.sort_values("Churn_rate"),
            x="Churn_rate_pct",
            y="Segment",
            orientation="h",
            color="Churn_rate_pct",
            color_continuous_scale="OrRd",
            hover_data={
                "Customers": True,
                "Exited": True,
                "Rate label": True,
                "vs_overall_pp": ":.1f",
                "Churn_rate_pct": False,
            },
            labels={
                "Churn_rate_pct": "Historical churn rate (%)",
                "vs_overall_pp": "Difference vs overall (pp)",
            },
            title=f"Historical churn rate by {selected_label}",
        )
        rate_chart.add_vline(
            x=overall_rate * 100,
            line_dash="dash",
            line_color="#344054",
            annotation_text=f"Overall {overall_rate:.1%}",
        )
        volume_chart = px.bar(
            visible.sort_values("Exited"),
            x="Exited",
            y="Segment",
            orientation="h",
            color="Exited",
            color_continuous_scale="Blues",
            hover_data=["Customers", "Rate label"],
            labels={"Exited": "Historical exits"},
            title="Number of observed exits by segment",
        )
        rate_col, volume_col = st.columns(2)
        rate_col.plotly_chart(rate_chart, use_container_width=True)
        volume_col.plotly_chart(volume_chart, use_container_width=True)
        table = visible[
            ["Segment", "Customers", "Exited", "Rate label", "vs_overall_pp"]
        ].rename(
            columns={
                "Rate label": "Churn rate",
                "vs_overall_pp": "Difference vs overall (percentage points)",
            }
        )
        st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("Campaign-planning hypotheses")
    st.caption(
        "Use these group patterns to design small, measurable and respectful "
        "service experiments—not to assume an individual will leave."
    )
    campaign_profiles = []
    suggested_actions = {
        "IsActiveMember": (
            "Consider a voluntary engagement check-in; ask what support or "
            "service would be useful."
        ),
        "NumOfProducts": (
            "Review product fit and service needs with the customer; do not "
            "assume more products are the right retention action."
        ),
        "Tenure": (
            "Consider a lifecycle check-in and ask whether onboarding or "
            "ongoing support needs remain unresolved."
        ),
    }
    for campaign_factor, action in suggested_actions.items():
        segment_profile = retention_segment_profile(data, campaign_factor)
        segment_profile.insert(0, "Factor", campaign_factor)
        segment_profile["Possible team hypothesis (not a causal claim)"] = action
        campaign_profiles.append(segment_profile)
    campaign_table = pd.concat(campaign_profiles, ignore_index=True)
    campaign_table = campaign_table.loc[
        campaign_table["Customers"] >= minimum_size
    ].sort_values(["Churn_rate", "Customers"], ascending=[False, False])
    campaign_table["Historical churn rate"] = campaign_table["Churn_rate"].map(
        lambda rate: f"{rate:.1%}"
    )
    campaign_table["Difference vs overall"] = campaign_table["vs_overall_pp"].map(
        lambda points: f"{points:+.1f} pp"
    )
    st.dataframe(
        campaign_table[
            [
                "Factor",
                "Segment",
                "Customers",
                "Exited",
                "Historical churn rate",
                "Difference vs overall",
                "Possible team hypothesis (not a causal claim)",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Signals that need timing or governance review"):
        st.warning(
            "Complaint and satisfaction fields are deliberately excluded from "
            "the campaign hypotheses below. In this dataset, a recorded "
            "complaint is almost a direct match for Exited; it may be "
            "post-outcome or leakage and must not be used as a campaign trigger "
            "until its timing and definition are verified."
        )
        review_factor = st.selectbox(
            "Inspect a review-only signal",
            ["Complain", "Satisfaction Score"],
        )
        review_profile = retention_segment_profile(data, review_factor)
        review_profile["Churn rate"] = review_profile["Churn_rate"].map(
            lambda rate: f"{rate:.1%}"
        )
        st.dataframe(
            review_profile[
                ["Segment", "Customers", "Exited", "Churn rate"]
            ],
            use_container_width=True,
            hide_index=True,
        )


def model_evaluation(data: pd.DataFrame) -> None:
    st.header("Model Lab and Evaluation")
    st.warning(
        "Candidate fields include age, gender and geography. Do not use for "
        "operational decisions unless approved by legal, compliance and model-risk owners."
    )
    st.write(
        "Leakage controls: identifiers and the target are excluded; complaint, "
        "satisfaction, card type and points are held out pending timing review."
    )
    approved = st.checkbox(
        "I confirm that these candidate features are approved for this prototype and have been reviewed for prediction-time availability."
    )
    if not approved:
        st.info("Confirm the governance review to enable model training.")
        return
    if st.button("Train and evaluate candidate models", type="primary"):
        with st.spinner("Training on a stratified split..."):
            try:
                x, y, excluded = prepare_training_frame(data)
                trained = train_and_evaluate(x, y)
                st.session_state["model_run"] = {
                    "selected_name": trained.selected_name,
                    "pipeline": trained.selected_pipeline,
                    "validation_results": trained.validation_results,
                    "validation_probabilities": trained.validation_probabilities,
                    "y_validation": trained.y_validation,
                    "x_test": trained.x_test,
                    "test_probabilities": trained.test_probabilities,
                    "y_test": trained.y_test,
                    "split_sizes": trained.split_sizes,
                    "excluded_rows": excluded,
                    "seed": 42,
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                    "threshold": 0.5,
                    "model_version": "prototype-v1",
                }
            except (ValueError, TypeError) as exc:
                st.error(f"Training could not complete: {exc}")
    run = st.session_state.get("model_run")
    if not run:
        st.info("Training uses a 60/20/20 train/validation/test split with seed 42.")
        return
    st.subheader("Validation comparison")
    st.caption("Candidate selection uses validation PR-AUC; the test set is kept aside until selection is complete.")
    st.dataframe(run["validation_results"].style.format(precision=3), use_container_width=True, hide_index=True)
    st.caption(
        f"Majority/prevalence baseline PR-AUC on validation: "
        f"{run['y_validation'].mean():.3f} (no customer-level ranking)."
    )
    st.success(f"Selected on validation PR-AUC: {run['selected_name']}")
    st.write(
        f"Split sizes: {run['split_sizes']} · rows excluded for invalid target: {run['excluded_rows']:,} · seed: {run['seed']}"
    )

    threshold = st.slider(
        "Decision threshold (scenario analysis on validation data only)",
        min_value=0.05,
        max_value=0.95,
        value=float(run["threshold"]),
        step=0.01,
    )
    run["threshold"] = threshold
    validation_metrics = evaluate_threshold(
        run["y_validation"], run["validation_probabilities"], threshold
    )
    test_metrics = evaluate_threshold(run["y_test"], run["test_probabilities"], 0.5)
    st.subheader("Validation threshold scenario")
    st.caption("Use validation data to consider a threshold. Do not select a threshold by repeatedly inspecting test results.")
    scenario_cols = st.columns(5)
    for col, key in zip(scenario_cols, ("PR-AUC", "Precision", "Recall", "F1", "False positives")):
        value = validation_metrics[key]
        col.metric(key, f"{value:.3f}" if isinstance(value, float) else f"{value:,}")
    st.subheader("One-time held-out test evaluation")
    st.caption("Fixed threshold 0.50, not adjustable here. Stratified random holdout; source data contains no observation dates, so this is not a temporal backtest.")
    metric_cols = st.columns(6)
    for col, key in zip(metric_cols, ("PR-AUC", "ROC-AUC", "Precision", "Recall", "F1", "Brier score")):
        col.metric(key, f"{test_metrics[key]:.3f}")
    counts = pd.DataFrame(
        [[test_metrics["True negatives"], test_metrics["False positives"]],
         [test_metrics["False negatives"], test_metrics["True positives"]]],
        index=["Actual stayed", "Actual exited"],
        columns=["Predicted stayed", "Predicted exited"],
    )
    st.write("Confusion matrix at fixed test threshold 0.50")
    st.dataframe(counts, use_container_width=True)
    st.write(
        f"False positives: {test_metrics['False positives']:,} · False negatives: "
        f"{test_metrics['False negatives']:,} · fixed test threshold: 0.50"
    )
    curve = threshold_curve(run["y_validation"], run["validation_probabilities"])
    chart = px.line(curve, x="Threshold", y=["Precision", "Recall"], markers=True, title="Precision and recall across thresholds")
    st.plotly_chart(chart, use_container_width=True)
    st.caption("Choose a threshold only after defining outreach capacity and the relative costs of missed churners and unnecessary outreach.")
    observed_rate, mean_probability = calibration_curve(
        run["y_test"], run["test_probabilities"], n_bins=10, strategy="quantile"
    )
    calibration = pd.DataFrame(
        {
            "Mean predicted probability": mean_probability,
            "Observed churn rate": observed_rate,
        }
    )
    calibration_chart = px.line(
        calibration,
        x="Mean predicted probability",
        y="Observed churn rate",
        markers=True,
        title="Held-out test calibration diagnostic",
    )
    calibration_chart.add_scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration"
    )
    st.plotly_chart(calibration_chart, use_container_width=True)
    st.caption("Brier score and calibration plot are diagnostics, not a guarantee that probabilities are calibrated for future customers.")

    summary = {
        "model_version": run["model_version"],
        "model": run["selected_name"],
        "scikit_learn_version": sklearn_version,
        "threshold": threshold,
        "seed": run["seed"],
        "trained_at": run["trained_at"],
        "split_sizes": run["split_sizes"],
        "candidate_features": list(CANDIDATE_FEATURES),
        "held_out_features": list(PENDING_REVIEW_FEATURES),
        "validation_metrics_at_selected_threshold": validation_metrics,
        "test_metrics_at_fixed_threshold_0_5": test_metrics,
        "validation_comparison": run["validation_results"].to_dict(orient="records"),
        "validation_type": "Stratified random split; not time-forward",
    }
    st.download_button(
        "Download evaluation summary (JSON)",
        pd.Series(summary).to_json(indent=2).encode("utf-8"),
        "churn_model_evaluation.json",
        "application/json",
    )


def customer_scoring(data: pd.DataFrame) -> None:
    st.header("Score Customers")
    run = st.session_state.get("model_run")
    if not run:
        st.info("Train a model in Model Evaluation before scoring customers.")
        return
    threshold = st.slider(
        "Risk threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(run["threshold"]),
        step=0.01,
        key="scoring_threshold",
    )
    tab_single, tab_batch = st.tabs(["Single customer", "Batch upload"])
    with tab_single:
        st.caption("Do not enter customer name or identifiers. Form values are held only in this session.")
        values = {}
        with st.form("single_customer_form"):
            fields = st.columns(2)
            normalized = normalize_training_data(data)
            for i, feature in enumerate(CANDIDATE_FEATURES):
                column = fields[i % 2]
                if feature in CATEGORICAL_FEATURES:
                    options = sorted(normalized[feature].dropna().astype(str).unique().tolist())
                    if not options:
                        options = ["Unknown"]
                    values[feature] = column.selectbox(feature, options)
                else:
                    available = normalized[feature].dropna()
                    low, high = EXPECTED_RANGES[feature]
                    min_value = float(available.min()) if not available.empty else float(low or 0)
                    max_value = float(available.max()) if not available.empty else float(high or 1)
                    if low is not None:
                        min_value = max(min_value, float(low))
                    if high is not None:
                        max_value = min(max_value, float(high))
                    if max_value <= min_value:
                        max_value = min_value + 1
                    median = float(available.median()) if not available.empty else min_value
                    step = 1.0 if feature in ("CreditScore", "Age", "Tenure", "NumOfProducts", "HasCrCard", "IsActiveMember") else max((max_value - min_value) / 100, 0.01)
                    values[feature] = column.number_input(
                        feature,
                        min_value=min_value,
                        max_value=max_value,
                        value=min(max(median, min_value), max_value),
                        step=step,
                    )
            submitted = st.form_submit_button("Estimate churn likelihood", type="primary")
        if submitted:
            record = pd.DataFrame([{feature: values[feature] for feature in CANDIDATE_FEATURES}])
            probability = float(run["pipeline"].predict_proba(record)[:, 1][0])
            st.metric("Estimated churn probability", f"{probability:.1%}")
            st.info(f"Risk band: **{risk_band(probability, threshold)}** at threshold {threshold:.2f}. This is a model estimate, not a causal explanation.")
    with tab_batch:
        st.write("Upload a CSV containing all approved candidate feature columns. `CustomerId` or `RowNumber` may optionally be included as a reference only.")
        batch_file = st.file_uploader("Batch scoring CSV", type=["csv"], key="batch_upload")
        if batch_file is not None:
            try:
                batch = read_csv(batch_file)
                features, issues, references = prepare_batch(batch)
                st.write(f"Rows: {len(batch):,} · valid: {len(features):,} · rejected: {issues['Source row'].nunique() if not issues.empty else 0:,}")
                ignored = [
                    column for column in batch.columns
                    if column not in CANDIDATE_FEATURES and column not in IDENTIFIER_COLUMNS
                ]
                if ignored:
                    st.warning("Extra columns will not be used: " + ", ".join(ignored))
                if not issues.empty:
                    st.dataframe(issues, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download rejected-row issues",
                        issues.to_csv(index=False).encode("utf-8"),
                        "rejected_rows.csv",
                        "text/csv",
                    )
                include_ids = st.checkbox(
                    "Include supplied RowNumber/CustomerId in the downloadable result",
                    value=False,
                    help="Reference fields are never sent to the model. Keep disabled unless needed for reconciliation.",
                )
                if st.button("Score valid rows", disabled=features.empty):
                    predictions = score_batch(run["pipeline"], features, threshold)
                    predictions.insert(0, "Source row", references["Source row"].to_numpy())
                    if include_ids:
                        for column in references.columns:
                            if column != "Source row":
                                predictions[column] = references[column].to_numpy()
                    predictions["Model version"] = run["model_version"]
                    predictions["Scored at UTC"] = datetime.now(timezone.utc).isoformat()
                    predictions["Threshold"] = threshold
                    st.dataframe(predictions.head(100), use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download predictions",
                        predictions.to_csv(index=False).encode("utf-8"),
                        "churn_predictions.csv",
                        "text/csv",
                    )
            except ValueError as exc:
                st.error(f"Batch file cannot be scored: {exc}")


def model_card(data: pd.DataFrame) -> None:
    st.header("Model Card and Limitations")
    run = st.session_state.get("model_run")
    st.subheader("Intended use")
    st.write("Analyst decision support for prioritizing human review of possible churn risk. Not for automated account, pricing, eligibility, or service decisions.")
    st.subheader("Data and target")
    st.write(
        f"Current source: {st.session_state.get('data_source', 'Uploaded file')} · "
        f"{len(data):,} rows · target: `Exited` (1 = exited, 0 = stayed)."
    )
    st.subheader("Feature policy")
    st.write("Candidate fields: " + ", ".join(CANDIDATE_FEATURES))
    st.write("Excluded identifiers: " + ", ".join(IDENTIFIER_COLUMNS))
    st.write("Held pending timing/governance review: " + ", ".join(PENDING_REVIEW_FEATURES))
    st.warning("Age, gender and geography can be sensitive or regulated. Operational use requires formal legal, compliance, fairness and model-risk approval.")
    if run:
        metrics = evaluate_threshold(run["y_test"], run["test_probabilities"], 0.5)
        st.subheader("Current model snapshot")
        st.write(
            f"Model: {run['model_version']} ({run['selected_name']}) · scoring threshold: {run['threshold']:.2f} · "
            f"random seed: {run['seed']} · trained: {run['trained_at']}"
        )
        st.caption(f"scikit-learn version: {sklearn_version}")
        st.caption("Metrics below use the fixed 0.50 test threshold. The scoring threshold is selected using validation data.")
        st.json({key: metrics[key] for key in ("PR-AUC", "ROC-AUC", "Precision", "Recall", "F1", "Brier score")})
        st.subheader("Subgroup audit snapshot")
        st.caption(
            "For internal fairness review only. These descriptive test-set "
            "metrics do not establish fairness or justify operational use. "
            "Groups with fewer than 30 test records are suppressed."
        )
        audit = subgroup_audit(run["x_test"], run["y_test"], run["test_probabilities"])
        if audit.empty:
            st.info("No subgroups meet the minimum reporting count.")
        else:
            st.dataframe(audit.style.format({
                "Observed churn": "{:.1%}",
                "Precision": "{:.1%}",
                "Recall": "{:.1%}",
            }), use_container_width=True, hide_index=True)
    st.subheader("Known limitations")
    for limitation in (
        "No observation dates are present, so temporal/future-period validation is unavailable.",
        "Historical churn associations are not causal and do not establish that outreach will prevent churn.",
        "The dataset reflects a specific population and may not represent current customers or future periods.",
        "Complaint and satisfaction measures may be post-outcome or too close to churn; they are excluded pending timing review.",
        "Churn classes are imbalanced, while some feature segments are sparse; small-group metrics may be unstable.",
        "No drift monitoring, production access controls, or authenticated deployment is included in this prototype.",
        "Probability calibration and subgroup fairness require further review before operational use.",
    ):
        st.markdown(f"- {limitation}")
    st.caption("Human review is required. Record and investigate unexpected or unfair outcomes before acting on any score.")


def subgroup_audit(
    test_features: pd.DataFrame,
    target: pd.Series,
    probabilities,
    minimum_group_size: int = 30,
) -> pd.DataFrame:
    """Summarize a small set of test-set groups for a cautious audit view."""
    audit_features = test_features.reset_index(drop=True).copy()
    audit_features["Actual exited"] = np.asarray(target)
    audit_features["Predicted exited"] = np.asarray(probabilities) >= 0.5
    if "Age" in audit_features:
        audit_features["Age band"] = pd.cut(
            audit_features["Age"],
            bins=[-np.inf, 29, 44, np.inf],
            labels=["Under 30", "30-44", "45+"],
        )
    dimensions = [
        column for column in ("Geography", "Gender", "Age band")
        if column in audit_features
    ]
    results = []
    for dimension in dimensions:
        for group, rows in audit_features.groupby(dimension, dropna=False, observed=True):
            if len(rows) < minimum_group_size:
                continue
            actual = rows["Actual exited"].astype(bool)
            predicted = rows["Predicted exited"].astype(bool)
            true_positives = int((actual & predicted).sum())
            results.append(
                {
                    "Dimension": dimension,
                    "Group": str(group),
                    "Customers": len(rows),
                    "Observed churn": float(actual.mean()),
                    "Precision": true_positives / int(predicted.sum()) if predicted.any() else 0.0,
                    "Recall": true_positives / int(actual.sum()) if actual.any() else 0.0,
                    "Flagged rate": float(predicted.mean()),
                }
            )
    return pd.DataFrame(results)


if __name__ == "__main__":
    main()
