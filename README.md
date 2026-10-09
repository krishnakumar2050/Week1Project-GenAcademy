# Bank Customer Churn Intelligence

An interactive Streamlit prototype for exploring historical bank churn,
evaluating candidate classification models, and scoring future customers.
The detailed product and delivery plan is in
[STREAMLIT_CHURN_APP_PLAN.md](./STREAMLIT_CHURN_APP_PLAN.md).

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Upload `Customer-Churn-Records.csv` in the **Training dataset** section at the
top of the dashboard. To use a local file
automatically instead, set `CHURN_DATA_PATH` to its path before starting the
app:

```bash
CHURN_DATA_PATH=/path/to/Customer-Churn-Records.csv streamlit run app.py
```

The app does not copy the training data into the repository or save uploads
to disk. The current in-memory model and temporary scoring data are scoped to
the Streamlit session. The user may download evaluation summaries and
predictions explicitly.

## App pages

- **Overview:** data and model status, target balance, and test-set snapshot.
- **Data Explorer:** schema/quality report, filters, distributions, and
  segment churn rates with denominators.
- **Retention Insights:** historical group-level churn patterns, observed exit
  counts, and cautious campaign-planning hypotheses; not individual targeting.
- **Model Evaluation:** class-weighted Logistic Regression and Random Forest
  comparison, validation-only threshold scenarios, and a fixed-threshold
  final test snapshot.
- **Score Customers:** single-customer and batch scoring using candidate
  features only.
- **Model Card:** intended use, feature policy, evaluation context, and
  limitations.

## Data and modeling safeguards

- `RowNumber`, `CustomerId`, and `Surname` are never model features.
- `Exited` is the target. `Complain`, `Satisfaction Score`, `Card Type`, and
  `Point Earned` are held out pending point-in-time and governance review.
- Age, gender, and geography require legal, compliance, fairness, and
  model-risk approval before operational use. The app requires a prototype
  approval acknowledgement before training.
- Candidate models use a stratified 60/20/20 train/validation/test split.
  Model selection is based on validation PR-AUC. Threshold exploration uses
  validation data only. The final test snapshot uses a fixed 0.50 threshold.
- The dataset has no observation dates; random holdout metrics are not
  time-forward estimates of future-period performance.
- Model output is decision support, not a causal explanation or an automated
  customer decision. Review the model card and obtain required bank approvals
  before operational deployment.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```
