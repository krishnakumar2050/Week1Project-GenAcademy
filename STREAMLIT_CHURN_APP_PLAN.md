# Bank Customer Churn Analysis and Prediction App — Detailed Plan

## 1. Purpose and intended users

Build an interactive Streamlit application that helps bank analysts understand
historical customer churn, train and compare churn models, and estimate churn
likelihood for a future customer or a batch of customers.

The app is a decision-support tool. A prediction is a risk estimate, not a
certainty or an instruction to deny service, change pricing, or contact a
customer. Retention actions and any customer-level decisions remain with
authorized bank staff.

### Primary users

- **Analysts / data scientists:** inspect the training data, validate model
  quality, set a decision threshold, and export evaluation results.
- **Retention or relationship teams:** review risk-ranked customers and use
  approved, non-sensitive explanations to prioritize human follow-up.
- **App administrator:** select the approved data source and model version,
  and review data/model warnings.

## 2. Dataset assessment

The supplied file is
[`Customer-Churn-Records.csv`](/Users/krishnakumar/Downloads/Customer-Churn-Records.csv).
The app should not depend on that absolute Downloads path; make the dataset
selectable through a file upload or a configurable local data path.

| Observation | Implication for the app |
|---|---|
| 10,000 records and 18 columns | Suitable for interactive analysis and a lightweight scikit-learn model in a local Streamlit process. |
| `Exited` is binary: 7,962 stayed and 2,038 exited (20.38% churn) | Show the class balance prominently. Accuracy alone is not an acceptable model-quality measure. |
| No blanks were found in the supplied file | Still validate missing and malformed values on every upload; future input files may differ. |
| `CustomerId` is unique; `RowNumber` is a row index; `Surname` is identifying text | Exclude these fields from training and model input. They are not useful generalizable customer risk features and can expose personal data. |
| Customer characteristics include credit score, age, tenure, balance, product count, card ownership, activity, salary, geography and gender | These are candidate predictors, subject to data-governance and fairness approval. |
| `Complain`, `Satisfaction Score`, `Card Type`, and `Point Earned` are also present | Their availability at the moment a prediction is made must be confirmed. Complaint and satisfaction measures may occur after or very near a churn event and can create target leakage. Default to excluding them until approved. |
| No event or observation date is present | A temporal backtest cannot be performed from this dataset. A stratified random holdout is only an initial benchmark and must not be presented as proof of future-period performance. |
| `NumOfProducts` has very few records at value 4 | Call out sparse segments and avoid overconfident segment-level conclusions. |

Before model training, the data owner must confirm the meaning of `Exited`,
the observation horizon, and whether each candidate feature is available
before the intervention being supported. The app must record this limitation
alongside the model results.

## 3. Scope and non-goals

### In scope

- Load and validate the supplied training dataset or a compatible upload.
- Explore churn patterns with interactive filters and descriptive charts.
- Train and compare a small set of reproducible classification pipelines.
- Evaluate a final model on untouched test data and display useful metrics.
- Enter one customer's approved features and receive a churn probability and
  threshold-based risk band.
- Upload a batch of future customer records, score them, inspect results, and
  download a CSV containing predictions.
- Display model limitations, feature-use decisions, data quality, and version
  information.

### Out of scope for the first release

- Production scoring APIs, bank-system integration, authentication or
  authorization infrastructure, scheduled retraining, automated campaigns,
  and direct changes to customer accounts.
- Claims of causal impact or recommendations that an action will prevent
  churn.
- A production-grade model-monitoring service. The first release may provide
  manual monitoring exports and a documented retraining workflow.

## 4. Proposed user journey and navigation

Use a sidebar for navigation, dataset/model status, and shared controls. The
main pages can be implemented as Streamlit multipage pages or a page selector;
choose the approach that fits the repository once implementation begins.

### A. Home / Executive Overview

- State the dataset currently loaded, row count, last validation result, and
  active model/version.
- Show headline figures: customers, churn rate, predicted high-risk count
  (when a model is available), and test-set recall/precision or PR-AUC with
  the evaluation population stated.
- Include a short interpretation of the selected threshold and a persistent
  decision-support disclaimer.
- Offer clear routes to **Explore data**, **Evaluate model**, and **Score
  customers**.

### B. Data Quality and Explorer

- Load the supplied CSV or upload a replacement CSV.
- Validate required columns, types, allowed categories, target encoding,
  duplicate identifiers/rows, missingness, and plausible numeric ranges.
- Show a blocking error for incompatible schema and a downloadable validation
  report for non-blocking warnings. Never silently discard rows or columns.
- Display a schema table with field name, inferred type, missing count,
  distinct count, range/categories, and whether the field is approved for
  modeling.
- Provide filters for geography, age range, tenure, product count, activity,
  and churn label; show filtered sample size.
- Include churn-rate views by geography, age bands, tenure, activity, product
  count, and other approved dimensions; show group denominators and avoid
  highlighting very small groups as reliable.
- Show distributions for numeric fields, category counts, missingness, and
  target balance. Do not expose customer names or identifiers in aggregate
  charts or tables.
- Make clear that observational differences are associations, not causes.

### C. Model Lab / Evaluation

- Select candidate model pipeline and approved feature set from a controlled
  list rather than accepting arbitrary executable model files.
- Recommended first candidates: a class-weighted logistic regression baseline
  and a tree-based ensemble such as Random Forest or HistGradientBoosting.
- Use a stratified train/validation/test split, with the test split held aside
  until final comparison. Fit preprocessing only on training folds.
- One-hot encode categorical features; scale numeric features for the
  logistic-regression pipeline; handle missing values with explicit,
  training-fitted imputation if uploads contain blanks.
- Tune only on training/validation data or via cross-validation. Do not use the
  test set for feature choice, hyperparameter selection, or threshold tuning.
- Report confusion matrix, precision, recall, F1, ROC-AUC, PR-AUC, class
  prevalence, and probability calibration. Include a threshold-versus-
  precision/recall plot and a comparison to a majority-class baseline.
- Use PR-AUC and churn-class recall/precision in context; there is no single
  “best” metric without an approved intervention capacity and error-cost
  tradeoff.
- Permit threshold adjustment for scenario analysis, clearly separating
  threshold selection from model retraining. Show false positives and false
  negatives at the selected threshold.
- Save evaluation summary, feature list, split seed, model parameters, and
  library/model version information. Do not represent a random split as
  time-forward validation.

### D. Single Customer Scoring

- Provide a form with only features approved for use at scoring time.
- Validate every value against the training schema and supported ranges.
- Return a calibrated churn probability where available, a configurable risk
  band (thresholds documented in the UI), and an explanation of what the
  score means.
- Show contributing factors using a method appropriate to the selected
  model (e.g., coefficients for logistic regression or local/global
  explanation tooling if later approved). Label explanations as model
  associations, not causes.
- Do not request name, surname, customer ID, or other identifiers for
  prediction. Do not persist submitted customer-level form data by default.

### E. Batch Scoring

- Accept a CSV with the same approved prediction-feature schema; identifiers
  may optionally be supplied solely to match results back to source rows, but
  must be excluded from the model and hidden from default previews.
- Validate and summarize row-level errors before scoring. Let the user
  download a rejected-rows report rather than silently dropping invalid
  records.
- Score only valid records when the user confirms; display counts for total,
  scored, and rejected rows.
- Provide a preview with probability, risk band, and any supplied reference
  key. Keep original uploaded fields out of the download unless explicitly
  needed and clearly labeled.
- Export predictions to CSV and state model version, score date, and chosen
  threshold in a companion summary or columns.

### F. Model Card / Limitations

- Document training data source, target definition, included/excluded
  features, split strategy, metrics, threshold, known data gaps, and model
  version.
- Explain known limitations: no time column, no evidence of drift performance,
  dataset-specific geography and population, possible feature leakage,
  imbalance, sparse subgroups, and possible demographic disparities.
- Show subgroup metrics for approved review dimensions where sample sizes
  support meaningful estimates. Avoid treating group metrics as definitive
  where denominators are small.
- Provide a clear route to report a problem and withdraw/replace an approved
  model.

## 5. Data and modeling design

### Feature governance

Initial default feature policy:

- **Exclude:** `RowNumber`, `CustomerId`, `Surname`, and `Exited` (the target).
- **Hold out pending timing/governance review:** `Complain`,
  `Satisfaction Score`, `Card Type`, and `Point Earned`. The first two are
  especially important to check for target leakage and point-in-time
  availability.
- **Candidate, subject to policy review:** `CreditScore`, `Geography`,
  `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`,
  `IsActiveMember`, and `EstimatedSalary`.

Gender, age, geography, and other potentially sensitive or regulated attributes
must not be enabled for operational scoring without the bank's legal,
compliance, and fairness review. They may need to be excluded from production
features while retained in a separately controlled fairness audit. Keep the
scoring feature schema explicit and versioned so training and prediction use
the same fields.

### Validation and splitting

- Treat CSV values as untrusted input; enforce schema and safe parsing.
- Preserve and report invalid rows rather than silently coercing arbitrary
  values.
- Check for duplicate rows, duplicate identifiers, unexpected target values,
  impossible ranges, unseen categories, and severe class imbalance.
- Use stratification for initial train/validation/test splits and fix/report
  the random seed for reproducibility.
- When dated records become available, replace random validation as the
  principal future-performance check with a time-based holdout.
- If preprocessing, resampling, or feature selection is used, fit it only
  within each training fold. Do not oversample before the split.

### Model and threshold choices

- Establish logistic regression as a transparent baseline, followed by one
  tree ensemble for non-linear patterns.
- Consider class weights rather than changing the observed prevalence.
- Calibrate probabilities on validation data if calibration diagnostics show
  they are needed.
- Select a threshold only after the bank defines the relative cost of missed
  churners versus unnecessary outreach and expected outreach capacity.
- Report threshold-specific confusion counts and precision/recall; do not use
  a universal risk cutoff without approval.

## 6. Suggested implementation architecture

Keep the initial deployment simple and local:

- **UI:** Streamlit pages and reusable display/form components.
- **Data layer:** a loader and schema/quality validator shared by training,
  exploration, and scoring.
- **Model layer:** scikit-learn `Pipeline`/`ColumnTransformer` artifacts with
  preprocessing and classifier packaged together.
- **Persistence:** local model artifact plus a small JSON metadata file for
  feature schema, metrics, threshold, training date, random seed, and
  dependency versions. Avoid pickle/joblib artifacts from untrusted sources.
- **State:** use Streamlit session state for page selections and temporary
  interactions; do not store customer data in session state longer than
  required.
- **Configuration:** separate paths and default threshold from source code.
  Never commit customer data, generated predictions, credentials, or model
  artifacts containing sensitive data.
- **Testing:** unit-test the validator, preprocessing, training split and
  metrics, threshold logic, and batch scoring; add a Streamlit smoke test for
  the navigation and key page flows.

### Proposed project layout (to confirm against the repository)

```text
app.py
pages/
  1_Data_Explorer.py
  2_Model_Evaluation.py
  3_Score_Customers.py
  4_Model_Card.py
src/
  data.py
  features.py
  modeling.py
  scoring.py
  reporting.py
models/                 # generated locally; not committed by default
tests/
  test_data.py
  test_modeling.py
  test_scoring.py
requirements.txt        # or the repository's existing dependency manager
```

This is a proposed structure, not a requirement to create each file if the
existing project establishes another convention.

## 7. Privacy, safety, and operational safeguards

- Keep the CSV local unless the bank has explicitly approved a hosted
  deployment and data-processing arrangement.
- Exclude direct identifiers and names from training, visualizations, logs,
  and default scoring exports.
- Do not log raw records, uploaded file contents, or individual predictions.
- Explain file retention and clear uploaded data when the session ends.
- Do not render user-provided strings as unsafe HTML or execute uploaded
  content.
- Make feature approval, access control, and retention requirements explicit
  before deployment beyond a local analyst prototype.
- Warn that model outputs can be biased or stale; provide human review and an
  escalation path for questionable scores.
- Display metric denominators and uncertainty where feasible; do not claim
  performance for a population or period not represented by the evaluation.

## 8. Delivery phases and completion criteria

### Phase 0 — Confirm use case and governance

- Confirm intended intervention, prediction horizon, label definition, and
  prediction-time availability of each field.
- Obtain legal/compliance approval for feature use and fairness review.
- Define intervention capacity and the cost of false positives and false
  negatives.
- **Exit criteria:** signed feature inclusion/exclusion decision and agreed
  success metrics.

### Phase 1 — Data validation and exploratory analysis

- Build upload/path-based loading, schema checks, validation reporting, and
  privacy-safe exploratory views.
- Verify counts against the source data: 10,000 rows; 7,962 `Exited=0`;
  2,038 `Exited=1`; 20.38% churn; no missing cells in the provided file.
- **Exit criteria:** valid source file loads consistently; invalid schemas
  produce understandable errors; exploratory summaries reconcile to source.

### Phase 2 — Baseline model and evaluation

- Build a reproducible preprocessing/model pipeline and baseline comparison.
- Keep an untouched stratified test set; evaluate PR-AUC, recall, precision,
  calibration, and confusion counts.
- **Exit criteria:** tests prevent identifier/target leakage; metrics and
  threshold are reproducible and clearly caveated as random-holdout results.

### Phase 3 — Interactive scoring

- Implement validated single-record and batch scoring with probability,
  threshold/risk-band controls, row error reports, and CSV download.
- **Exit criteria:** scoring schema matches training schema; malformed rows
  are reported; identifiers do not affect scores; downloaded output includes
  model/threshold context.

### Phase 4 — Review, usability, and deployment readiness

- Add model card, limitation/fairness review, accessibility checks, and
  Streamlit smoke tests.
- Review resource use, file handling, dependency pinning, and deployment
  configuration.
- **Exit criteria:** analysts can complete exploration, evaluation, and
  scoring flows without silent data loss; governance owners approve any
  non-local deployment.

### Phase 5 — Future production extension (separate approval)

- Add authenticated access, approved storage and retention, temporal
  validation, scheduled drift/performance monitoring, versioned model
  registry, and controlled retraining/deployment.
- **Exit criteria:** security, privacy, fairness, operational, and model-risk
  reviews are complete before operational use.

## 9. Acceptance checklist

- [ ] A user can load the supplied CSV and see data quality and target balance.
- [ ] Churn analyses respond to filters and show sample denominators.
- [ ] Training excludes identifiers, `Exited`, and unapproved or
      post-outcome fields.
- [ ] Preprocessing is fit on training data only; test data remains untouched
      through model and threshold selection.
- [ ] Evaluation includes churn-focused metrics and communicates the
      non-temporal validation limitation.
- [ ] Single and batch scoring accept only the approved feature schema and
      report invalid input explicitly.
- [ ] Risk bands can be inspected/adjusted without retraining and are
      described as decision-support categories.
- [ ] Batch results can be downloaded without unnecessarily exposing
      identifiers or raw uploaded columns.
- [ ] The UI does not imply causation, certainty, or automatic decisioning.
- [ ] Tests cover data validation, leakage safeguards, split/metric behavior,
      thresholding, batch errors, and core Streamlit flows.

## 10. Open decisions before implementation

1. What retention intervention and prediction horizon does the bank intend to
   support?
2. Which fields are demonstrably available before that intervention, and
   which are approved for model use?
3. Should the prototype run strictly locally, or is an approved hosted
   environment available?
4. What outreach capacity and error-cost tradeoff should guide the
   threshold?
5. Can records with observation dates be supplied for time-based validation?
