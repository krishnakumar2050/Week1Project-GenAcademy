# Bank Customer Churn Intelligence — Project Overview

## Project at a glance

This project is an interactive Streamlit prototype for exploring historical
bank customer churn, comparing baseline prediction models, and scoring
individual or batch records. It also includes a retention-insights dashboard
and a one-page executive brief.

The app is designed for analyst decision support. Segment differences and
model scores are not proof of why an individual customer leaves, and the
prototype does not automate customer treatment or account decisions.

## Main components

- **Streamlit dashboard (`app.py`):** Overview, Data Explorer, Retention
  Insights, Model Evaluation, Score Customers, and Model Card.
- **Data validation (`churn_app/data.py`):** validates training CSV schema,
  identifies warnings, and keeps identifiers out of model features.
- **Modeling (`churn_app/modeling.py`):** compares class-weighted Logistic
  Regression and Random Forest with a stratified train/validation/test split.
- **Batch and single scoring (`churn_app/scoring.py`, `app.py`):** accepts
  approved prediction fields and produces risk estimates.
- **Tests (`tests/test_churn_app.py`):** covers data, model, scoring, and app
  behavior.
- **Executive brief (`CUSTOMER_CHURN_EXECUTIVE_BRIEF.pdf`):** one-page summary
  of historical churn signals and controlled-pilot recommendations.

To run locally, install `requirements.txt`, then use `streamlit run app.py`.
Upload the training CSV in the dashboard, or configure `CHURN_DATA_PATH`.

## Dataset used

**Source file:** `Customer-Churn-Records.csv`, supplied as a training-data
attachment and located at `/Users/krishnakumar/Downloads/Customer-Churn-Records.csv`
in the development environment. The application does not require this fixed
path; the CSV can be uploaded in the dashboard.

**Observed contents:**

- 10,000 customer records and 18 columns.
- Target: `Exited` (0 = stayed, 1 = exited).
- 2,038 recorded exits; historical exit rate 20.38%.
- The supplied file had no blank cells and no duplicate `CustomerId` values.
- It includes customer characteristics and relationship indicators such as
  credit score, geography, gender, age, tenure, balance, number of products,
  card ownership, activity, salary, complaint status, satisfaction score,
  card type, and points earned.

`RowNumber`, `CustomerId`, and `Surname` are excluded from model features.
`Complain`, `Satisfaction Score`, `Card Type`, and `Point Earned` are held out
pending point-in-time and governance review. The data contains no observation
dates, so random holdout evaluation is not a temporal test of future
performance.

## Key descriptive findings

| Historical segment | Customers | Exits | Exit rate |
|---|---:|---:|---:|
| Inactive | 4,849 | 1,303 | 26.9% |
| Active | 5,151 | 735 | 14.3% |
| 1 product | 5,084 | 1,409 | 27.7% |
| 2 products | 4,590 | 349 | 7.6% |
| 3 products | 266 | 220 | 82.7% |
| 4 products | 60 | 60 | 100.0% |
| Germany | 2,509 | 814 | 32.4% |
| France | 5,014 | 811 | 16.2% |
| Spain | 2,477 | 413 | 16.7% |

These are descriptive associations. In particular, the 3- and 4-product
groups are small and their rates may be unstable. Complaint status almost
matches the target in this file (2,034 of 2,044 complaint-flagged rows exited);
this may reflect target leakage or feature timing and is not treated as a
campaign trigger.

## Development prompts used

The following summarizes the user requests that shaped the project during
this conversation:

1. **Plan the application:** “Create a detailed level plan of interactive web
   application using Streamlit to analyse dataset of customer churn records
   of bank based on attached training dataset to enable predictions for future
   customers.”
2. **Build the application:** “Create streamlit application based on this md
   file.”
3. **Support dashboard uploads:** “Can you modify current source code of
   application to make dataset uploadable via dashboard?”
4. **Run the application:** “Run this application.”
5. **Create retention visuals:** “Create visual representation on some of
   important factors from dataset that determine key criteria for customers
   to choose to exit that can be used by retention teams to prepare campaigns
   to proactively reach out and retain them.”
6. **Create an executive deliverable:** “Generate a single page pdf document
   to be presented to C-suite executives with top 3 key reasons for customers
   churning based on the attached dataset and highlight 3 actionable insights
   for business recommendations that can be incorporated to significantly
   reduce churn percentage ratio of customers.”
7. **Run/restart iterations:** “Run this application” and “Restart the
   application.”
8. **Evaluate production readiness:** “Can you evaluate this prototype
   application against industry standards and ensure production grade coding
   standards, resilient and high performance benchmarks. If any changes
   required in modifying the application, please seek approval before
   proceeding.” The review found the prototype was not production-ready;
   proposed code changes were presented for approval, and no application code
   was modified.

These are user-facing project prompts from this conversation, presented as a
brief paraphrased prompt history; no hidden system or developer instructions
are included.

## Important interpretation and use limits

- The dataset alone cannot establish causal “reasons” or prove that an action
  will reduce churn.
- The dataset has no dates or intervention results; there is no evidence here
  of model performance over time or campaign uplift.
- Campaign proposals should be tested with controlled pilots and a holdout,
  measuring retention lift, customer experience, opt-outs, complaints, and
  incremental economics before scaling.
- Age, gender, geography, and other potentially sensitive fields require
  legal, compliance, fairness, and model-risk review before operational use.
- Human review is required; scores should not be used as automated grounds
  for eligibility, pricing, or service decisions.
