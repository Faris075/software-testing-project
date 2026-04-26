# AI-Powered Student Performance Analytics System
### Dev2 Instruction Sheet — ML, Visualisation & Dashboard

**Repository:** https://github.com/Faris075/software-testing-project/  
**Project Date:** 2026  
**Your counterpart's sheet:** `INSTRUCTIONS_DEV1.md`

---

## Your Responsibilities at a Glance

| Task | Phase | File(s) |
|---|---|---|
| Data visualisation (all charts) | Phase 4 | `visualisation/charts.py` |
| ML models — regression + classification | Phase 5 | `models/train.py`, `models/predict.py` |
| Model evaluation | Phase 6 | `models/evaluate.py` |
| Streamlit dashboard (your tabs) | Phase 7 | `dashboard/app.py` (shared with Dev1) |
| Unit tests for ML functions | Testing | `tests/test_models.py`, `tests/test_predict.py`, `tests/test_vle.py` |
| Review Dev1's PRs that touch data used by models | Ongoing | — |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack](#2-tech-stack)
3. [Repository Setup & Git Workflow](#3-repository-setup--git-workflow)
4. [Project Structure](#4-project-structure)
5. [Environment Setup](#5-environment-setup)
6. [Phase 4 — Data Visualisation](#phase-4--data-visualisation)
7. [Phase 5 — AI/ML Prediction Models](#phase-5--aiml-prediction-models)
8. [Phase 6 — Model Evaluation](#phase-6--model-evaluation)
9. [Phase 7 — Streamlit Dashboard (Your Tabs)](#phase-7--streamlit-dashboard-your-tabs)
10. [Data Schema Reference](#data-schema-reference)
11. [Commit & PR Guidelines](#commit--pr-guidelines)
12. [Your Testing Checklist](#your-testing-checklist)
13. [Your Future Enhancements](#your-future-enhancements)
14. [Submission Checklist](#submission-checklist)

---

## 1. Project Overview

This system works with **two complementary datasets**:

1. **Synthetic cohort marks** — historical subject marks for 2024 and 2025, used to train regression models predicting 2026 marks.
2. **Real VLE dataset** — 160 real student instances with online behaviour and neighbourhood features, used to train a pass/fail classifier.

**Your role** is to own the ML pipeline: you build all charts, train and save models for both prediction tasks, evaluate model quality, and wire up your tabs in the shared Streamlit dashboard.

### Dataset 1 — Synthetic Cohort (Marks)
- **Students:** 15 (IDs: `S001` – `S015`)
- **Subjects:** Math, Physics, CS, English, Statistics
- **Years:** 2024, 2025 (historical) → 2026 (predicted)
- **Mark range:** 0–100
- **File:** `data/students_clean.csv` (produced by Dev1's `preprocessing/clean.py`)

### Dataset 2 — Real VLE Dataset (Pass/Fail)
- **Instances:** 160 students (136 pass, 24 fail — **imbalanced**)
- **Features:** 19 (8 behaviour + 11 neighbourhood/contextual)
- **Label:** `label` — 0 = pass, 1 = fail
- **Files:** `data/vle/original/original.csv`, `data/vle/test/test.csv`, `data/vle/train_validate/csv/*.csv`
- Use `data/vle/train_validate/csv/smote.csv` for training to address class imbalance.

#### VLE Feature Reference

| Feature | Description |
|---|---|
| `gender` | 0 = male, 1 = female |
| `age` | Student age at week 6 |
| `logins` | Count of module area logins |
| `total_hours` | Total hours in module area |
| `pct_avg_hours` | % of average total hours |
| `presence_count` | Count of attended compulsory sessions |
| `absence_count` | Count of missed compulsory sessions |
| `pct_attended` | % of compulsory sessions attended |
| `attending_from_home` | 1 = home address, 0 = alternative |
| `distance_to_uni_km` | Distance from term-time address to campus |
| `polar4_quintile` | HE participation area classification (current) |
| `polar3_quintile` | HE participation area classification (old) |
| `adult_he_2001_quintile` | Adult HE proportion in area (2001 census) |
| `adult_he_2011_quintile` | Adult HE proportion in area (2011 census) |
| `tundra_msoa_quintile` | State school HE participation rate (MSOA) |
| `tundra_lsoa_quintile` | State school HE participation rate (LSOA) |
| `gaps_gcse_quintile` | Observed vs expected GCSE attainment by area |
| `gaps_gcse_ethnicity_quintile` | GCSE attainment gap adjusted for ethnicity |
| `uni_connect_target_ward` | 1 if POLAR3=1 and GCSE Gap = 1 or 2 |

---

## 2. Tech Stack

| Category | Library / Tool | Your usage |
|---|---|---|
| Data handling | `pandas`, `numpy` | Feature engineering, data loading |
| Visualisation | `matplotlib`, `seaborn` | All chart functions |
| Machine learning | `scikit-learn` | Regression + classification models |
| Model persistence | `joblib` | Save/load `.pkl` files |
| Interactive dashboard | `streamlit` | Your dashboard tabs |
| Environment | `python 3.10+`, `pip`, `venv` | — |
| Version control | `git`, GitHub | — |

---

## 3. Repository Setup & Git Workflow

### Clone & set up

```bash
git clone https://github.com/Faris075/software-testing-project.git
cd software-testing-project
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Your branches

```
main           ← protected; only accepts PRs
├── dev        ← shared integration branch
│   ├── feature/dev2-visualisation     ← your Phase 4 work
│   ├── feature/dev2-ml-models         ← your Phase 5 work
│   ├── feature/dev2-evaluation        ← your Phase 6 work
│   └── feature/dashboard              ← shared with Dev1 (Phase 7)
```

### Daily workflow

```bash
# Start of day — sync dev
git checkout dev
git pull origin dev

# Switch to your branch
git checkout feature/dev2-visualisation   # or whichever phase you're on

# Work, commit, push, then open PR into dev
git add .
git commit -m "feat(viz): your description here"
git push origin feature/dev2-visualisation
# Open PR on GitHub: base=dev, compare=feature/dev2-visualisation
```

### Merging to main

Once all features are merged into `dev` and all tests pass, coordinate with Dev1:

```bash
git checkout main
git merge --no-ff dev -m "release: v1.0.0 - initial system"
git push origin main
```

> Never push directly to `main`. Every change goes through a PR reviewed by Dev1.

---

## 4. Project Structure

Focus on the folders/files you own:

```
software-testing-project/
├── data/
│   ├── students_clean.csv             ← read from Dev1 (clean.py output)
│   └── vle/
│       ├── original/original.csv      ← read-only source data
│       ├── test/test.csv              ← held-out eval set
│       └── train_validate/csv/
│           ├── smote.csv              ← use this for classifier training
│           └── ...                    ← other SMOTE variants
├── visualisation/
│   └── charts.py                      ← YOU implement this
├── models/
│   ├── train.py                       ← YOU implement this
│   ├── predict.py                     ← YOU implement this
│   └── evaluate.py                    ← YOU implement this
├── saved_models/
│   ├── linear_regression_{subject}.pkl  ← YOU save these
│   ├── decision_tree_{subject}.pkl
│   ├── random_forest_{subject}.pkl
│   ├── classifier_logistic.pkl
│   ├── classifier_rf.pkl
│   └── vle_scaler.pkl
├── dashboard/
│   └── app.py                         ← shared; you own Tabs 3 + VLE view
├── tests/
│   ├── test_models.py                 ← YOU write this
│   ├── test_predict.py                ← YOU write this
│   └── test_vle.py                    ← YOU write this
└── outputs/
    └── charts/                        ← YOU save chart PNGs here
```

> **Prerequisite:** You depend on Dev1's `data/students_clean.csv` and the seeded SQLite DB. Run `python database/seed.py` and `python preprocessing/clean.py` first (or pull the files from `dev` once Dev1 pushes them).

---

## 5. Environment Setup

### requirements.txt (already in repo)

```
pandas>=2.1.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
scikit-learn>=1.4.0
scipy>=1.12.0
streamlit>=1.33.0
sqlalchemy>=2.0.0
reportlab>=4.1.0
joblib>=1.3.0
pytest>=8.0.0
```

Install everything with:

```bash
pip install -r requirements.txt
```

---

## Phase 4 — Data Visualisation

**File:** `visualisation/charts.py`

**Goal:** Create reusable, self-contained chart functions consumed by both the dashboard and the PDF report (Dev1 will embed your chart PNGs in the PDF).

### Module-level setup

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pathlib

sns.set_theme(style="whitegrid", palette="muted")

CHARTS_DIR = pathlib.Path(__file__).parent.parent / "outputs" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
```

### Charts to implement

Every function accepts an optional `ax=None`. When `ax is None`, create a new figure, save it to `outputs/charts/`, and return the `fig`.

| Function signature | Chart type | Description |
|---|---|---|
| `bar_subject_averages(df, ax=None)` | Bar | Average mark per subject across all students |
| `bar_student_averages(df, year, ax=None)` | Bar | Average mark per student for a given year |
| `line_progress(df, student_id, ax=None)` | Line | 2024 → 2025 per-subject progression for one student |
| `line_class_progress(df, ax=None)` | Line | Class average per subject: 2024 vs 2025 |
| `comparison_bar(df, student_id, ax=None)` | Grouped bar | Side-by-side 2024 vs 2025 for one student |
| `heatmap_correlation(corr_matrix, ax=None)` | Heatmap | Seaborn heatmap of subject correlations |
| `scatter_actual_vs_predicted(y_true, y_pred, subject, ax=None)` | Scatter | Actual vs predicted marks for a subject |
| `bar_predictions_2026(pred_df, student_id, ax=None)` | Bar | Predicted 2026 marks per subject for one student |
| `bar_vle_class_distribution(df_vle, ax=None)` | Bar | Pass vs fail counts in VLE dataset |
| `heatmap_vle_correlation(df_vle, ax=None)` | Heatmap | Correlation between VLE features |
| `bar_feature_importance(importances, feature_names, ax=None)` | Bar | Random Forest feature importances for VLE classifier |
| `plot_roc_curve(fpr, tpr, auc_score, model_name, ax=None)` | Line | ROC curve for a classifier |
| `plot_confusion_matrix(cm, model_name, ax=None)` | Heatmap | Confusion matrix for a classifier |

### Save helper pattern

```python
def _save(fig, name: str) -> None:
    fig.savefig(CHARTS_DIR / f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
```

---

## Phase 5 — AI/ML Prediction Models

**File:** `models/train.py` + `models/predict.py`

**Goal:** Train models for two tasks and save them as `.pkl` files for use in the dashboard.

---

### Task A — Mark Regression (Synthetic Cohort)

Build training examples from `data/students_clean.csv`:

- **Features (X):** `[year_encoded, Math, Physics, CS, English, Statistics]` from 2024 rows
- **Target (y):** mark in one subject for 2025 row of the same student
- Train on 2024→2025; evaluate on 2025 actuals; then retrain on **all data (2024+2025)** to predict **2026**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib

SUBJECTS = ["Math", "Physics", "CS", "English", "Statistics"]

REGRESSION_MODELS = {
    "LinearRegression": LinearRegression(),
    "DecisionTree":     DecisionTreeRegressor(max_depth=4, random_state=42),
    "RandomForest":     RandomForestRegressor(n_estimators=100, random_state=42),
}

# For each subject and each model:
joblib.dump(model, f"saved_models/{model_name}_{subject}.pkl")
```

---

### Task B — Pass/Fail Classification (VLE Dataset)

**Important:** The dataset is imbalanced (136 pass vs 24 fail). Use the SMOTE-balanced training CSV:

```python
df_train = pd.read_csv("data/vle/train_validate/csv/smote.csv")
df_test  = pd.read_csv("data/vle/test/test.csv").dropna(subset=["label (fail=1, pass=0)"])
```

#### Feature columns

After loading, rename columns (see `database/db_utils.py` for the exact column map). Use all 19 feature columns — drop `id`, `label`, `split`.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

CLASSIFIERS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest":       RandomForestClassifier(n_estimators=100, random_state=42),
}

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
joblib.dump(scaler, "saved_models/vle_scaler.pkl")

# Save each classifier
joblib.dump(model, f"saved_models/classifier_{name}.pkl")
```

---

### predict.py — Inference functions

```python
def predict_2026(student_features: dict, model_name: str) -> dict:
    """
    Input:  student_features = {"Math": 82, "Physics": 71, ...} (2025 marks)
            model_name = "LinearRegression" | "DecisionTree" | "RandomForest"
    Output: {"Math": 84.2, "Physics": 73.1, ...}  (2026 predicted marks)
    Loads the relevant .pkl for each subject.
    """

def predict_pass_fail(feature_row: dict, model_name: str) -> tuple[int, float]:
    """
    Input:  feature_row = {"gender": 0, "age": 21, "logins": 15, ...}
            model_name = "LogisticRegression" | "RandomForest"
    Output: (predicted_label: int, fail_probability: float)
    Loads vle_scaler.pkl and classifier_{model_name}.pkl.
    """
```

Run training with:

```bash
python models/train.py
```

---

## Phase 6 — Model Evaluation

**File:** `models/evaluate.py`

**Goal:** Quantify model quality for both tasks so results appear in the dashboard and PDF.

---

### Regression metrics (Synthetic Cohort)

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import numpy as np

def rmse(y_true, y_pred) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred) -> float:
    return mean_absolute_error(y_true, y_pred)

def cross_validate_model(model, X, y, cv: int = 5) -> dict:
    """Returns {'mean': float, 'std': float} of RMSE across k folds."""

def evaluate_all_models(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
    [model, subject, RMSE, MAE, CV_RMSE_mean, CV_RMSE_std]
    Evaluates all 3 regression models across all 5 subjects.
    """
```

---

### Classification metrics (VLE Dataset)

> **Key insight:** Accuracy alone is misleading here — a model that predicts all-pass gets 85% accuracy but is useless. Always prioritise `recall_fail` and `roc_auc`.

```python
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)

def evaluate_classifier(model, X_test, y_test) -> dict:
    """
    Returns:
    {
        'accuracy':    float,
        'f1_macro':    float,
        'roc_auc':     float,
        'recall_fail': float,   # recall for class 1 (fail) — most important
        'cm':          ndarray, # confusion matrix
        'fpr':         ndarray, # for ROC curve plot
        'tpr':         ndarray,
    }
    """

def evaluate_all_classifiers(X_test, y_test) -> pd.DataFrame:
    """
    Loads each saved classifier .pkl and evaluates.
    Returns DataFrame with columns:
    [model, accuracy, f1_macro, roc_auc, recall_fail]
    """
```

Save the evaluation results to `outputs/eval_results.csv` so Dev1 can embed the table in the PDF.

---

## Phase 7 — Streamlit Dashboard (Your Tabs)

**File:** `dashboard/app.py` (shared with Dev1 — coordinate before editing)

**Goal:** Wire your charts, models, and evaluation functions into specific dashboard tabs. Do not place business logic here — only UI calls.

### To run

```bash
streamlit run dashboard/app.py
```

### Your sections of the dashboard

You are responsible for the following parts of the layout:

```
Main area — Marks Cohort view
  └── [Tab 3] 2026 Predictions                           ← YOU build this
        ├── Sidebar: model selector (LinearRegression / DecisionTree / RandomForest)
        ├── bar_predictions_2026(pred_df, student_id)
        ├── scatter_actual_vs_predicted for selected subject
        └── Model evaluation table (RMSE, MAE, CV score per model/subject)

Main area — VLE Pass/Fail view                           ← YOU build this entire view
  ├── [Tab 1] Dataset Overview
  │     ├── KPI cards: total students, pass rate, fail rate
  │     ├── bar_vle_class_distribution chart
  │     └── heatmap_vle_correlation chart
  │
  ├── [Tab 2] At-Risk Predictor
  │     ├── Input sliders for all 19 VLE features
  │     │     → use st.slider / st.number_input per feature
  │     ├── st.button("Predict") → calls predict_pass_fail()
  │     ├── Display: "PASS ✓" or "FAIL ✗" + fail probability gauge
  │     └── bar_feature_importance chart (Random Forest)
  │
  └── [Tab 3] Classifier Evaluation
        ├── plot_confusion_matrix (Logistic Regression vs Random Forest)
        ├── plot_roc_curve for both models
        └── Metrics table (accuracy, F1, ROC-AUC, recall_fail)
```

Dev1 owns Tab 1 (Overview), Tab 2 (Student Profile), and Tab 4 (Export) in the Marks view.

### Useful Streamlit patterns

```python
import streamlit as st
from models.predict import predict_2026, predict_pass_fail
from models.evaluate import evaluate_all_models, evaluate_all_classifiers
from visualisation.charts import (
    bar_predictions_2026, scatter_actual_vs_predicted,
    bar_vle_class_distribution, heatmap_vle_correlation,
    bar_feature_importance, plot_roc_curve, plot_confusion_matrix,
)
from database.db_utils import get_vle_data, get_student_marks

# Tab 3 — Predictions example
with tab3:
    model_name = st.sidebar.selectbox("Regression Model",
                    ["LinearRegression", "DecisionTree", "RandomForest"])
    student_id = st.sidebar.selectbox("Student", student_list)
    marks_2025 = get_student_marks(student_id).query("year == 2025")
    # ... build feature dict, call predict_2026(), plot results

# VLE At-Risk Predictor example
with vle_tab2:
    st.subheader("Enter Student VLE Features")
    logins = st.slider("Login Count", 0, 200, 20)
    total_hours = st.number_input("Total Hours in Module", 0.0, 500.0, 10.0)
    # ... collect all 19 features
    if st.button("Predict Pass/Fail"):
        label, prob = predict_pass_fail(feature_row, classifier_name)
        if label == 0:
            st.success(f"Prediction: PASS (fail probability: {prob:.1%})")
        else:
            st.error(f"Prediction: FAIL (fail probability: {prob:.1%})")
```

---

## Data Schema Reference

### Synthetic Cohort — Wide-format CSV (`students_clean.csv`)

| student_id | name | year | Math | Physics | CS | English | Statistics |
|---|---|---|---|---|---|---|---|
| S001 | Ali | 2024 | 78.0 | 65.0 | 82.0 | 71.0 | 69.0 |
| S001 | Ali | 2025 | 81.0 | 68.0 | 85.0 | 74.0 | 72.0 |

### VLE CSV columns (raw headers → after renaming)

The raw CSV from `data/vle/` has verbose headers. Use `database/db_utils.py`'s `get_vle_data()` which returns already-renamed columns:

`gender, age, logins, total_hours, pct_avg_hours, presence_count, absence_count, pct_attended, attending_from_home, distance_to_uni_km, polar4_quintile, polar3_quintile, adult_he_2001_quintile, adult_he_2011_quintile, tundra_msoa_quintile, tundra_lsoa_quintile, gaps_gcse_quintile, gaps_gcse_ethnicity_quintile, uni_connect_target_ward, label, split`

---

## Commit & PR Guidelines

### Commit message format

```
<type>(<scope>): <short description>

Types: feat | fix | docs | test | refactor | chore
Scope for Dev2: viz | models | eval | dashboard

Examples:
feat(viz): add heatmap_vle_correlation chart function
feat(models): train random forest classifier on SMOTE data
fix(eval): fix recall_fail metric for imbalanced classes
test(models): add test_classifier_recall_fail_nonzero
feat(dashboard): wire VLE at-risk predictor tab
```

### PR rules
- Title must follow the format above.
- Every PR needs Dev1 as reviewer.
- PR description must say: what changed + how to test it locally.
- `pytest tests/` must pass before requesting review.

---

## Your Testing Checklist

Run from the project root:

```bash
pytest tests/test_models.py tests/test_predict.py tests/test_vle.py -v
```

### tests/test_models.py
- [ ] `test_model_trains_without_error` — all 3 regression models fit without exception
- [ ] `test_pkl_files_saved` — `saved_models/LinearRegression_Math.pkl` (and others) exist after training
- [ ] `test_rmse_reasonable` — RMSE < 20 for all regression models (sanity check)
- [ ] `test_classifier_trains_without_error` — both Logistic Regression and Random Forest classifiers fit without exception
- [ ] `test_classifier_pkl_saved` — `saved_models/classifier_LogisticRegression.pkl` and `classifier_RandomForest.pkl` exist

### tests/test_predict.py
- [ ] `test_predict_returns_five_subjects` — `predict_2026()` output dict has exactly 5 keys
- [ ] `test_predicted_marks_in_range` — all values in the output of `predict_2026()` are in [0, 100]
- [ ] `test_predict_pass_fail_output_type` — `predict_pass_fail()` returns a tuple of `(int, float)`
- [ ] `test_predict_pass_fail_probability_range` — fail probability is in [0.0, 1.0]

### tests/test_vle.py
- [ ] `test_vle_seed_row_count` — `vle_students` table has 160 rows after seeding
- [ ] `test_vle_class_distribution` — 136 pass (0), 24 fail (1) in the seeded data
- [ ] `test_classifier_recall_fail_nonzero` — recall for class 1 (fail) > 0 — model does not ignore minority class
- [ ] `test_evaluate_all_classifiers_columns` — output DataFrame has columns: model, accuracy, f1_macro, roc_auc, recall_fail

---

## Your Future Enhancements

After the initial version ships, these improvements are assigned to you:

### High priority
| # | Enhancement |
|---|---|
| 2 | **Neural Network model** — add `MLPRegressor` from scikit-learn as a 4th regression model. Train per-subject, save `.pkl`, include in `evaluate_all_models()`. |
| 3 | **Hyperparameter tuning** — wrap `DecisionTreeRegressor` and `RandomForestRegressor` with `GridSearchCV`. Log best params to a JSON file. |
| 4 | **Confidence intervals** — for the Random Forest regressor, use per-tree predictions (`estimators_`) to produce a 95% interval. Display as error bars in `bar_predictions_2026()`. |

### Medium priority
| # | Enhancement |
|---|---|
| 7 | **Grade classification** — alongside the numeric 2026 prediction, add a classifier that outputs a letter grade (A/B/C/D/F). Train using binned 2025 marks as targets. |
| 9 | **Docker containerisation** — write `Dockerfile` and `docker-compose.yml` (coordinate with Dev1). |
| 10 | **Scheduled re-training** — use `APScheduler` within the Streamlit app to auto-retrain models whenever new marks are inserted into the DB. |

### Lower priority
| # | Enhancement |
|---|---|
| 12 | Add dark/light mode toggle via `st.set_page_config(theme=...)` and a sidebar toggle. |
| 13 | Add an Arabic language toggle to the dashboard using a dict-based translation layer. |
| 15 | Set up GitHub Actions CI — create `.github/workflows/ci.yml` to run `pytest` on every push (YAML below). |

### GitHub Actions starter (Enhancement #15)

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main, dev]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v
```

---

## Submission Checklist

Verify every item before the final PR to `main`:

- [ ] `python models/train.py` runs without errors and saves all `.pkl` files
- [ ] `pytest tests/test_models.py tests/test_predict.py tests/test_vle.py -v` — 0 failures
- [ ] `pytest tests/ -v` — 0 failures (all tests including Dev1's)
- [ ] All chart PNGs saved to `outputs/charts/` before the PDF export tab is used
- [ ] `streamlit run dashboard/app.py` — all your tabs render without errors
- [ ] VLE At-Risk Predictor returns a prediction for valid feature inputs
- [ ] `outputs/eval_results.csv` exists for Dev1 to embed in the PDF
- [ ] No hardcoded absolute paths — use `pathlib.Path(__file__).parent` for relative paths
- [ ] No credentials or personal data committed
- [ ] All feature branches merged into `dev` before merging `dev` → `main`

---

*Last updated: April 26, 2026*
