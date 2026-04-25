# AI-Powered Student Performance Analytics System
### Project Instruction Sheet — Team of 2

**Repository:** https://github.com/Faris075/software-testing-project/  
**Project Date:** 2026  
**Dataset Scope:** 15 students × 5 subjects × 2 years (2024–2025) with 2026 predictions

---

## Table of Contents

- [AI-Powered Student Performance Analytics System](#ai-powered-student-performance-analytics-system)
    - [Project Instruction Sheet — Team of 2](#project-instruction-sheet--team-of-2)
  - [Table of Contents](#table-of-contents)
  - [1. Project Overview](#1-project-overview)
  - [2. Tech Stack](#2-tech-stack)
  - [3. Team Role Split](#3-team-role-split)
    - [Dev1 — Data, Backend \& Reports](#dev1--data-backend--reports)
    - [Dev2 — ML, Visualisation \& Dashboard](#dev2--ml-visualisation--dashboard)
  - [4. Repository Setup \& Git Workflow](#4-repository-setup--git-workflow)
    - [4.1 Clone \& initialise](#41-clone--initialise)
    - [4.2 Branch strategy](#42-branch-strategy)
    - [4.3 Recommended daily workflow](#43-recommended-daily-workflow)
    - [4.4 Merging to main](#44-merging-to-main)
  - [5. Project Structure](#5-project-structure)
  - [6. Environment Setup](#6-environment-setup)
    - [6.1 requirements.txt (create this at project root)](#61-requirementstxt-create-this-at-project-root)
    - [6.2 .gitignore (create this at project root)](#62-gitignore-create-this-at-project-root)
  - [7. Phase-by-Phase Implementation Guide](#7-phase-by-phase-implementation-guide)
    - [Phase 1 — Database \& Data Layer *(Dev1)*](#phase-1--database--data-layer-dev1)
      - [7.1.1 SQLite Schema (`database/schema.sql`)](#711-sqlite-schema-databaseschemasql)
      - [7.1.2 Seed data (`database/seed.py`)](#712-seed-data-databaseseedpy)
      - [7.1.3 Query helpers (`database/db_utils.py`)](#713-query-helpers-databasedb_utilspy)
    - [Phase 2 — Data Cleaning \& Preprocessing *(Dev1)*](#phase-2--data-cleaning--preprocessing-dev1)
      - [Steps to implement:](#steps-to-implement)
    - [Phase 3 — Statistical Analysis *(Dev1)*](#phase-3--statistical-analysis-dev1)
      - [Functions to implement:](#functions-to-implement)
    - [Phase 4 — Data Visualization *(Dev2)*](#phase-4--data-visualization-dev2)
      - [Charts to implement:](#charts-to-implement)
    - [Phase 5 — AI/ML Prediction Models *(Dev2)*](#phase-5--aiml-prediction-models-dev2)
      - [Feature engineering](#feature-engineering)
      - [Models](#models)
    - [Phase 6 — Model Evaluation *(Dev2)*](#phase-6--model-evaluation-dev2)
      - [Metrics to compute](#metrics-to-compute)
    - [Phase 7 — Streamlit Dashboard *(Dev1 \& Dev2)*](#phase-7--streamlit-dashboard-dev1--dev2)
      - [To run:](#to-run)
      - [Dashboard layout](#dashboard-layout)
    - [Phase 8 — PDF Report Generation *(Dev1)*](#phase-8--pdf-report-generation-dev1)
      - [PDF content (pages):](#pdf-content-pages)
  - [8. Data Schema](#8-data-schema)
    - [Wide-format CSV (`students_clean.csv`)](#wide-format-csv-students_cleancsv)
    - [Marks generation guidance (for `seed.py`)](#marks-generation-guidance-for-seedpy)
  - [9. Commit \& PR Guidelines](#9-commit--pr-guidelines)
    - [Commit message format](#commit-message-format)
    - [PR rules](#pr-rules)
  - [10. Testing Checklist](#10-testing-checklist)
    - [tests/test\_clean.py (Dev1)](#teststest_cleanpy-dev1)
    - [tests/test\_stats.py (Dev1)](#teststest_statspy-dev1)
    - [tests/test\_models.py (Dev2)](#teststest_modelspy-dev2)
    - [tests/test\_predict.py (Dev2)](#teststest_predictpy-dev2)
  - [11. Future Modifications \& Enhancements](#11-future-modifications--enhancements)
    - [High priority (implement next sprint)](#high-priority-implement-next-sprint)
    - [Medium priority](#medium-priority)
    - [Lower priority (polish)](#lower-priority-polish)
    - [GitHub Actions starter (for enhancement #15)](#github-actions-starter-for-enhancement-15)
  - [12. Submission Checklist](#12-submission-checklist)

---

## 1. Project Overview

This system ingests historical student marks for the years 2024 and 2025, cleans and analyses that data, trains machine learning models to predict 2026 marks, and surfaces every insight through an interactive Streamlit dashboard and an exported PDF report.

**Core data facts:**
- **Students:** 15 (IDs: `S001` – `S015`)
- **Subjects:** Math, Physics, CS, English, Statistics
- **Years:** 2024, 2025 (historical) → 2026 (predicted)
- **Mark range:** 0–100

---

## 2. Tech Stack

| Category | Library / Tool |
|---|---|
| Data handling | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Machine learning | `scikit-learn` |
| Statistical analysis | `scipy` |
| Interactive dashboard | `streamlit` |
| Database | `sqlite3` (stdlib) + `sqlalchemy` |
| Report generation | `reportlab` |
| Environment | `python 3.10+`, `pip`, `venv` |
| Version control | `git`, GitHub |

---

## 3. Team Role Split

### Dev1 — Data, Backend & Reports
Owns everything that touches raw data, the database, preprocessing, statistics, and the PDF export.

| Task | Phase |
|---|---|
| Create SQLite schema and seed data | Phase 1 |
| Write data cleaning / preprocessing pipeline | Phase 2 |
| Perform statistical analysis (mean, median, std, correlation) | Phase 3 |
| Build PDF report generator | Phase 8 |
| Write unit tests for data functions | Phase 10 |
| Review Dev2's PRs related to data access in the dashboard | Ongoing |

### Dev2 — ML, Visualisation & Dashboard
Owns the ML pipeline, all charts, the Streamlit UI, and model evaluation.

| Task | Phase |
|---|---|
| Build visualisation module (bar, line, comparison charts) | Phase 4 |
| Implement Linear Regression, Decision Tree, Random Forest models | Phase 5 |
| Evaluate models (RMSE, MAE, cross-validation) | Phase 6 |
| Build Streamlit dashboard and wire up all components | Phase 7 |
| Write unit tests for ML functions | Phase 10 |
| Review Dev1's PRs related to data that feeds the models | Ongoing |

> Both Dev1 and Dev2 merge their features into `main` via Pull Requests — never push directly to `main`.

---

## 4. Repository Setup & Git Workflow

### 4.1 Clone & initialise

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

### 4.2 Branch strategy

```
main           ← protected; only accepts PRs
├── dev        ← shared integration branch
│   ├── feature/dev1-database          (Dev1)
│   ├── feature/dev1-preprocessing     (Dev1)
│   ├── feature/dev1-statistics        (Dev1)
│   ├── feature/dev1-pdf-report        (Dev1)
│   ├── feature/dev2-visualisation     (Dev2)
│   ├── feature/dev2-ml-models         (Dev2)
│   ├── feature/dev2-evaluation        (Dev2)
│   └── feature/dashboard              (Dev1 & Dev2)
```

### 4.3 Recommended daily workflow

```bash
# Start of day — sync dev
git checkout dev
git pull origin dev

# Create or switch to your feature branch
git checkout -b feature/your-feature   # first time
git checkout feature/your-feature      # subsequent times

# Work, then stage and commit
git add .
git commit -m "feat(scope): short description"

# Push and open PR into dev (not main)
git push origin feature/your-feature
# Open PR on GitHub: base=dev, compare=your-feature
```

### 4.4 Merging to main

Once all features are on `dev` and tests pass:

```bash
git checkout main
git merge --no-ff dev -m "release: v1.0.0 - initial system"
git push origin main
```

---

## 5. Project Structure

```
software-testing-project/
├── data/
│   ├── students_raw.csv          # original seeded data
│   └── students_clean.csv        # output of preprocessing
├── database/
│   ├── schema.sql                # table definitions
│   ├── seed.py                   # populates SQLite from CSV
│   └── db_utils.py               # query helpers (SQLAlchemy)
├── preprocessing/
│   └── clean.py                  # missing value handling, normalisation
├── analysis/
│   └── stats.py                  # mean, median, std, correlation
├── visualisation/
│   └── charts.py                 # all matplotlib / seaborn chart functions
├── models/
│   ├── train.py                  # fit all 3 models + save .pkl files
│   ├── predict.py                # load .pkl and produce 2026 predictions
│   └── evaluate.py               # RMSE, MAE, cross-validation
├── saved_models/
│   ├── linear_regression.pkl
│   ├── decision_tree.pkl
│   └── random_forest.pkl
├── report/
│   └── generate_pdf.py           # ReportLab PDF builder
├── dashboard/
│   └── app.py                    # Streamlit entry point
├── tests/
│   ├── test_clean.py
│   ├── test_stats.py
│   ├── test_models.py
│   └── test_predict.py
├── outputs/
│   ├── charts/                   # saved chart PNGs
│   └── reports/                  # generated PDFs
├── requirements.txt
├── .gitignore
├── README.md
└── INSTRUCTIONS.md               # ← this file
```

---

## 6. Environment Setup

### 6.1 requirements.txt (create this at project root)

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

### 6.2 .gitignore (create this at project root)

```
.venv/
__pycache__/
*.pyc
*.pyo
.DS_Store
outputs/
saved_models/*.pkl
database/students.db
*.egg-info/
.env
```

> Note: `outputs/` and `.pkl` files are generated artifacts — they should not be committed. The `outputs/` folder should be created locally by running the pipeline.

---

## 7. Phase-by-Phase Implementation Guide

---

### Phase 1 — Database & Data Layer *(Dev1)*

**Goal:** Persist all student data in SQLite; expose a clean query API.

#### 7.1.1 SQLite Schema (`database/schema.sql`)

```sql
CREATE TABLE IF NOT EXISTS students (
    student_id   TEXT PRIMARY KEY,   -- e.g. S001
    name         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS marks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id   TEXT NOT NULL REFERENCES students(student_id),
    year         INTEGER NOT NULL,   -- 2024 or 2025
    subject      TEXT NOT NULL,      -- Math | Physics | CS | English | Statistics
    mark         REAL                -- NULL if missing
);

CREATE TABLE IF NOT EXISTS predictions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id   TEXT NOT NULL REFERENCES students(student_id),
    subject      TEXT NOT NULL,
    predicted_mark REAL,
    model_used   TEXT,               -- LinearRegression | DecisionTree | RandomForest
    created_at   TEXT DEFAULT (datetime('now'))
);
```

#### 7.1.2 Seed data (`database/seed.py`)

- Hard-code (or generate randomly with `numpy`) marks for 15 students across 5 subjects for 2024 and 2025.
- Introduce **5–8 intentional missing values** across the dataset so the preprocessing step has something to handle.
- Write the data to `data/students_raw.csv` as well as inserting into SQLite.

Example student names: Ali, Sara, Omar, Lena, Tariq, Maya, Khalid, Nour, Reem, Jad, Hana, Zaid, Dina, Faris, Layla.

#### 7.1.3 Query helpers (`database/db_utils.py`)

```python
# Key functions to implement:
def get_all_marks() -> pd.DataFrame: ...          # returns full marks table
def get_student_marks(student_id: str) -> pd.DataFrame: ...
def save_predictions(predictions_df: pd.DataFrame): ...
def get_predictions(student_id: str) -> pd.DataFrame: ...
```

Use `sqlalchemy.create_engine("sqlite:///database/students.db")` with pandas `read_sql`.

---

### Phase 2 — Data Cleaning & Preprocessing *(Dev1)*

**Goal:** Produce a clean, normalised dataset ready for ML.

**File:** `preprocessing/clean.py`

#### Steps to implement:

1. **Load raw data** from `data/students_raw.csv`.

2. **Handle missing values:**
   - For numeric mark columns: fill with the **column mean** (per subject per year).
   - Log which cells were imputed so the report can mention it.

3. **Detect and cap outliers:**
   - Any mark outside `[0, 100]` → clip to boundary.

4. **Normalise marks** (for ML input features):
   - Use `sklearn.preprocessing.MinMaxScaler` → scale to `[0, 1]`.
   - Save the fitted scaler as `saved_models/scaler.pkl` (using `joblib.dump`).

5. **Encode year** as integer feature (2024 → 0, 2025 → 1).

6. **Save cleaned data** to `data/students_clean.csv`.

```python
# Key function signatures:
def load_raw_data(path: str) -> pd.DataFrame: ...
def clean_missing(df: pd.DataFrame) -> pd.DataFrame: ...
def clip_outliers(df: pd.DataFrame) -> pd.DataFrame: ...
def normalise(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame: ...
def run_pipeline(raw_path: str, clean_path: str) -> pd.DataFrame: ...
```

---

### Phase 3 — Statistical Analysis *(Dev1)*

**Goal:** Surface key statistical insights used by both the dashboard and the report.

**File:** `analysis/stats.py`

#### Functions to implement:

```python
def subject_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Returns mean, median, std deviation per subject across all students and years."""

def student_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Returns mean mark per student across all subjects per year."""

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between subjects (ignore year/student columns)."""

def top_performers(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Returns top-n students by overall average."""

def failing_students(df: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
    """Returns students whose average in any subject falls below threshold."""

def year_on_year_change(df: pd.DataFrame) -> pd.DataFrame:
    """Returns per-student per-subject delta between 2024 and 2025."""
```

All functions take the cleaned DataFrame (wide format: columns = `[student_id, name, year, Math, Physics, CS, English, Statistics]`).

---

### Phase 4 — Data Visualization *(Dev2)*

**Goal:** Produce reusable chart functions callable from both the dashboard and the PDF report.

**File:** `visualisation/charts.py`

#### Charts to implement:

| Function | Chart type | Description |
|---|---|---|
| `bar_subject_averages(df)` | Bar | Average mark per subject (all students combined) |
| `bar_student_averages(df, year)` | Bar | Average mark per student for a given year |
| `line_progress(df, student_id)` | Line | 2024 → 2025 mark progression per subject for one student |
| `line_class_progress(df)` | Line | Class average per subject: 2024 vs 2025 |
| `comparison_bar(df, student_id)` | Grouped bar | Side-by-side 2024 vs 2025 for one student |
| `heatmap_correlation(corr_matrix)` | Heatmap | Seaborn heatmap of subject correlations |
| `scatter_actual_vs_predicted(y_true, y_pred, subject)` | Scatter | Actual vs predicted marks for a subject |
| `bar_predictions_2026(pred_df, student_id)` | Bar | Predicted 2026 marks per subject for one student |

**Guidelines:**
- Every function should accept an optional `ax=None` parameter so it can be embedded in Streamlit (`st.pyplot`) or saved to file.
- When `ax is None`, create a new `fig, ax = plt.subplots(...)` and call `fig.savefig(f"outputs/charts/{name}.png", bbox_inches='tight')`.
- Use a consistent seaborn theme: `sns.set_theme(style="whitegrid", palette="muted")` at module level.

---

### Phase 5 — AI/ML Prediction Models *(Dev2)*

**Goal:** Train three models to predict a student's mark in a subject given year + prior marks as features.

**File:** `models/train.py`

#### Feature engineering

Build training data from the cleaned CSV:
- **Features (X):** `[year_encoded, Math_2024, Physics_2024, CS_2024, English_2024, Statistics_2024]`
- **Target (y):** mark in the target subject for a given year
- Train on **2024 data** to predict **2025**; evaluate on 2025; then retrain on all data (2024+2025) to predict **2026**.

#### Models

```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

MODELS = {
    "LinearRegression": LinearRegression(),
    "DecisionTree":     DecisionTreeRegressor(max_depth=4, random_state=42),
    "RandomForest":     RandomForestRegressor(n_estimators=100, random_state=42),
}
```

For each subject, train all three models and save with:

```python
import joblib
joblib.dump(model, f"saved_models/{model_name}_{subject}.pkl")
```

**File:** `models/predict.py`

```python
def predict_2026(student_features: dict, model_name: str) -> dict:
    """
    Given a dict of {subject: mark} for 2025 and the model name,
    returns a dict of {subject: predicted_2026_mark}.
    """
```

---

### Phase 6 — Model Evaluation *(Dev2)*

**Goal:** Quantify model quality and select the best model per subject.

**File:** `models/evaluate.py`

#### Metrics to compute

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import numpy as np

def rmse(y_true, y_pred) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred) -> float:
    return mean_absolute_error(y_true, y_pred)

def cross_validate_model(model, X, y, cv: int = 5) -> dict:
    """Returns mean and std of negative RMSE across k folds."""

def evaluate_all_models(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
    [model, subject, RMSE, MAE, CV_RMSE_mean, CV_RMSE_std]
    Useful for displaying in the dashboard and including in the PDF.
    """
```

Print a summary table at the end of the script showing which model performs best per subject.

---

### Phase 7 — Streamlit Dashboard *(Dev1 & Dev2)*

**Goal:** A polished interactive web UI that wraps everything built in Phases 1–6.

**File:** `dashboard/app.py`

#### To run:

```bash
streamlit run dashboard/app.py
```

#### Dashboard layout

```
Sidebar
  ├── Student selector (selectbox)
  ├── Year selector (2024 / 2025 / 2026 prediction)
  └── Model selector (LinearRegression / DecisionTree / RandomForest)

Main area
  ├── [Tab 1] Overview
  │     ├── KPI cards: class average, top student, most-improved student
  │     ├── bar_subject_averages chart
  │     └── heatmap_correlation chart
  │
  ├── [Tab 2] Student Profile
  │     ├── Student name + stats table (mean, best subject, worst subject)
  │     ├── comparison_bar (2024 vs 2025)
  │     ├── line_progress chart
  │     └── failing alert (red warning if any subject < 50)
  │
  ├── [Tab 3] 2026 Predictions
  │     ├── bar_predictions_2026 (per selected student + model)
  │     ├── scatter_actual_vs_predicted
  │     └── Model evaluation table (RMSE, MAE, CV score)
  │
  └── [Tab 4] Export
        ├── "Generate PDF Report" button → calls generate_pdf.py
        └── Download link for the PDF
```

**Key implementation note:** Import and call functions from `database/db_utils.py`, `analysis/stats.py`, `visualisation/charts.py`, `models/predict.py`, and `report/generate_pdf.py`. The dashboard should not contain business logic — only UI wiring.

---

### Phase 8 — PDF Report Generation *(Dev1)*

**Goal:** Export a professional multi-page PDF summarising the entire analysis.

**File:** `report/generate_pdf.py`

#### PDF content (pages):

| Page | Content |
|---|---|
| 1 | Title page: project name, date, number of students |
| 2 | Executive summary: class statistics table (mean, median, std per subject) |
| 3 | Top 3 performers + failing students alert table |
| 4 | Bar chart — subject averages (embed PNG from outputs/charts/) |
| 5 | Correlation heatmap (embed PNG) |
| 6–20 | One page per student: name, 2024 marks, 2025 marks, YoY delta, 2026 prediction (best model) |
| Last | Model evaluation summary (RMSE / MAE table for all 3 models) |

```python
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_report(output_path: str = "outputs/reports/report.pdf"): ...
```

---

## 8. Data Schema

### Wide-format CSV (`students_clean.csv`)

| student_id | name | year | Math | Physics | CS | English | Statistics |
|---|---|---|---|---|---|---|---|
| S001 | Ali | 2024 | 78.0 | 65.0 | 82.0 | 71.0 | 69.0 |
| S001 | Ali | 2025 | 81.0 | 68.0 | 85.0 | 74.0 | 72.0 |
| … | … | … | … | … | … | … | … |

### Marks generation guidance (for `seed.py`)

```python
import numpy as np

np.random.seed(42)
base_marks = np.random.randint(50, 95, size=(15, 5))  # 2024 base
# 2025: small improvement or decline
marks_2025 = np.clip(base_marks + np.random.randint(-8, 12, size=(15, 5)), 0, 100)
# Introduce ~6 missing values
missing_indices = [(2,1), (5,3), (7,0), (9,4), (11,2), (13,1)]
for r, c in missing_indices:
    base_marks[r, c] = np.nan
```

---

## 9. Commit & PR Guidelines

### Commit message format

```
<type>(<scope>): <short description>

Types: feat | fix | docs | test | refactor | chore
Scope: db | clean | stats | viz | models | eval | dashboard | report

Examples:
feat(db): add seed script with 15 students
feat(models): implement random forest per-subject training
fix(clean): handle NaN in statistics column
test(models): add cross-validation unit tests
docs(dashboard): update README with run instructions
```

### PR rules
- Title must follow the commit format above.
- Every PR must have at least **1 reviewer** (the other team member).
- PR must include a short description: what was done, how to test it.
- No PR merges to `dev` until:
  - `pytest tests/` passes locally
  - No unresolved review comments

---

## 10. Testing Checklist

Run all tests from the project root:

```bash
pytest tests/ -v
```

### tests/test_clean.py (Dev1)
- [ ] `test_missing_values_filled` — no NaNs after `clean_missing()`
- [ ] `test_outliers_clipped` — no values outside [0, 100] after `clip_outliers()`
- [ ] `test_normalise_range` — all normalised values in [0, 1]

### tests/test_stats.py (Dev1)
- [ ] `test_subject_summary_columns` — output has expected columns
- [ ] `test_top_performers_count` — returns correct number of students
- [ ] `test_year_on_year_shape` — delta DataFrame has expected shape

### tests/test_models.py (Dev2)
- [ ] `test_model_trains_without_error` — all 3 models fit without exception
- [ ] `test_pkl_files_saved` — `.pkl` files exist after training
- [ ] `test_rmse_reasonable` — RMSE < 20 for all models (sanity check)

### tests/test_predict.py (Dev2)
- [ ] `test_predict_returns_five_subjects` — output dict has 5 keys
- [ ] `test_predicted_marks_in_range` — all predicted values in [0, 100]

---

## 11. Future Modifications & Enhancements

These are recommended improvements to implement after the initial version is complete. They are prioritised by impact.

### High priority (implement next sprint)

| # | Enhancement | Owner suggestion |
|---|---|---|
| 1 | **Add more years of data (2022, 2023)** to give ML models more training samples and improve prediction accuracy. | Dev1 |
| 2 | **Neural Network model** — add a `MLPRegressor` from scikit-learn or a simple Keras model as a 4th predictor. | Dev2 |
| 3 | **Hyperparameter tuning** — use `GridSearchCV` or `RandomizedSearchCV` on Decision Tree and Random Forest. | Dev2 |
| 4 | **Confidence intervals** on predictions — use Random Forest's per-tree variance to display a prediction range (e.g. 78 ± 4). | Dev2 |
| 5 | **User authentication** in Streamlit — add a simple login page so only authorised users see student data. | Dev1 |

### Medium priority

| # | Enhancement | Owner suggestion |
|---|---|---|
| 6 | **CSV / Excel upload** in the dashboard so teachers can add new student cohorts without editing code. | Dev1 |
| 7 | **Grade classification** — alongside the numeric prediction, predict a letter grade (A/B/C/D/F) using a classifier. | Dev2 |
| 8 | **Email alerts** — send automated email via `smtplib` to students predicted to fail a subject. | Dev1 |
| 9 | **Docker containerisation** — add a `Dockerfile` and `docker-compose.yml` so the app runs identically on any machine. | Dev1 & Dev2 |
| 10 | **Scheduled re-training** — use `APScheduler` or a cron job to retrain models whenever new data is inserted into SQLite. | Dev2 |

### Lower priority (polish)

| # | Enhancement |
|---|---|
| 11 | Replace SQLite with PostgreSQL for concurrent multi-user production use. |
| 12 | Add dark/light mode toggle to the Streamlit dashboard via `st.set_page_config`. |
| 13 | Internationalise (i18n) the dashboard — support Arabic and English using a language toggle. |
| 14 | Add a `logging` module throughout so errors are recorded to a rotating log file instead of crashing silently. |
| 15 | CI/CD pipeline via GitHub Actions — auto-run `pytest` on every push to `dev` and `main`. |

### GitHub Actions starter (for enhancement #15)

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

## 12. Submission Checklist

Before the final push to `main`, verify every item below:

- [ ] All feature branches merged into `dev`, then `dev` merged into `main`
- [ ] `pytest tests/ -v` passes with 0 failures
- [ ] `streamlit run dashboard/app.py` launches without errors
- [ ] PDF report generates to `outputs/reports/report.pdf`
- [ ] `data/students_raw.csv` and `data/students_clean.csv` present in repo
- [ ] `requirements.txt` is up to date (`pip freeze > requirements.txt`)
- [ ] `README.md` includes: project description, setup steps, how to run dashboard, how to run tests
- [ ] `.gitignore` excludes `.venv/`, `__pycache__/`, `.pkl` files, `outputs/`, `*.db`
- [ ] No hardcoded absolute paths anywhere in the codebase — use `pathlib.Path` relative to project root
- [ ] No credentials, API keys, or personal data committed (check with `git log --all -p | grep -i password`)
- [ ] All chart outputs saved to `outputs/charts/` before PDF generation is called

---

*Last updated: April 26, 2026*
