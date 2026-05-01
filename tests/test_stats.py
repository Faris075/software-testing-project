"""
tests/test_stats.py
===================
Unit tests for analysis/stats.py functions.
"""

import pandas as pd
import pytest

from analysis.stats import (
    SUBJECTS,
    correlation_matrix,
    failing_students,
    student_summary,
    subject_summary,
    top_performers,
    year_on_year_change,
)


# ---------------------------------------------------------------------------
# Fixture — 15-student, 2-year wide-format DataFrame
# ---------------------------------------------------------------------------

STUDENT_NAMES = [
    "Ali", "Sara", "Omar", "Lena", "Tariq",
    "Maya", "Khalid", "Nour", "Reem", "Jad",
    "Hana", "Zaid", "Dina", "Faris", "Layla",
]


@pytest.fixture
def cohort_df():
    import numpy as np

    rng = np.random.default_rng(42)
    rows = []
    for i, name in enumerate(STUDENT_NAMES):
        sid = f"S{i+1:03d}"
        for year in [2024, 2025]:
            row = {"student_id": sid, "name": name, "year": year}
            for subj in SUBJECTS:
                row[subj] = float(rng.integers(50, 96))
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def failing_df():
    """DataFrame with one student who clearly fails a subject."""
    rows = []
    for i, name in enumerate(STUDENT_NAMES):
        sid = f"S{i+1:03d}"
        for year in [2024, 2025]:
            row = {"student_id": sid, "name": name, "year": year}
            for subj in SUBJECTS:
                row[subj] = 80.0
            rows.append(row)
    # Set S001 Math to 30 (below threshold) in both years
    for row in rows:
        if row["student_id"] == "S001":
            row["Math"] = 30.0
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_subject_summary_columns(cohort_df):
    """subject_summary() result has columns: subject, mean, median, std."""
    result = subject_summary(cohort_df)
    assert set(result.columns) == {"subject", "mean", "median", "std"}
    assert len(result) == len(SUBJECTS)


def test_top_performers_count(cohort_df):
    """top_performers() returns exactly n rows for any n <= 15."""
    for n in [1, 3, 5]:
        result = top_performers(cohort_df, n=n)
        assert len(result) == n, f"Expected {n} rows, got {len(result)}"


def test_year_on_year_shape(cohort_df):
    """year_on_year_change() returns one row per student (15 rows)."""
    result = year_on_year_change(cohort_df)
    assert len(result) == 15


def test_failing_students_threshold(failing_df):
    """failing_students() returns S001 when threshold=60 (Math avg = 30)."""
    result = failing_students(failing_df, threshold=60.0)
    assert not result.empty
    assert "S001" in result["student_id"].values
    failing_subjects = result[result["student_id"] == "S001"]["subject"].tolist()
    assert "Math" in failing_subjects


def test_correlation_matrix_shape(cohort_df):
    """correlation_matrix() returns a 5×5 DataFrame."""
    result = correlation_matrix(cohort_df)
    assert result.shape == (5, 5)
    assert list(result.columns) == SUBJECTS
    assert list(result.index) == SUBJECTS
