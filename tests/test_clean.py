"""
tests/test_clean.py
===================
Unit tests for preprocessing/clean.py functions.
"""

import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest

from preprocessing.clean import (
    SUBJECTS,
    clean_missing,
    clip_outliers,
    encode_year,
    load_raw_data,
    normalise,
    run_pipeline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Small wide-format DataFrame with known values."""
    data = {
        "student_id": ["S001", "S001", "S002", "S002"],
        "name": ["Ali", "Ali", "Sara", "Sara"],
        "year": [2024, 2025, 2024, 2025],
        "Math": [70.0, 80.0, 60.0, 65.0],
        "Physics": [np.nan, 75.0, 55.0, 60.0],
        "CS": [90.0, 85.0, 50.0, 55.0],
        "English": [65.0, 70.0, np.nan, 72.0],
        "Statistics": [72.0, 78.0, 58.0, 62.0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def outlier_df():
    data = {
        "student_id": ["S001"],
        "name": ["Ali"],
        "year": [2024],
        "Math": [-5.0],
        "Physics": [110.0],
        "CS": [50.0],
        "English": [75.0],
        "Statistics": [80.0],
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_missing_values_filled(sample_df):
    """No NaNs remain in subject columns after clean_missing()."""
    result = clean_missing(sample_df)
    assert result[SUBJECTS].isna().sum().sum() == 0


def test_outliers_clipped(outlier_df):
    """No values outside [0, 100] after clip_outliers()."""
    result = clip_outliers(outlier_df)
    assert result[SUBJECTS].min().min() >= 0
    assert result[SUBJECTS].max().max() <= 100


def test_normalise_range(sample_df):
    """All *_norm columns are within [0, 1] after normalise()."""
    cleaned = clean_missing(sample_df)
    normed = normalise(cleaned, fit=True)
    norm_cols = [f"{s}_norm" for s in SUBJECTS]
    assert all(c in normed.columns for c in norm_cols)
    assert normed[norm_cols].min().min() >= -1e-9
    assert normed[norm_cols].max().max() <= 1 + 1e-9


def test_run_pipeline_creates_csv(tmp_path, sample_df):
    """run_pipeline() saves students_clean.csv to the given path."""
    raw_path = tmp_path / "raw.csv"
    clean_path = tmp_path / "clean.csv"
    sample_df.to_csv(raw_path, index=False)
    run_pipeline(str(raw_path), str(clean_path))
    assert clean_path.exists()
    loaded = pd.read_csv(clean_path)
    assert len(loaded) == len(sample_df)


def test_year_encoding(sample_df):
    """year_encoded column contains only 0 and 1."""
    cleaned = clean_missing(sample_df)
    encoded = encode_year(cleaned)
    assert "year_encoded" in encoded.columns
    assert set(encoded["year_encoded"].unique()).issubset({0, 1})
