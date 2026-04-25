"""
database/db_utils.py
====================
Query helpers for the SQLite database.
Uses SQLAlchemy for connection management and pandas for result sets.

Usage:
    from database.db_utils import get_all_marks, get_vle_data, ...
"""

import pathlib
import pandas as pd
from sqlalchemy import create_engine, text

ROOT    = pathlib.Path(__file__).parent.parent
DB_PATH = ROOT / "database" / "students.db"

# ---------------------------------------------------------------------------
# Engine (lazy singleton)
# ---------------------------------------------------------------------------

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(f"sqlite:///{DB_PATH}")
    return _engine


# ---------------------------------------------------------------------------
# Synthetic cohort helpers
# ---------------------------------------------------------------------------

def get_all_marks() -> pd.DataFrame:
    """Return the full marks table joined with student names (wide format)."""
    sql = """
        SELECT m.student_id, s.name, m.year, m.subject, m.mark
        FROM marks m
        JOIN students s ON s.student_id = m.student_id
        ORDER BY m.student_id, m.year, m.subject
    """
    with _get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn)


def get_all_marks_wide() -> pd.DataFrame:
    """Return marks pivoted to wide format: student_id, name, year, Math, Physics, ..."""
    long_df = get_all_marks()
    wide_df = long_df.pivot_table(
        index=["student_id", "name", "year"],
        columns="subject",
        values="mark",
    ).reset_index()
    wide_df.columns.name = None
    return wide_df


def get_student_marks(student_id: str) -> pd.DataFrame:
    """Return all marks for a single student (wide format)."""
    df = get_all_marks_wide()
    return df[df["student_id"] == student_id].copy()


def save_predictions(predictions_df: pd.DataFrame) -> None:
    """
    Persist predicted marks to the predictions table.

    Expected columns: student_id, subject, predicted_mark, model_used
    """
    required = {"student_id", "subject", "predicted_mark", "model_used"}
    if not required.issubset(predictions_df.columns):
        missing = required - set(predictions_df.columns)
        raise ValueError(f"predictions_df is missing columns: {missing}")

    predictions_df[list(required)].to_sql(
        "predictions",
        _get_engine(),
        if_exists="append",
        index=False,
    )


def get_predictions(student_id: str) -> pd.DataFrame:
    """Return stored predictions for a single student."""
    sql = """
        SELECT student_id, subject, predicted_mark, model_used, created_at
        FROM predictions
        WHERE student_id = :sid
        ORDER BY created_at DESC
    """
    with _get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params={"sid": student_id})


def get_all_students() -> pd.DataFrame:
    """Return the students lookup table."""
    with _get_engine().connect() as conn:
        return pd.read_sql(text("SELECT * FROM students ORDER BY student_id"), conn)


# ---------------------------------------------------------------------------
# VLE dataset helpers
# ---------------------------------------------------------------------------

def get_vle_data(split: str = "all") -> pd.DataFrame:
    """
    Return VLE student rows.

    Parameters
    ----------
    split : str
        'all'      → all rows
        'original' → original dataset rows
        'test'     → held-out test rows
    """
    if split == "all":
        sql = "SELECT * FROM vle_students ORDER BY id"
    else:
        sql = "SELECT * FROM vle_students WHERE split = :split ORDER BY id"

    with _get_engine().connect() as conn:
        if split == "all":
            return pd.read_sql(text(sql), conn)
        return pd.read_sql(text(sql), conn, params={"split": split})


def get_vle_features_and_labels(split: str = "original") -> tuple[pd.DataFrame, pd.Series]:
    """
    Return (X, y) ready for scikit-learn.

    Features: all columns except id, label, split
    Target:   label (0=pass, 1=fail)
    """
    df = get_vle_data(split=split)
    drop_cols = ["id", "label", "split"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["label"]
    return X, y


def save_vle_predictions(vle_predictions_df: pd.DataFrame) -> None:
    """
    Persist pass/fail predictions for VLE students.

    Expected columns: vle_student_id, predicted_label, predicted_proba, model_used
    """
    required = {"vle_student_id", "predicted_label", "predicted_proba", "model_used"}
    if not required.issubset(vle_predictions_df.columns):
        missing = required - set(vle_predictions_df.columns)
        raise ValueError(f"vle_predictions_df is missing columns: {missing}")

    vle_predictions_df[list(required)].to_sql(
        "vle_predictions",
        _get_engine(),
        if_exists="append",
        index=False,
    )


def get_vle_predictions(vle_student_id: int) -> pd.DataFrame:
    """Return stored predictions for a single VLE student row."""
    sql = """
        SELECT vle_student_id, predicted_label, predicted_proba, model_used, created_at
        FROM vle_predictions
        WHERE vle_student_id = :sid
        ORDER BY created_at DESC
    """
    with _get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params={"sid": vle_student_id})
