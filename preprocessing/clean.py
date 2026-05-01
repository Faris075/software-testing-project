import pathlib
from typing import List

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STUDENTS_RAW = DATA_DIR / "students_raw.csv"
STUDENTS_CLEAN = DATA_DIR / "students_clean.csv"
SUBJECTS = ["Math", "Physics", "CS", "English", "Statistics"]


def load_raw_data(path: str = None) -> pd.DataFrame:
    path = pathlib.Path(path) if path else STUDENTS_RAW
    return pd.read_csv(path)


def clean_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[SUBJECTS] = df[SUBJECTS].apply(pd.to_numeric, errors="coerce")
    df[SUBJECTS] = df[SUBJECTS].apply(lambda col: col.fillna(col.mean()))
    return df


def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[SUBJECTS] = df[SUBJECTS].clip(0, 100)
    return df


def normalise(df: pd.DataFrame, fit: bool = True):
    return df


def run_pipeline(raw_path: str = None, clean_path: str = None) -> pd.DataFrame:
    df = load_raw_data(raw_path)
    df = clean_missing(df)
    df = clip_outliers(df)
    df.to_csv(clean_path or STUDENTS_CLEAN, index=False)
    return df
