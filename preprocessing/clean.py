import pathlib

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STUDENTS_RAW = DATA_DIR / "students_raw.csv"
STUDENTS_CLEAN = DATA_DIR / "students_clean.csv"
SCALER_PATH = ROOT / "saved_models" / "scaler.pkl"
SUBJECTS = ["Math", "Physics", "CS", "English", "Statistics"]

IMPUTATION_LOG: list = []


def load_raw_data(path: str = None) -> pd.DataFrame:
    path = pathlib.Path(path) if path else STUDENTS_RAW
    return pd.read_csv(path)


def clean_missing(df: pd.DataFrame) -> pd.DataFrame:
    global IMPUTATION_LOG
    IMPUTATION_LOG = []
    df = df.copy()
    df[SUBJECTS] = df[SUBJECTS].apply(pd.to_numeric, errors="coerce")
    for year, group_idx in df.groupby("year").groups.items():
        for subj in SUBJECTS:
            col = df.loc[group_idx, subj]
            missing_mask = col.isna()
            if missing_mask.any():
                fill_val = col.mean()
                for idx in col[missing_mask].index:
                    sid = df.loc[idx, "student_id"]
                    entry = (sid, year, subj, round(fill_val, 2))
                    IMPUTATION_LOG.append(entry)
                    print(f"[clean] Imputed: student={sid}, year={year}, subject={subj}, value={fill_val:.2f}")
                df.loc[group_idx, subj] = col.fillna(fill_val)
    return df


def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[SUBJECTS] = df[SUBJECTS].clip(0, 100)
    return df


def normalise(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    df = df.copy()
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if fit:
        scaler = MinMaxScaler()
        df[[f"{s}_norm" for s in SUBJECTS]] = scaler.fit_transform(df[SUBJECTS])
        joblib.dump(scaler, SCALER_PATH)
    else:
        scaler = joblib.load(SCALER_PATH)
        df[[f"{s}_norm" for s in SUBJECTS]] = scaler.transform(df[SUBJECTS])
    return df


def encode_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year_encoded"] = df["year"].map({2024: 0, 2025: 1})
    return df


def run_pipeline(raw_path: str = None, clean_path: str = None) -> pd.DataFrame:
    df = load_raw_data(raw_path)
    df = clean_missing(df)
    df = clip_outliers(df)
    df = normalise(df, fit=True)
    df = encode_year(df)
    out = pathlib.Path(clean_path) if clean_path else STUDENTS_CLEAN
    df.to_csv(out, index=False)
    print(f"[clean] Saved cleaned data to {out}")
    return df


if __name__ == "__main__":
    run_pipeline()
