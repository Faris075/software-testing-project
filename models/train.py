import pathlib
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SAVED_MODELS = ROOT / "saved_models"
SAVED_MODELS.mkdir(parents=True, exist_ok=True)

STUDENTS_RAW = DATA_DIR / "students_raw.csv"
STUDENTS_CLEAN = DATA_DIR / "students_clean.csv"
VLE_SMOTE_CSV = DATA_DIR / "vle" / "train_validate" / "csv" / "smote.csv"
VLE_TEST_CSV = DATA_DIR / "vle" / "test" / "test.csv"

SUBJECTS = ["Math", "Physics", "CS", "English", "Statistics"]
REGRESSION_MODEL_FACTORIES = {
    "LinearRegression": lambda: LinearRegression(),
    "DecisionTree": lambda: DecisionTreeRegressor(max_depth=4, random_state=42),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=100, random_state=42),
}
VLE_FEATURE_COLUMNS = [
    "gender",
    "age",
    "logins",
    "total_hours",
    "pct_avg_hours",
    "presence_count",
    "absence_count",
    "pct_attended",
    "attending_from_home",
    "distance_to_uni_km",
    "polar4_quintile",
    "polar3_quintile",
    "adult_he_2001_quintile",
    "adult_he_2011_quintile",
    "tundra_msoa_quintile",
    "tundra_lsoa_quintile",
    "gaps_gcse_quintile",
    "gaps_gcse_ethnicity_quintile",
    "uni_connect_target_ward",
]
CLASSIFIERS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
}

VLE_COLUMN_MAP = {
    "gender": "gender",
    "age": "age",
    "polar4quintile": "polar4_quintile",
    "polar3quintile": "polar3_quintile",
    "adulthe2001quintile": "adult_he_2001_quintile",
    "adulthe2011quintile": "adult_he_2011_quintile",
    "tundramsoaquintile": "tundra_msoa_quintile",
    "tundralsoaquintile": "tundra_lsoa_quintile",
    "gapsgcsequintile": "gaps_gcse_quintile",
    "gapsgcsethnicityquintile": "gaps_gcse_ethnicity_quintile",
    "uniconnecttargetward": "uni_connect_target_ward",
    "attendingfromhome": "attending_from_home",
    "distancetouniversitykm": "distance_to_uni_km",
    "countofmodulearealogins": "logins",
    "totalhoursinmodulearea": "total_hours",
    "pofaveragehoursinmodulearea": "pct_avg_hours",
    "ofpresence": "presence_count",
    "ofabsence": "absence_count",
    "percentattended": "pct_attended",
    "labelfail1pass0": "label",
    "label": "label",
}


def _norm_header(col: str) -> str:
    return "".join(ch.lower() for ch in str(col) if ch.isalnum())


def _rename_vle_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        key = _norm_header(col)
        if key in VLE_COLUMN_MAP:
            rename_map[col] = VLE_COLUMN_MAP[key]
    df = df.rename(columns=rename_map)
    return df


def _clean_marks_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[SUBJECTS] = df[SUBJECTS].apply(pd.to_numeric, errors="coerce")
    df[SUBJECTS] = df[SUBJECTS].apply(lambda col: col.fillna(col.mean()))
    df[SUBJECTS] = df[SUBJECTS].clip(0, 100)
    df["year"] = df["year"].astype(int)
    df["year_encoded"] = df["year"].map({2024: 0, 2025: 1})
    return df


def run_clean_pipeline() -> pd.DataFrame:
    if not STUDENTS_RAW.exists():
        raise FileNotFoundError(f"Missing raw student file: {STUDENTS_RAW}")
    df = pd.read_csv(STUDENTS_RAW)
    df = _clean_marks_df(df)
    df.to_csv(STUDENTS_CLEAN, index=False)
    return df


def load_clean_marks() -> pd.DataFrame:
    if STUDENTS_CLEAN.exists():
        df = pd.read_csv(STUDENTS_CLEAN)
    else:
        df = run_clean_pipeline()
    return _clean_marks_df(df)


def build_regression_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    base = df[df["year"] == 2024].set_index("student_id")
    next_year = df[df["year"] == 2025].set_index("student_id")
    merged = base[SUBJECTS + ["year_encoded"]].join(
        next_year[SUBJECTS].add_suffix("_next"), how="inner", lsuffix="", rsuffix="_next"
    )
    return merged.reset_index()


def train_regression_models() -> None:
    df = load_clean_marks()
    merged = build_regression_dataset(df)
    X = merged[["year_encoded"] + SUBJECTS].values
    for model_name, factory in REGRESSION_MODEL_FACTORIES.items():
        for target in SUBJECTS:
            y = merged[f"{target}_next"].values
            estimator = factory()
            estimator.fit(X, y)
            joblib.dump(estimator, SAVED_MODELS / f"{model_name}_{target}.pkl")
    print(f"[train] Saved regression models to {SAVED_MODELS}")


def _load_vle_dataframe(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = _rename_vle_columns(df)
    if "label" not in df.columns:
        raise ValueError("VLE dataset missing label column after mapping")
    df = df.loc[:, ~df.columns.str.fullmatch(r"Unnamed.*")]
    df = df.rename(columns=lambda c: c.strip())
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def load_vle_train_test() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = _load_vle_dataframe(VLE_SMOTE_CSV)
    test_df = _load_vle_dataframe(VLE_TEST_CSV)
    return train_df, test_df


def get_vle_feature_columns(df: pd.DataFrame) -> List[str]:
    if all(col in df.columns for col in VLE_FEATURE_COLUMNS):
        return VLE_FEATURE_COLUMNS.copy()
    drop = ["label", "id", "split"]
    return [c for c in df.columns if c not in drop]


def train_vle_classifiers() -> None:
    train_df, test_df = load_vle_train_test()
    feature_cols = get_vle_feature_columns(train_df)
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["label"].astype(int).values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, SAVED_MODELS / "vle_scaler.pkl")

    for name, model in CLASSIFIERS.items():
        estimator = model
        estimator.fit(X_train_scaled, y_train)
        joblib.dump(estimator, SAVED_MODELS / f"classifier_{name}.pkl")
    print(f"[train] Saved VLE classifiers and scaler to {SAVED_MODELS}")


def main() -> None:
    train_regression_models()
    train_vle_classifiers()


if __name__ == "__main__":
    main()
