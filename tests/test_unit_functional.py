import pandas as pd

from preprocessing.clean import clean_missing, clip_outliers, SUBJECTS


def test_clean_missing_fills_nans():
    df = pd.DataFrame({
        "student_id": ["S001", "S002", "S003"],
        "year": [2024, 2024, 2024],
        "Math": [90, None, 80],
        "Physics": [None, 70, 75],
        "CS": [65, 70, None],
        "English": [None, None, 85],
        "Statistics": [78, 82, None],
    })

    out = clean_missing(df)
    # No NaNs remain in subject columns
    assert not out[SUBJECTS].isna().any().any()
    # Mean fill for Math should be (90 + 80) / 2 = 85
    assert abs(out.loc[1, "Math"] - 85.0) < 1e-6


def test_clip_outliers_clips_range():
    df = pd.DataFrame({
        "Math": [-10, 150],
        "Physics": [200, -5],
        "CS": [0, 100],
        "English": [101, -1],
        "Statistics": [50, 60],
    })

    out = clip_outliers(df)
    # All values should be within [0, 100]
    assert out[["Math", "Physics", "CS", "English", "Statistics"]].min().min() >= 0
    assert out[["Math", "Physics", "CS", "English", "Statistics"]].max().max() <= 100
