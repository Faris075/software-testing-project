"""
analysis/stats.py
=================
Statistical analysis functions for the synthetic cohort (marks) dataset.

All functions accept a wide-format DataFrame with columns:
    student_id, name, year, Math, Physics, CS, English, Statistics
"""

import pandas as pd

SUBJECTS = ["Math", "Physics", "CS", "English", "Statistics"]


def subject_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per subject with columns: subject, mean, median, std.
    Aggregated across all students and both years.
    """
    rows = []
    for subj in SUBJECTS:
        col = df[subj].dropna()
        rows.append({
            "subject": subj,
            "mean": round(col.mean(), 2),
            "median": round(col.median(), 2),
            "std": round(col.std(), 2),
        })
    return pd.DataFrame(rows)


def student_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per (student_id, year) with columns:
    student_id, name, year, overall_mean, best_subject, worst_subject.
    """
    rows = []
    for _, row in df.iterrows():
        marks = {s: row[s] for s in SUBJECTS}
        overall_mean = round(sum(marks.values()) / len(marks), 2)
        best = max(marks, key=marks.get)
        worst = min(marks, key=marks.get)
        rows.append({
            "student_id": row["student_id"],
            "name": row["name"],
            "year": int(row["year"]),
            "overall_mean": overall_mean,
            "best_subject": best,
            "worst_subject": worst,
        })
    return pd.DataFrame(rows)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation between subjects.
    Drops non-subject columns (student_id, name, year) before computing.
    """
    return df[SUBJECTS].corr(method="pearson")


def top_performers(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    Returns top-n students by overall average across both years.
    Columns: student_id, name, overall_mean.
    """
    summary = (
        df.groupby(["student_id", "name"])[SUBJECTS]
        .mean()
        .mean(axis=1)
        .reset_index(name="overall_mean")
    )
    summary["overall_mean"] = summary["overall_mean"].round(2)
    return summary.nlargest(n, "overall_mean").reset_index(drop=True)


def failing_students(df: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
    """
    Returns students whose average mark in ANY subject across both years
    falls below threshold.
    Columns: student_id, name, subject, subject_avg.
    """
    avg = (
        df.groupby(["student_id", "name"])[SUBJECTS]
        .mean()
        .reset_index()
    )
    rows = []
    for _, row in avg.iterrows():
        for subj in SUBJECTS:
            if row[subj] < threshold:
                rows.append({
                    "student_id": row["student_id"],
                    "name": row["name"],
                    "subject": subj,
                    "subject_avg": round(row[subj], 2),
                })
    return pd.DataFrame(rows)


def year_on_year_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns per-student per-subject delta: 2025_mark - 2024_mark.
    Positive = improvement, negative = decline.
    Columns: student_id, name, Math_delta, Physics_delta, CS_delta,
             English_delta, Statistics_delta, overall_delta.
    """
    df2024 = df[df["year"] == 2024].set_index("student_id")
    df2025 = df[df["year"] == 2025].set_index("student_id")
    rows = []
    for sid in df2024.index:
        if sid not in df2025.index:
            continue
        row24 = df2024.loc[sid]
        row25 = df2025.loc[sid]
        deltas = {f"{s}_delta": round(float(row25[s]) - float(row24[s]), 2) for s in SUBJECTS}
        overall = round(sum(deltas.values()) / len(SUBJECTS), 2)
        entry = {"student_id": sid, "name": row24["name"]}
        entry.update(deltas)
        entry["overall_delta"] = overall
        rows.append(entry)
    return pd.DataFrame(rows)
