import pathlib
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
ROOT = pathlib.Path(__file__).parent.parent
CHARTS_DIR = ROOT / "outputs" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT_COLUMNS = ["Math", "Physics", "CS", "English", "Statistics"]


def _save(fig: plt.Figure, name: str) -> plt.Figure:
    path = CHARTS_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return fig


def _ensure_wide(df: pd.DataFrame) -> pd.DataFrame:
    if set(SUBJECT_COLUMNS).issubset(df.columns):
        return df.copy()
    if {"subject", "mark"}.issubset(df.columns):
        return df.pivot_table(
            index=[col for col in df.columns if col not in ["subject", "mark"]],
            columns="subject",
            values="mark",
        ).reset_index()
    raise ValueError("DataFrame must contain either subject columns or long-format subject/mark columns.")


def bar_subject_averages(df: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Figure:
    wide = _ensure_wide(df)
    summary = wide[SUBJECT_COLUMNS].mean().sort_values(ascending=False)
    _own_fig = ax is None
    if _own_fig:
        fig, ax = plt.subplots(figsize=(8, 5))
    summary.plot.bar(ax=ax, color="#4c72b0")
    ax.set_title("Average mark per subject")
    ax.set_ylabel("Average mark")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Subject")
    if _own_fig:
        return _save(fig, "bar_subject_averages")
    return ax.figure


def bar_student_averages(df: pd.DataFrame, year: int, ax: Optional[plt.Axes] = None) -> plt.Figure:
    wide = _ensure_wide(df)
    subset = wide[wide["year"] == year]
    student_averages = subset.set_index("student_id")[SUBJECT_COLUMNS].mean(axis=1).sort_values()
    _own_fig = ax is None
    if _own_fig:
        fig, ax = plt.subplots(figsize=(10, 5))
    student_averages.plot.bar(ax=ax, color="#55a868")
    ax.set_title(f"Average mark per student ({year})")
    ax.set_ylabel("Average mark")
    ax.set_xlabel("Student ID")
    ax.set_ylim(0, 100)
    if _own_fig:
        return _save(fig, f"bar_student_averages_{year}")
    return ax.figure


def line_progress(df: pd.DataFrame, student_id: str, ax: Optional[plt.Axes] = None) -> plt.Figure:
    wide = _ensure_wide(df)
    student = wide[wide["student_id"] == student_id].sort_values("year")
    if student.empty:
        raise ValueError(f"No data found for student_id={student_id}")
    _own_fig = ax is None
    if _own_fig:
        fig, ax = plt.subplots(figsize=(8, 5))
    for subject in SUBJECT_COLUMNS:
        ax.plot(student["year"], student[subject], marker="o", label=subject)
    ax.set_title(f"Subject progress for {student_id}")
    ax.set_ylabel("Mark")
    ax.set_xlabel("Year")
    ax.set_xticks(student["year"].unique())
    ax.set_ylim(0, 100)
    ax.legend(title="Subject", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    if _own_fig:
        return _save(fig, f"line_progress_{student_id}")
    return ax.figure


def line_class_progress(df: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Figure:
    wide = _ensure_wide(df)
    summary = wide.groupby("year")[SUBJECT_COLUMNS].mean()
    _own_fig = ax is None
    if _own_fig:
        fig, ax = plt.subplots(figsize=(8, 5))
    for subject in SUBJECT_COLUMNS:
        ax.plot(summary.index, summary[subject], marker="o", label=subject)
    ax.set_title("Class average progress by subject")
    ax.set_ylabel("Average mark")
    ax.set_xlabel("Year")
    ax.set_xticks(summary.index)
    ax.set_ylim(0, 100)
    ax.legend(title="Subject", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    if _own_fig:
        return _save(fig, "line_class_progress")
    return ax.figure


def comparison_bar(df: pd.DataFrame, student_id: str, ax: Optional[plt.Axes] = None) -> plt.Figure:
    wide = _ensure_wide(df)
    student = wide[wide["student_id"] == student_id].sort_values("year")
    if student.empty:
        raise ValueError(f"No data found for student_id={student_id}")
    data = student.set_index("year")[SUBJECT_COLUMNS].T
    _own_fig = ax is None
    if _own_fig:
        fig, ax = plt.subplots(figsize=(10, 5))
    data.plot.bar(ax=ax)
    ax.set_title(f"2024 vs 2025 comparison for {student_id}")
    ax.set_ylabel("Mark")
    ax.set_xlabel("Subject")
    ax.set_ylim(0, 100)
    ax.legend(title="Year", loc="upper left")
    if _own_fig:
        return _save(fig, f"comparison_bar_{student_id}")
    return ax.figure


def heatmap_correlation(corr_matrix: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Figure:
    _own_fig = ax is None
    if _own_fig:
        fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Subject correlation matrix")
    if _own_fig:
        return _save(fig, "heatmap_correlation")
    return ax.figure


def scatter_actual_vs_predicted(y_true, y_pred, subject: str, ax: Optional[plt.Axes] = None) -> plt.Figure:
    _own_fig = ax is None
    if _own_fig:
        fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.75, edgecolor="k", color="#ff7f0e")
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray")
    ax.set_title(f"Actual vs Predicted — {subject}")
    ax.set_xlabel("Actual mark")
    ax.set_ylabel("Predicted mark")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    if _own_fig:
        return _save(fig, f"scatter_actual_vs_predicted_{subject}")
    return ax.figure


def bar_predictions_2026(pred_df: pd.DataFrame, student_id: str, ax: Optional[plt.Axes] = None) -> plt.Figure:
    if "subject" in pred_df.columns and "predicted_mark" in pred_df.columns:
        plot_data = pred_df.set_index("subject")["predicted_mark"].sort_values()
    else:
        plot_data = pd.Series(pred_df, name="predicted_mark")
    _own_fig = ax is None
    if _own_fig:
        fig, ax = plt.subplots(figsize=(8, 5))
    plot_data.plot.bar(ax=ax, color="#beaed4")
    ax.set_title(f"Predicted 2026 marks for {student_id}")
    ax.set_ylabel("Predicted mark")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Subject")
    if _own_fig:
        return _save(fig, f"bar_predictions_2026_{student_id}")
    return ax.figure


def bar_vle_class_distribution(df_vle: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Figure:
    counts = df_vle["label"].value_counts().sort_index()
    labels = ["Pass" if idx == 0 else "Fail" for idx in counts.index]
    _own_fig = ax is None
    if _own_fig:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, counts.values, color=["#2ca02c", "#d62728"])
    ax.set_title("VLE pass/fail distribution")
    ax.set_ylabel("Count")
    if _own_fig:
        return _save(fig, "bar_vle_class_distribution")
    return ax.figure


def heatmap_vle_correlation(df_vle: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Figure:
    features = df_vle.select_dtypes(include=["number"]).drop(columns=[col for col in ["id", "label"] if col in df_vle.columns], errors="ignore")
    corr_matrix = features.corr()
    _own_fig = ax is None
    if _own_fig:
        fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("VLE feature correlation")
    if _own_fig:
        return _save(fig, "heatmap_vle_correlation")
    return ax.figure


def bar_feature_importance(importances, feature_names, ax: Optional[plt.Axes] = None) -> plt.Figure:
    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    _own_fig = ax is None
    if _own_fig:
        fig, ax = plt.subplots(figsize=(8, 6))
    imp_series.plot.barh(ax=ax, color="#377eb8")
    ax.set_title("Feature importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    if _own_fig:
        return _save(fig, "bar_feature_importance")
    return ax.figure


def plot_roc_curve(fpr, tpr, auc_score: float, model_name: str, ax: Optional[plt.Axes] = None) -> plt.Figure:
    _own_fig = ax is None
    if _own_fig:
        fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc_score:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(f"ROC curve — {model_name}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    if _own_fig:
        return _save(fig, f"roc_curve_{model_name}")
    return ax.figure


def plot_confusion_matrix(cm, model_name: str, ax: Optional[plt.Axes] = None) -> plt.Figure:
    _own_fig = ax is None
    if _own_fig:
        fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {model_name}")
    if _own_fig:
        return _save(fig, f"confusion_matrix_{model_name}")
    return ax.figure
