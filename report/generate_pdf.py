"""
report/generate_pdf.py
======================
Generates a multi-page PDF report using ReportLab.

Pages:
  1  – Title page
  2  – Executive summary (subject_summary table)
  3  – Top 3 performers + failing students table
  4  – Subject averages bar chart
  5  – Subject correlation heatmap
  6+ – One page per student (2024 marks, 2025 marks, YoY delta, 2026 prediction)
  Last – Model evaluation summary (RMSE / MAE)

Usage:
    from report.generate_pdf import generate_report
    path = generate_report()
"""

import pathlib
from datetime import date

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHARTS_DIR = ROOT / "outputs" / "charts"
REPORTS_DIR = ROOT / "outputs" / "reports"

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

_HEADER_BG = colors.HexColor("#2c3e50")
_ALT_ROW = colors.HexColor("#ecf0f1")
_FAIL_RED = colors.HexColor("#e74c3c")


def _table_style(header_rows: int = 1) -> TableStyle:
    style = [
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), _HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), colors.white),
        ("FONTNAME", (0, 0), (-1, header_rows - 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, header_rows), (-1, -1), [colors.white, _ALT_ROW]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    return TableStyle(style)


def _df_to_table(df: pd.DataFrame, col_widths=None) -> Table:
    headers = list(df.columns)
    data = [headers] + [
        [str(round(v, 2)) if isinstance(v, float) else str(v) for v in row]
        for row in df.itertuples(index=False)
    ]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(_table_style())
    return t


def _embed_chart(name: str, width: float = 14 * cm, height: float = 9 * cm):
    path = CHARTS_DIR / f"{name}.png"
    if path.exists():
        return Image(str(path), width=width, height=height)
    return Paragraph(f"[Chart not found: {name}.png]", getSampleStyleSheet()["Normal"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(output_path: str = None) -> str:
    """
    Build the full PDF report.  Returns the absolute path to the saved file.

    Tries to import live data from analysis.stats and database.db_utils.
    If the database is not yet populated (e.g. in tests) the data sections
    are skipped gracefully.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = str(REPORTS_DIR / "report.pdf")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    story = []

    # --- lazy imports so the module can be imported without a DB present ---
    try:
        from analysis.stats import (
            failing_students,
            subject_summary,
            top_performers,
            year_on_year_change,
        )
        from database.db_utils import get_all_marks_wide, get_all_students

        marks_df = get_all_marks_wide()
        students_df = get_all_students()
        has_data = not marks_df.empty
    except Exception:
        has_data = False

    # =====================================================================
    # PAGE 1 – Title
    # =====================================================================
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("Student Performance Analytics Report", styles["Title"]))
    story.append(Spacer(1, 0.8 * cm))
    story.append(Paragraph("AI-Powered Student Performance Analytics System", styles["Heading2"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(f"Generated: {date.today().strftime('%d %B %Y')}", styles["Normal"]))
    story.append(Spacer(1, 0.4 * cm))

    if has_data:
        n_students = marks_df["student_id"].nunique()
        n_years = marks_df["year"].nunique()
        story.append(Paragraph(f"Students: {n_students}", styles["Normal"]))
        story.append(Paragraph(f"Years covered: {sorted(marks_df['year'].unique())}", styles["Normal"]))
        story.append(Paragraph(f"Year-group rows: {n_years * n_students}", styles["Normal"]))
        story.append(Paragraph("Subjects: Math, Physics, CS, English, Statistics", styles["Normal"]))
    else:
        story.append(Paragraph("Data unavailable at report time.", styles["Normal"]))

    story.append(PageBreak())

    # =====================================================================
    # PAGE 2 – Executive summary
    # =====================================================================
    story.append(Paragraph("Executive Summary", styles["Heading1"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        "The table below shows aggregate statistics across all students and both years.",
        styles["Normal"],
    ))
    story.append(Spacer(1, 0.3 * cm))

    if has_data:
        summary_df = subject_summary(marks_df)
        story.append(_df_to_table(summary_df, col_widths=[5 * cm, 4 * cm, 4 * cm, 4 * cm]))
    else:
        story.append(Paragraph("No data.", styles["Normal"]))

    story.append(PageBreak())

    # =====================================================================
    # PAGE 3 – Top performers + failing students
    # =====================================================================
    story.append(Paragraph("Top Performers", styles["Heading1"]))
    story.append(Spacer(1, 0.3 * cm))

    if has_data:
        top_df = top_performers(marks_df, n=3)
        story.append(_df_to_table(top_df, col_widths=[4 * cm, 5 * cm, 5 * cm]))
        story.append(Spacer(1, 0.8 * cm))

        story.append(Paragraph("At-Risk Students (subject average < 50)", styles["Heading1"]))
        story.append(Spacer(1, 0.3 * cm))
        fail_df = failing_students(marks_df, threshold=50.0)
        if fail_df.empty:
            story.append(Paragraph("No students with a subject average below 50.", styles["Normal"]))
        else:
            t = _df_to_table(fail_df, col_widths=[3.5 * cm, 4 * cm, 4 * cm, 3 * cm])
            t.setStyle(_table_style())
            story.append(t)
    else:
        story.append(Paragraph("No data.", styles["Normal"]))

    story.append(PageBreak())

    # =====================================================================
    # PAGE 4 – Subject averages bar chart
    # =====================================================================
    story.append(Paragraph("Subject Averages", styles["Heading1"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(_embed_chart("bar_subject_averages"))
    story.append(PageBreak())

    # =====================================================================
    # PAGE 5 – Correlation heatmap
    # =====================================================================
    story.append(Paragraph("Subject Correlation Matrix", styles["Heading1"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(_embed_chart("heatmap_correlation"))
    story.append(PageBreak())

    # =====================================================================
    # PAGES 6-N – One page per student
    # =====================================================================
    if has_data:
        yoy_df = year_on_year_change(marks_df)
        yoy_index = yoy_df.set_index("student_id")

        # Try to load best-model predictions
        try:
            from database.db_utils import get_predictions
            has_predictions = True
        except Exception:
            has_predictions = False

        for student_id in sorted(marks_df["student_id"].unique()):
            student_rows = marks_df[marks_df["student_id"] == student_id].sort_values("year")
            name = student_rows.iloc[0]["name"] if not student_rows.empty else student_id

            story.append(Paragraph(f"Student: {name} ({student_id})", styles["Heading1"]))
            story.append(Spacer(1, 0.3 * cm))

            SUBJECTS = ["Math", "Physics", "CS", "English", "Statistics"]

            # 2024 / 2025 marks table
            table_rows = [["Subject"] + [str(y) for y in sorted(student_rows["year"].unique())]]
            for subj in SUBJECTS:
                row = [subj]
                for yr in sorted(student_rows["year"].unique()):
                    yr_row = student_rows[student_rows["year"] == yr]
                    val = yr_row.iloc[0][subj] if not yr_row.empty else "—"
                    row.append(f"{val:.1f}" if isinstance(val, float) else str(val))
                table_rows.append(row)

            t = Table(table_rows, colWidths=[5 * cm, 3.5 * cm, 3.5 * cm])
            t.setStyle(_table_style())
            story.append(t)
            story.append(Spacer(1, 0.5 * cm))

            # YoY delta
            if student_id in yoy_index.index:
                yoy_row = yoy_index.loc[student_id]
                delta_rows = [["Subject", "Delta (2025 − 2024)"]]
                for subj in SUBJECTS:
                    delta = yoy_row.get(f"{subj}_delta", "—")
                    delta_rows.append([subj, f"{delta:+.2f}" if isinstance(delta, float) else str(delta)])
                overall = yoy_row.get("overall_delta", "—")
                delta_rows.append(["Overall", f"{overall:+.2f}" if isinstance(overall, float) else str(overall)])
                t2 = Table(delta_rows, colWidths=[5 * cm, 5 * cm])
                t2.setStyle(_table_style())
                story.append(Paragraph("Year-on-Year Change", styles["Heading2"]))
                story.append(Spacer(1, 0.2 * cm))
                story.append(t2)
                story.append(Spacer(1, 0.5 * cm))

            # 2026 predictions (best model: RandomForest)
            if has_predictions:
                try:
                    pred_df = get_predictions(student_id)
                    if not pred_df.empty:
                        # Take the most recent per subject
                        latest = pred_df.sort_values("created_at", ascending=False).drop_duplicates("subject")
                        pred_table = [["Subject", "Predicted 2026 Mark", "Model"]]
                        for _, pr in latest.iterrows():
                            pred_table.append([pr["subject"], f"{pr['predicted_mark']:.1f}", pr["model_used"]])
                        t3 = Table(pred_table, colWidths=[4 * cm, 5 * cm, 5 * cm])
                        t3.setStyle(_table_style())
                        story.append(Paragraph("2026 Predictions", styles["Heading2"]))
                        story.append(Spacer(1, 0.2 * cm))
                        story.append(t3)
                except Exception:
                    pass

            story.append(PageBreak())

    # =====================================================================
    # LAST PAGE – Model evaluation summary
    # =====================================================================
    story.append(Paragraph("Model Evaluation Summary", styles["Heading1"]))
    story.append(Spacer(1, 0.4 * cm))

    eval_csv = ROOT / "outputs" / "eval_results.csv"
    if eval_csv.exists():
        eval_df = pd.read_csv(eval_csv)
        story.append(_df_to_table(eval_df))
    else:
        # Try reading regression eval live
        try:
            from models.evaluate import evaluate_all_models
            eval_df = evaluate_all_models()
            eval_df = eval_df[["model", "subject", "RMSE", "MAE"]].round(3)
            story.append(_df_to_table(eval_df))
        except Exception as exc:
            story.append(Paragraph(
                f"Evaluation metrics unavailable: {exc}", styles["Normal"]
            ))

    # =====================================================================
    # Build
    # =====================================================================
    doc.build(story)
    return str(pathlib.Path(output_path).resolve())
