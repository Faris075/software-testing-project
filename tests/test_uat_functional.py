from models.train import load_clean_marks, SUBJECTS
from models.predict import predict_2026
from visualisation.charts import bar_predictions_2026


def test_uat_predict_and_chart():
    # Load cleaned marks and pick a real student with a 2025 row
    df = load_clean_marks()
    row = df[df["year"] == 2025].iloc[0]
    student_id = row["student_id"]

    sample = {subject: float(row[subject]) for subject in SUBJECTS}
    preds = predict_2026(sample, "RandomForest")
    assert isinstance(preds, dict)
    pred_df = __import__("pandas").DataFrame({"subject": list(preds.keys()), "predicted_mark": list(preds.values())})

    fig = bar_predictions_2026(pred_df, student_id)
    assert fig is not None
