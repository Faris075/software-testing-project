import pandas as pd


from preprocessing.clean import run_pipeline


def test_run_pipeline_writes_clean(tmp_path):
    # Create a small raw CSV representing students
    raw = tmp_path / "raw_students.csv"
    df = pd.DataFrame([
        {"student_id": "S001", "year": 2024, "Math": 80, "Physics": 75, "CS": 70, "English": 85, "Statistics": 78},
        {"student_id": "S002", "year": 2025, "Math": 82, "Physics": 77, "CS": 73, "English": 88, "Statistics": 80},
    ])
    df.to_csv(raw, index=False)

    clean_path = tmp_path / "students_clean.csv"
    out = run_pipeline(str(raw), str(clean_path))

    assert clean_path.exists()
    assert "Math" in out.columns
    assert not out.empty
