import pathlib
import time
import pytest
from database import seed as seed_module
from database.db_utils import get_all_marks_wide, get_vle_data
from models.train import load_clean_marks, train_regression_models, train_vle_classifiers
from models.predict import predict_2026, predict_pass_fail
from models.evaluate import evaluate_all_models, evaluate_all_classifiers
from preprocessing.clean import run_pipeline
from visualisation.charts import bar_subject_averages

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_full_data_pipeline():
    """Functional: Test end-to-end data pipeline from raw CSV to cleaned data."""
    seed_module.main()
    df_clean = run_pipeline()
    assert not df_clean.empty
    assert "Math" in df_clean.columns
    assert df_clean["Math"].max() <= 100
    assert df_clean["Math"].min() >= 0


def test_model_training_and_prediction_integration():
    """Functional: Test training models and making predictions."""
    seed_module.main()
    run_pipeline()
    train_regression_models()
    train_vle_classifiers()

    # Test regression prediction
    sample_marks = {"Math": 80, "Physics": 75, "CS": 70, "English": 85, "Statistics": 78}
    pred = predict_2026(sample_marks, "LinearRegression")
    assert isinstance(pred, dict)
    assert len(pred) == 5
    assert all(0 <= v <= 100 for v in pred.values())

    # Test VLE prediction
    train_df = seed_module._load_vle_csv(ROOT / "data" / "vle" / "train_validate" / "csv" / "smote.csv", "train")
    sample_vle = train_df.iloc[0].to_dict()
    sample_vle.pop("label", None)
    label, prob = predict_pass_fail(sample_vle, "RandomForest")
    assert isinstance(label, int)
    assert 0 <= prob <= 1


def test_dashboard_data_loading():
    """Functional: Test that dashboard can load required data."""
    seed_module.main()
    run_pipeline()
    marks_df = load_clean_marks()
    vle_df = get_vle_data("original")

    assert not marks_df.empty
    assert not vle_df.empty
    assert "student_id" in marks_df.columns
    assert "label" in vle_df.columns


def test_chart_generation():
    """Functional: Test that charts generate without errors."""
    seed_module.main()
    run_pipeline()
    df = load_clean_marks()
    fig = bar_subject_averages(df)
    assert fig is not None
    # Note: In a real test, you might check if file exists in outputs/charts/


def test_evaluation_metrics():
    """Functional: Test that evaluation produces expected outputs."""
    seed_module.main()
    run_pipeline()
    train_regression_models()
    train_vle_classifiers()

    reg_eval = evaluate_all_models()
    clf_eval = evaluate_all_classifiers()

    assert not reg_eval.empty
    assert "RMSE" in reg_eval.columns
    assert not clf_eval.empty
    assert "accuracy" in clf_eval.columns
