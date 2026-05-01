import time
import pathlib
import pytest
from database import seed as seed_module
from models.train import train_regression_models, train_vle_classifiers
from models.predict import predict_2026, predict_pass_fail
from preprocessing.clean import run_pipeline

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_model_training_performance():
    """Non-Functional: Test that model training completes within reasonable time."""
    seed_module.main()
    run_pipeline()

    start = time.time()
    train_regression_models()
    reg_time = time.time() - start

    start = time.time()
    train_vle_classifiers()
    clf_time = time.time() - start

    # Assert reasonable training times (e.g., under 30 seconds each)
    assert reg_time < 30, f"Regression training too slow: {reg_time:.2f}s"
    assert clf_time < 30, f"Classifier training too slow: {clf_time:.2f}s"


def test_prediction_performance():
    """Non-Functional: Test prediction speed."""
    seed_module.main()
    run_pipeline()
    train_regression_models()
    train_vle_classifiers()

    sample_marks = {"Math": 80, "Physics": 75, "CS": 70, "English": 85, "Statistics": 78}

    # Test regression prediction time
    start = time.time()
    for _ in range(10):
        predict_2026(sample_marks, "LinearRegression")
    reg_pred_time = (time.time() - start) / 10

    # Test VLE prediction time
    train_df = seed_module._load_vle_csv(ROOT / "data" / "vle" / "train_validate" / "csv" / "smote.csv", "train")
    sample_vle = train_df.iloc[0].to_dict()
    sample_vle.pop("label", None)

    start = time.time()
    for _ in range(10):
        predict_pass_fail(sample_vle, "RandomForest")
    clf_pred_time = (time.time() - start) / 10

    # Assert predictions are fast (under 0.1s each)
    assert reg_pred_time < 0.1, f"Regression prediction too slow: {reg_pred_time:.4f}s"
    assert clf_pred_time < 0.1, f"Classifier prediction too slow: {clf_pred_time:.4f}s"


def test_data_loading_performance():
    """Non-Functional: Test data loading speed."""
    seed_module.main()

    start = time.time()
    run_pipeline()
    load_time = time.time() - start

    assert load_time < 5, f"Data loading too slow: {load_time:.2f}s"


def test_memory_usage_estimate():
    """Non-Functional: Basic memory check (rough estimate via data size)."""
    seed_module.main()
    run_pipeline()

    # Check if dataframes are reasonable size
    from models.train import load_clean_marks
    df = load_clean_marks()
    # Rough check: not too large for in-memory processing
    assert len(df) < 10000, "Data too large for typical memory"


def test_usability_proxy_error_handling():
    """Non-Functional: Test error handling for usability (e.g., invalid inputs)."""
    # Test prediction with invalid inputs
    with pytest.raises(ValueError):
        predict_2026({"Math": 80}, "LinearRegression")  # Missing subjects

    with pytest.raises(FileNotFoundError):
        predict_pass_fail({}, "InvalidModel")  # Invalid model
