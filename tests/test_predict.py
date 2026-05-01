import pathlib

from models.predict import predict_2026, predict_pass_fail
from models.train import load_vle_train_test, train_regression_models, train_vle_classifiers

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_predict_returns_five_subjects():
    train_regression_models()
    sample = {"Math": 80, "Physics": 75, "CS": 70, "English": 85, "Statistics": 78}
    predictions = predict_2026(sample, "LinearRegression")
    assert isinstance(predictions, dict)
    assert len(predictions) == 5


def test_predicted_marks_in_range():
    train_regression_models()
    sample = {"Math": 80, "Physics": 75, "CS": 70, "English": 85, "Statistics": 78}
    predictions = predict_2026(sample, "RandomForest")
    assert all(0 <= value <= 100 for value in predictions.values())


def test_predict_pass_fail_output():
    train_vle_classifiers()
    train_df, _ = load_vle_train_test()
    sample = train_df.iloc[0].to_dict()
    if "label" in sample:
        sample.pop("label")
    label, prob = predict_pass_fail(sample, "RandomForest")
    assert isinstance(label, int)
    assert isinstance(prob, float)


def test_predict_pass_fail_probability_range():
    train_vle_classifiers()
    train_df, _ = load_vle_train_test()
    sample = train_df.iloc[0].to_dict()
    if "label" in sample:
        sample.pop("label")
    _, prob = predict_pass_fail(sample, "LogisticRegression")
    assert 0.0 <= prob <= 1.0
