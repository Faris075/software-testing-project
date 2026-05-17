from database import seed as seed_module

from preprocessing.clean import run_pipeline
from models.train import train_regression_models, train_vle_classifiers, load_vle_train_test
from models.predict import predict_2026, predict_pass_fail


def test_full_system_flow():
    # Seed DB and run cleaning
    seed_module.main()
    df = run_pipeline()

    # Train models
    train_regression_models()
    train_vle_classifiers()

    # Regression prediction
    sample_marks = {"Math": 80, "Physics": 75, "CS": 70, "English": 85, "Statistics": 78}
    preds = predict_2026(sample_marks, "LinearRegression")
    assert isinstance(preds, dict)
    assert len(preds) == 5

    # VLE prediction
    train_df, _ = load_vle_train_test()
    sample_vle = train_df.iloc[0].to_dict()
    sample_vle.pop("label", None)
    label, prob = predict_pass_fail(sample_vle, "RandomForest")
    assert isinstance(label, int)
    assert 0.0 <= prob <= 1.0
