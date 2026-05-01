import pathlib

import joblib
import pytest

from models.evaluate import evaluate_all_models
from models.train import CLASSIFIERS, REGRESSION_MODEL_FACTORIES, train_regression_models, train_vle_classifiers

ROOT = pathlib.Path(__file__).resolve().parent.parent
SAVED_MODELS = ROOT / "saved_models"


def test_model_trains_without_error():
    train_regression_models()
    for model_name in REGRESSION_MODEL_FACTORIES:
        for subject in ["Math", "Physics", "CS", "English", "Statistics"]:
            assert (SAVED_MODELS / f"{model_name}_{subject}.pkl").exists()


def test_pkl_files_saved():
    train_regression_models()
    for model_name in REGRESSION_MODEL_FACTORIES:
        for subject in ["Math", "Physics", "CS", "English", "Statistics"]:
            assert (SAVED_MODELS / f"{model_name}_{subject}.pkl").exists()


def test_rmse_reasonable():
    train_regression_models()
    df = evaluate_all_models()
    assert not df.empty
    assert (df["RMSE"] < 20).all()


def test_classifier_trains_without_error():
    train_vle_classifiers()
    for name in CLASSIFIERS:
        assert (SAVED_MODELS / f"classifier_{name}.pkl").exists()
        assert (SAVED_MODELS / "vle_scaler.pkl").exists()


def test_classifier_pkl_saved():
    for name in CLASSIFIERS:
        assert (SAVED_MODELS / f"classifier_{name}.pkl").exists()
