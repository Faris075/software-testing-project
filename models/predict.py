import pathlib
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from models.train import SUBJECTS, VLE_FEATURE_COLUMNS

ROOT = pathlib.Path(__file__).resolve().parent.parent
SAVED_MODELS = ROOT / "saved_models"


def _load_model(model_name: str, subject: str):
    path = SAVED_MODELS / f"{model_name}_{subject}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Saved regression model not found: {path}")
    return joblib.load(path)


def _load_vle_scaler():
    path = SAVED_MODELS / "vle_scaler.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Saved VLE scaler not found: {path}")
    return joblib.load(path)


def _load_vle_classifier(name: str):
    path = SAVED_MODELS / f"classifier_{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Saved classifier not found: {path}")
    return joblib.load(path)


def predict_2026(student_features: Dict[str, float], model_name: str) -> Dict[str, float]:
    if not all(subject in student_features for subject in SUBJECTS):
        missing = [s for s in SUBJECTS if s not in student_features]
        raise ValueError(f"Missing required subjects for prediction: {missing}")
    year_encoded = 1
    X = np.array([[year_encoded] + [float(student_features[s]) for s in SUBJECTS]])
    predictions = {}
    for subject in SUBJECTS:
        model = _load_model(model_name, subject)
        pred = float(model.predict(X).item())
        predictions[subject] = float(np.clip(pred, 0, 100))
    return predictions


def predict_pass_fail(feature_row: Dict[str, float], model_name: str) -> Tuple[int, float]:
    scaler = _load_vle_scaler()
    classifier = _load_vle_classifier(model_name)
    row = pd.DataFrame([{col: feature_row.get(col, 0) for col in VLE_FEATURE_COLUMNS}])
    X = row[VLE_FEATURE_COLUMNS].astype(float).values
    X_scaled = scaler.transform(X)
    label = int(classifier.predict(X_scaled).item())
    if hasattr(classifier, "predict_proba"):
        proba = float(classifier.predict_proba(X_scaled)[0, 1])
    else:
        proba = float(classifier.decision_function(X_scaled).ravel()[0])
        proba = 1 / (1 + np.exp(-proba))
    return label, proba
