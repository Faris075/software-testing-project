import pathlib
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score

from models.train import (
    CLASSIFIERS,
    REGRESSION_MODEL_FACTORIES,
    SUBJECTS,
    build_regression_dataset,
    get_vle_feature_columns,
    load_clean_marks,
    load_vle_train_test,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
SAVED_MODELS = ROOT / "saved_models"


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def cross_validate_model(model, X, y, cv: int = 5) -> Dict[str, float]:
    scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=cv)
    rmses = np.sqrt(-scores)
    return {"mean": float(rmses.mean()), "std": float(rmses.std())}


def evaluate_all_models() -> pd.DataFrame:
    df = load_clean_marks()
    merged = build_regression_dataset(df)
    X = merged[["year_encoded"] + SUBJECTS].values
    rows = []
    for model_name in REGRESSION_MODEL_FACTORIES:
        for subject in SUBJECTS:
            y = merged[f"{subject}_next"].values
            estimator = joblib.load(SAVED_MODELS / f"{model_name}_{subject}.pkl")
            y_pred = estimator.predict(X)
            model = REGRESSION_MODEL_FACTORIES[model_name]()
            rows.append(
                {
                    "model": model_name,
                    "subject": subject,
                    "RMSE": rmse(y, y_pred),
                    "MAE": mae(y, y_pred),
                    **cross_validate_model(model, X, y),
                }
            )
    return pd.DataFrame(rows)


def evaluate_classifier(model, X_test, y_test) -> Dict[str, object]:
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    auc = roc_auc_score(y_test, proba) if proba is not None else float("nan")
    recall_fail = recall_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, proba) if proba is not None else ([], [], [])
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "roc_auc": float(auc),
        "recall_fail": float(recall_fail),
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
    }


def evaluate_all_classifiers() -> pd.DataFrame:
    _, test_df = load_vle_train_test()
    feature_cols = get_vle_feature_columns(test_df)
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df["label"].astype(int).values
    scaler = joblib.load(SAVED_MODELS / "vle_scaler.pkl")
    X_test = scaler.transform(X_test)
    results = []
    for model_name in CLASSIFIERS:
        model = joblib.load(SAVED_MODELS / f"classifier_{model_name}.pkl")
        metrics = evaluate_classifier(model, X_test, y_test)
        results.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "roc_auc": metrics["roc_auc"],
                "recall_fail": metrics["recall_fail"],
            }
        )
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "eval_results.csv", index=False)
    return df
