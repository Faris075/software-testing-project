import pathlib

from database import seed as seed_module
from database.db_utils import get_vle_data
from models.evaluate import evaluate_all_classifiers
from models.predict import predict_pass_fail
from models.train import train_vle_classifiers

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _ensure_seeded():
    seed_module.main()


def test_vle_seed_row_count():
    _ensure_seeded()
    df = get_vle_data("all")
    assert len(df) == 160


def test_vle_class_distribution():
    _ensure_seeded()
    df = get_vle_data("all")
    counts = df["label"].value_counts().to_dict()
    assert counts.get(0, 0) == 136
    assert counts.get(1, 0) == 24


def test_predict_pass_fail_output():
    _ensure_seeded()
    train_vle_classifiers()
    train_df = seed_module._load_vle_csv(ROOT / "data" / "vle" / "train_validate" / "csv" / "smote.csv", "train")
    sample = train_df.iloc[0].to_dict()
    if "label" in sample:
        sample.pop("label")
    label, prob = predict_pass_fail(sample, "LogisticRegression")
    assert isinstance(label, int)
    assert isinstance(prob, float)


def test_classifier_recall_fail_nonzero():
    _ensure_seeded()
    train_vle_classifiers()
    df = evaluate_all_classifiers()
    assert (df["recall_fail"] > 0).all()


def test_evaluate_all_classifiers_columns():
    _ensure_seeded()
    train_vle_classifiers()
    df = evaluate_all_classifiers()
    expected = {"model", "accuracy", "f1_macro", "roc_auc", "recall_fail"}
    assert expected.issubset(set(df.columns))
