from models.train import train_vle_classifiers
from models.evaluate import evaluate_all_classifiers


def test_classifiers_accuracy_threshold():
    # Train classifiers and evaluate; ensure at least one classifier meets expected accuracy
    train_vle_classifiers()
    df = evaluate_all_classifiers()
    assert not df.empty
    # Require at least one classifier to have accuracy >= 0.7
    assert (df["accuracy"] >= 0.7).any()
