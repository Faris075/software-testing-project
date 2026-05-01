import pathlib
import pytest
from database import seed as seed_module
from models.predict import predict_pass_fail
from models.train import train_vle_classifiers

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_security_input_validation():
    """Specialized: Test for security vulnerabilities like SQL injection or invalid inputs."""
    seed_module.main()
    train_vle_classifiers()

    # Test with malicious inputs (should not crash or inject)
    malicious_input = {
        "gender": "'; DROP TABLE students; --",
        "age": 20,
        "logins": 10,
        "total_hours": 50,
        "pct_avg_hours": 1.0,
        "presence_count": 8,
        "absence_count": 2,
        "pct_attended": 80,
        "attending_from_home": 0,
        "distance_to_uni_km": 10,
        "polar4_quintile": 3,
        "polar3_quintile": 3,
        "adult_he_2001_quintile": 3,
        "adult_he_2011_quintile": 3,
        "tundra_msoa_quintile": 3,
        "tundra_lsoa_quintile": 3,
        "gaps_gcse_quintile": 3,
        "gaps_gcse_ethnicity_quintile": 3,
        "uni_connect_target_ward": 0,
    }

    # Should raise ValueError for invalid input types (security: prevent crashes from bad data)
    with pytest.raises(ValueError):
        predict_pass_fail(malicious_input, "LogisticRegression")


def test_load_testing_basic():
    """Specialized: Basic load testing for predictions (simulate multiple calls)."""
    seed_module.main()
    train_vle_classifiers()

    train_df = seed_module._load_vle_csv(ROOT / "data" / "vle" / "train_validate" / "csv" / "smote.csv", "train")
    sample_vle = train_df.iloc[0].to_dict()
    sample_vle.pop("label", None)

    # Simulate 100 predictions
    for _ in range(100):
        label, prob = predict_pass_fail(sample_vle, "RandomForest")
        assert isinstance(label, int)
        assert 0 <= prob <= 1


def test_accessibility_proxy_data_types():
    """Specialized: Test for accessibility/usability via data type consistency."""
    seed_module.main()
    train_vle_classifiers()

    # Ensure predictions return expected types for UI consumption
    train_df = seed_module._load_vle_csv(ROOT / "data" / "vle" / "train_validate" / "csv" / "smote.csv", "train")
    sample_vle = train_df.iloc[0].to_dict()
    sample_vle.pop("label", None)

    label, prob = predict_pass_fail(sample_vle, "RandomForest")
    assert isinstance(label, int) and label in [0, 1]
    assert isinstance(prob, float) and 0.0 <= prob <= 1.0


def test_robustness_edge_cases():
    """Specialized: Test robustness with edge case inputs."""
    seed_module.main()
    train_vle_classifiers()

    # Test with extreme values
    edge_input = {
        "gender": 1,
        "age": 100,  # Very old
        "logins": 0,  # No logins
        "total_hours": 0.0,
        "pct_avg_hours": 0.0,
        "presence_count": 0,
        "absence_count": 12,
        "pct_attended": 0.0,
        "attending_from_home": 1,
        "distance_to_uni_km": 1000,  # Far away
        "polar4_quintile": 5,
        "polar3_quintile": 5,
        "adult_he_2001_quintile": 5,
        "adult_he_2011_quintile": 5,
        "tundra_msoa_quintile": 5,
        "tundra_lsoa_quintile": 5,
        "gaps_gcse_quintile": 5,
        "gaps_gcse_ethnicity_quintile": 5,
        "uni_connect_target_ward": 1,
    }

    label, prob = predict_pass_fail(edge_input, "LogisticRegression")
    assert isinstance(label, int)
    assert isinstance(prob, float)


def test_configuration_testing():
    """Specialized: Test different configurations (e.g., different models)."""
    seed_module.main()
    train_vle_classifiers()

    train_df = seed_module._load_vle_csv(ROOT / "data" / "vle" / "train_validate" / "csv" / "smote.csv", "train")
    sample_vle = train_df.iloc[0].to_dict()
    sample_vle.pop("label", None)

    # Test both classifiers
    for model in ["LogisticRegression", "RandomForest"]:
        label, prob = predict_pass_fail(sample_vle, model)
        assert isinstance(label, int)
        assert 0 <= prob <= 1
