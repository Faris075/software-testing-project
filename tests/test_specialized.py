"""
tests/test_specialized.py
==========================
Specialized Tests

Standards alignment:
  - Installation Testing    IEEE 29119-1              Install / uninstall / configuration checks
  - Accessibility Testing   ISO/IEC 25010 Usability   Screen-reader proxy, WCAG, Section 508
  - Mutation Testing        IEEE 29119-4              Fault injection — test-suite sensitivity
  - Concurrency Testing     ISO/IEC 25010 Compat.     Race conditions, simultaneous access
  - Compliance Testing      ISO/IEC 25010 Security    GDPR, data minimisation, anonymisation
"""

import json
import pathlib
import threading

import numpy as np
import pandas as pd
import pytest

from database import seed as seed_module
from database.db_utils import get_all_students, get_all_marks_wide
from models.evaluate import evaluate_all_models, evaluate_all_classifiers
from models.predict import predict_2026, predict_pass_fail
from models.train import (
    SUBJECTS,
    VLE_FEATURE_COLUMNS,
    load_clean_marks,
    train_regression_models,
    train_vle_classifiers,
)
from preprocessing.clean import (
    clip_outliers,
    clean_missing,
    encode_year,
    load_raw_data,
    run_pipeline,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def prepared_system():
    """Seed DB, run pipeline, and train all models once per module."""
    seed_module.main()
    run_pipeline()
    train_regression_models()
    train_vle_classifiers()


@pytest.fixture(scope="module")
def sample_vle_row():
    train_df = seed_module._load_vle_csv(
        ROOT / "data" / "vle" / "train_validate" / "csv" / "smote.csv", "train"
    )
    row = train_df.iloc[0].to_dict()
    row.pop("label", None)
    return row


@pytest.fixture(scope="module")
def sample_marks():
    return {"Math": 80, "Physics": 75, "CS": 70, "English": 85, "Statistics": 78}


# ===========================================================================
# 1. INSTALLATION TESTING — IEEE 29119-1
#    Industry: Installation Testing (setup / configuration)
# ===========================================================================

class TestInstallation:

    def test_required_data_directories_exist(self):
        """Installation: Core data directories are present after project setup."""
        assert (ROOT / "data").is_dir(), "data/ directory missing"
        assert (ROOT / "data" / "vle").is_dir(), "data/vle/ directory missing"
        assert (ROOT / "data" / "vle" / "train_validate" / "csv").is_dir(), (
            "data/vle/train_validate/csv/ directory missing"
        )

    def test_required_source_files_exist(self):
        """Installation: All required source modules are present on disk."""
        required = [
            ROOT / "preprocessing" / "clean.py",
            ROOT / "models" / "train.py",
            ROOT / "models" / "predict.py",
            ROOT / "models" / "evaluate.py",
            ROOT / "database" / "seed.py",
            ROOT / "database" / "db_utils.py",
            ROOT / "database" / "schema.sql",
            ROOT / "analysis" / "stats.py",
            ROOT / "visualisation" / "charts.py",
            ROOT / "dashboard" / "app.py",
        ]
        for path in required:
            assert path.exists(), f"Required file missing: {path.relative_to(ROOT)}"

    def test_required_vle_csv_files_exist(self):
        """Installation: Required VLE CSV data files are present."""
        required = [
            ROOT / "data" / "vle" / "original" / "original.csv",
            ROOT / "data" / "vle" / "test" / "test.csv",
            ROOT / "data" / "vle" / "train_validate" / "csv" / "smote.csv",
        ]
        for path in required:
            assert path.exists(), f"Required VLE file missing: {path.relative_to(ROOT)}"

    def test_all_python_dependencies_importable(self):
        """Installation: All declared runtime dependencies can be imported."""
        import pandas        # noqa: F401
        import numpy         # noqa: F401
        import sklearn       # noqa: F401
        import joblib        # noqa: F401
        import matplotlib    # noqa: F401
        import seaborn       # noqa: F401
        import reportlab     # noqa: F401
        import streamlit     # noqa: F401
        import sqlalchemy    # noqa: F401
        import scipy         # noqa: F401

    def test_database_schema_applies_cleanly(self, tmp_path):
        """Installation: Database schema can be applied to a fresh SQLite file."""
        import sqlite3
        db_path = tmp_path / "test_fresh.db"
        schema_path = ROOT / "database" / "schema.sql"
        conn = sqlite3.connect(db_path)
        with open(schema_path, "r", encoding="utf-8") as fh:
            conn.executescript(fh.read())
        conn.commit()
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        assert "students" in tables
        assert "marks" in tables
        assert "vle_students" in tables

    def test_saved_models_directory_created_automatically(self):
        """Installation: saved_models/ directory is created automatically by train module."""
        models_dir = ROOT / "saved_models"
        seed_module.main()
        run_pipeline()
        train_regression_models()
        assert models_dir.is_dir(), "saved_models/ was not created by training"

    def test_models_trainable_from_seeded_state(self):
        """Installation: Full training pipeline executes end-to-end without manual intervention."""
        seed_module.main()
        run_pipeline()
        train_regression_models()
        train_vle_classifiers()
        # Verify at least one model file was created
        pkl_files = list((ROOT / "saved_models").glob("*.pkl"))
        assert len(pkl_files) > 0, "No .pkl model files found after training"

    def test_requirements_txt_exists_and_non_empty(self):
        """Installation: requirements.txt is present and lists at least 5 packages."""
        req_path = ROOT / "requirements.txt"
        assert req_path.exists(), "requirements.txt not found"
        packages = [
            line.strip() for line in req_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        assert len(packages) >= 5, f"requirements.txt has only {len(packages)} entries"


# ===========================================================================
# 2. ACCESSIBILITY TESTING — ISO/IEC 25010 Usability (disability-support)
#    Industry: Accessibility / a11y (WCAG 2.1, Section 508)
# ===========================================================================

class TestAccessibility:

    def test_regression_prediction_is_json_serializable(self, prepared_system, sample_marks):
        """Accessibility (WCAG 1.1): Prediction output is JSON-serializable for API/screen-reader delivery."""
        result = predict_2026(sample_marks, "LinearRegression")
        serialized = json.dumps(result)
        assert serialized is not None
        parsed = json.loads(serialized)
        assert set(parsed.keys()) == set(SUBJECTS)

    def test_vle_prediction_is_json_serializable(self, prepared_system, sample_vle_row):
        """Accessibility (WCAG 1.1): VLE prediction output is JSON-serializable."""
        label, prob = predict_pass_fail(sample_vle_row, "RandomForest")
        serialized = json.dumps({"label": label, "probability": prob})
        assert serialized is not None
        parsed = json.loads(serialized)
        assert "label" in parsed
        assert "probability" in parsed

    def test_subject_summary_values_have_no_nan(self):
        """Accessibility (WCAG 1.3): Statistical summaries are NaN-free — displayable to all users."""
        from analysis.stats import subject_summary
        seed_module.main()
        run_pipeline()
        df = load_clean_marks()
        summary = subject_summary(df)
        assert not summary["mean"].isna().any(), "NaN in subject mean"
        assert not summary["median"].isna().any(), "NaN in subject median"
        assert not summary["std"].isna().any(), "NaN in subject std"

    def test_student_summary_has_no_nan_overall_mean(self):
        """Accessibility (WCAG 1.3): Student summary overall_mean contains no NaN values."""
        from analysis.stats import student_summary
        seed_module.main()
        run_pipeline()
        df = load_clean_marks()
        summary = student_summary(df)
        assert not summary["overall_mean"].isna().any(), "NaN in student overall_mean"

    def test_predicted_values_are_finite_and_representable(self, prepared_system, sample_marks):
        """Accessibility (Section 508): Predicted values are finite numbers — renderable in assistive tech."""
        result = predict_2026(sample_marks, "RandomForest")
        for subj, val in result.items():
            assert np.isfinite(val), f"Non-finite prediction for {subj}: {val}"

    def test_vle_probability_is_finite(self, prepared_system, sample_vle_row):
        """Accessibility: Pass/fail probability is a finite float — suitable for aria-valuenow."""
        _, prob = predict_pass_fail(sample_vle_row, "RandomForest")
        assert np.isfinite(prob), f"Non-finite probability: {prob}"

    def test_eval_results_saved_to_csv_for_screen_reader_access(self, prepared_system):
        """Accessibility: Evaluation results are persisted as CSV — accessible beyond graphical UI."""
        results = evaluate_all_models()
        out_path = ROOT / "outputs" / "eval_results.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_path, index=False)
        assert out_path.exists(), "eval_results.csv not written"
        df_check = pd.read_csv(out_path)
        assert not df_check.empty

    def test_prediction_labels_have_human_readable_values(self, prepared_system, sample_vle_row):
        """Accessibility: VLE label is 0 (pass) or 1 (fail) — maps directly to readable text."""
        label, _ = predict_pass_fail(sample_vle_row, "RandomForest")
        assert label in {0, 1}, (
            f"Label {label} cannot be mapped to human-readable pass/fail text"
        )


# ===========================================================================
# 3. MUTATION TESTING — IEEE 29119-4 (Mutation Analysis)
#    Industry: Mutation Testing — verify test-suite sensitivity to code faults
# ===========================================================================

class TestMutation:

    def test_kill_mutant_clip_lower_bound(self):
        """Mutation (IEEE 29119-4): Off-by-one at lower clip bound (0→1) would be detected."""
        df = load_raw_data()
        df_test = df.copy()
        df_test.loc[df_test.index[0], "Math"] = 0.0   # exactly at boundary
        clipped = clip_outliers(df_test)
        # A mutant using clip(1, 100) would set this to 1 — test kills that mutant
        assert clipped.loc[df_test.index[0], "Math"] == 0.0, (
            "Mark of 0 should NOT be clipped — boundary mutant detected"
        )

    def test_kill_mutant_clip_upper_bound(self):
        """Mutation (IEEE 29119-4): Off-by-one at upper clip bound (100→99) would be detected."""
        df = load_raw_data()
        df_test = df.copy()
        df_test.loc[df_test.index[0], "Math"] = 100.0   # exactly at boundary
        clipped = clip_outliers(df_test)
        # A mutant using clip(0, 99) would set this to 99 — test kills that mutant
        assert clipped.loc[df_test.index[0], "Math"] == 100.0, (
            "Mark of 100 should NOT be clipped — boundary mutant detected"
        )

    def test_kill_mutant_missing_subject_guard_exists(self, prepared_system):
        """Mutation: Removing the 'missing subjects' guard would silently accept partial input — test kills that mutant."""
        with pytest.raises(ValueError):
            predict_2026({"Math": 80}, "LinearRegression")

    def test_kill_mutant_probability_strictly_bounded(self, prepared_system, sample_vle_row):
        """Mutation: Negated range checks (prob >= 0, prob <= 1) must both hold independently."""
        _, prob = predict_pass_fail(sample_vle_row, "LogisticRegression")
        assert prob >= 0.0, f"Probability {prob} is below 0 (lower-bound mutant)"
        assert prob <= 1.0, f"Probability {prob} is above 1 (upper-bound mutant)"
        assert not (prob > 1.0), "Negated upper-bound condition must be false"
        assert not (prob < 0.0), "Negated lower-bound condition must be false"

    def test_kill_mutant_label_is_strictly_binary(self, prepared_system, sample_vle_row):
        """Mutation: Label must be 0 or 1 — continuous or off-by-one mutant would be killed."""
        label, _ = predict_pass_fail(sample_vle_row, "RandomForest")
        assert label in {0, 1}, f"Label {label} is not in {{0, 1}}"
        assert label != 2, "Label must not be 2"
        assert label != -1, "Label must not be -1"

    def test_kill_mutant_subject_count_exactly_five(self, prepared_system, sample_marks):
        """Mutation: Regression output must have exactly 5 subjects — count mutant detected."""
        result = predict_2026(sample_marks, "LinearRegression")
        assert len(result) == 5, f"Expected 5 subjects, got {len(result)}"
        assert len(result) != 4, "Output must not have 4 subjects"
        assert len(result) != 6, "Output must not have 6 subjects"

    def test_kill_mutant_imputation_uses_mean_not_zero(self):
        """Mutation: Imputation must use column mean — replacing with 0 would be killed by this test."""
        df = pd.DataFrame({
            "student_id": ["S001", "S002", "S003"],
            "name": ["A", "B", "C"],
            "year": [2024, 2024, 2024],
            "Math": [70.0, 80.0, float("nan")],
            "Physics": [60.0, 70.0, 65.0],
            "CS": [50.0, 60.0, 55.0],
            "English": [80.0, 90.0, 85.0],
            "Statistics": [75.0, 85.0, 80.0],
        })
        cleaned = clean_missing(df)
        expected_mean = (70.0 + 80.0) / 2   # mean of non-NaN Math for year 2024
        assert cleaned.loc[2, "Math"] == pytest.approx(expected_mean, abs=0.01), (
            f"Imputed {cleaned.loc[2, 'Math']:.2f} — expected mean {expected_mean:.2f}; "
            "zero-fill mutant would produce 0.0"
        )

    def test_kill_mutant_year_encoding_not_swapped(self):
        """Mutation: Year encoding 2024→0, 2025→1 — swapped mutant would be killed."""
        df = pd.DataFrame({"year": [2024, 2025]})
        result = encode_year(df)
        assert result.loc[result["year"] == 2024, "year_encoded"].iloc[0] == 0, (
            "2024 must encode to 0 — swapped-encoding mutant detected"
        )
        assert result.loc[result["year"] == 2025, "year_encoded"].iloc[0] == 1, (
            "2025 must encode to 1 — swapped-encoding mutant detected"
        )

    def test_kill_mutant_prediction_changes_with_different_input(self, prepared_system):
        """Mutation: Prediction must change when input marks change — constant-return mutant killed."""
        result_high = predict_2026(
            {"Math": 95, "Physics": 95, "CS": 95, "English": 95, "Statistics": 95},
            "LinearRegression",
        )
        result_low = predict_2026(
            {"Math": 20, "Physics": 20, "CS": 20, "English": 20, "Statistics": 20},
            "LinearRegression",
        )
        # At least one subject must predict differently for different inputs
        changed = any(result_high[s] != result_low[s] for s in SUBJECTS)
        assert changed, "Model returns identical predictions for very different inputs — constant-return mutant"


# ===========================================================================
# 4. CONCURRENCY TESTING — ISO/IEC 25010 Compatibility (simultaneous access)
#    Industry: Concurrency Testing (race conditions, thread safety)
# ===========================================================================

class TestConcurrency:

    def test_concurrent_vle_predictions_no_race_condition(self, prepared_system, sample_vle_row):
        """Concurrency: 10 simultaneous VLE predictions return consistent, error-free results."""
        results = []
        errors = []

        def predict():
            try:
                label, prob = predict_pass_fail(sample_vle_row, "RandomForest")
                results.append((label, prob))
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=predict) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors during concurrent VLE prediction: {errors}"
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"
        # All threads must agree on the label (deterministic model)
        labels = {r[0] for r in results}
        assert len(labels) == 1, f"Non-deterministic labels under concurrency: {labels}"

    def test_concurrent_regression_predictions_no_race_condition(self, prepared_system, sample_marks):
        """Concurrency: 10 simultaneous regression predictions return consistent results."""
        results = []
        errors = []

        def predict():
            try:
                result = predict_2026(sample_marks, "LinearRegression")
                results.append(result)
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=predict) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors during concurrent regression prediction: {errors}"
        assert len(results) == 10
        # All threads must agree on Math prediction
        math_preds = {round(r["Math"], 4) for r in results}
        assert len(math_preds) == 1, f"Non-deterministic Math predictions: {math_preds}"

    def test_concurrent_db_reads_no_data_corruption(self):
        """Concurrency: Simultaneous DB reads from 10 threads return correct row counts."""
        seed_module.main()
        results = []
        errors = []

        def read_db():
            try:
                df = get_all_marks_wide()
                results.append(len(df))
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=read_db) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"DB read errors under concurrency: {errors}"
        assert len(results) == 10
        # All threads must see the same number of rows
        assert len(set(results)) == 1, (
            f"Inconsistent row counts under concurrent reads: {set(results)}"
        )

    def test_concurrent_pipeline_runs_produce_consistent_output(self):
        """Concurrency: Two simultaneous pipeline executions (different output paths) don't interfere."""
        import shutil
        seed_module.main()
        results = {}
        errors = []

        def run(key, tmp_out):
            try:
                src = ROOT / "data" / "students_raw.csv"
                tmp_raw = pathlib.Path(tmp_out) / "raw.csv"
                tmp_clean = pathlib.Path(tmp_out) / "clean.csv"
                pathlib.Path(tmp_out).mkdir(parents=True, exist_ok=True)
                shutil.copy(src, tmp_raw)
                df = run_pipeline(raw_path=str(tmp_raw), clean_path=str(tmp_clean))
                results[key] = df[SUBJECTS].values.tolist()
            except Exception as exc:
                errors.append(str(exc))

        import tempfile
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            thread_a = threading.Thread(target=run, args=("A", t1))
            thread_b = threading.Thread(target=run, args=("B", t2))
            thread_a.start()
            thread_b.start()
            thread_a.join()
            thread_b.join()

        assert not errors, f"Concurrent pipeline errors: {errors}"
        assert "A" in results and "B" in results
        assert results["A"] == results["B"], (
            "Concurrent pipelines produced different cleaned data — possible race condition"
        )

    def test_concurrent_model_evaluations_no_crash(self, prepared_system):
        """Concurrency: Multiple threads calling evaluate_all_models() simultaneously do not crash."""
        errors = []

        def evaluate():
            try:
                evaluate_all_models()
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=evaluate) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent model evaluation: {errors}"


# ===========================================================================
# 5. COMPLIANCE TESTING — ISO/IEC 25010 Security (regulatory scope)
#    Industry: Compliance Testing (GDPR, HIPAA, PCI DSS, SOC 2)
# ===========================================================================

class TestCompliance:

    def test_gdpr_student_ids_are_anonymised_codes(self):
        """Compliance (GDPR Art.4): Student IDs follow anonymous S-code format, not real names or emails."""
        seed_module.main()
        students = get_all_students()
        import re
        pattern = re.compile(r"^S\d{3}$")
        for sid in students["student_id"]:
            assert pattern.match(sid), (
                f"Student ID '{sid}' does not match anonymous S-code pattern (GDPR pseudonymisation)"
            )

    def test_gdpr_prediction_output_contains_no_pii(self, prepared_system, sample_marks):
        """Compliance (GDPR Art.5): Regression prediction output contains no personally identifiable information."""
        result = predict_2026(sample_marks, "LinearRegression")
        for key in result:
            # Keys must be subject names, not personal identifiers
            assert key in SUBJECTS, f"Unexpected key '{key}' in prediction output"
        # Values must be numeric marks, not strings that could encode PII
        for val in result.values():
            assert isinstance(val, float), f"Non-float value in prediction: {val}"

    def test_gdpr_vle_features_contain_no_direct_identifiers(self):
        """Compliance (GDPR Art.25 data minimisation): VLE feature columns contain no direct identifiers."""
        pii_indicators = {"name", "email", "phone", "dob", "national_id", "passport", "address"}
        leaked = pii_indicators.intersection({col.lower() for col in VLE_FEATURE_COLUMNS})
        assert not leaked, (
            f"VLE feature columns contain potential direct identifiers: {leaked}"
        )

    def test_gdpr_data_minimisation_vle_model_uses_only_declared_features(self, prepared_system, sample_vle_row):
        """Compliance (GDPR Art.5 data minimisation): VLE model processes only the 19 declared features."""
        # Provide extra undeclared keys alongside the declared ones
        extended_input = dict(sample_vle_row)
        extended_input["secret_flag"] = 99
        extended_input["internal_notes"] = "sensitive"
        # The model should silently ignore undeclared keys (uses .get(col, 0) for declared cols only)
        label, prob = predict_pass_fail(extended_input, "RandomForest")
        assert isinstance(label, int)
        assert 0.0 <= prob <= 1.0

    def test_gdpr_right_to_erasure_predictions_deletable(self):
        """Compliance (GDPR Art.17 right to erasure): Stored predictions can be deleted from the DB."""
        from sqlalchemy import create_engine, text
        seed_module.main()
        run_pipeline()
        train_regression_models()

        engine = create_engine(f"sqlite:///{ROOT / 'database' / 'students.db'}")
        # Insert a test prediction
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO predictions (student_id, subject, predicted_mark, model_used) "
                    "VALUES (:sid, :subj, :mark, :model)"
                ),
                {"sid": "S001", "subj": "Math", "mark": 75.0, "model": "TestModel"},
            )
        # Verify it was inserted
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT COUNT(*) FROM predictions WHERE model_used = 'TestModel'")
            ).scalar()
        assert rows >= 1, "Test prediction was not inserted"

        # Delete (erasure)
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM predictions WHERE model_used = 'TestModel'")
            )
        with engine.connect() as conn:
            remaining = conn.execute(
                text("SELECT COUNT(*) FROM predictions WHERE model_used = 'TestModel'")
            ).scalar()
        assert remaining == 0, "Predictions were not fully erased (GDPR Art.17 violation)"

    def test_compliance_no_credentials_in_source_files(self):
        """Compliance (SOC 2 / PCI DSS): Source files contain no hardcoded passwords or API keys."""
        import re
        # Pattern catches obvious hardcoded credential assignments
        credential_pattern = re.compile(
            r'(password|passwd|secret|api_key|token)\s*=\s*["\'][^"\']{4,}["\']',
            re.IGNORECASE,
        )
        source_dirs = [
            ROOT / "database",
            ROOT / "models",
            ROOT / "preprocessing",
            ROOT / "analysis",
            ROOT / "visualisation",
        ]
        matches = []
        for src_dir in source_dirs:
            for py_file in src_dir.glob("*.py"):
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                for match in credential_pattern.finditer(content):
                    matches.append(f"{py_file.relative_to(ROOT)}: {match.group()}")
        assert not matches, (
            f"Potential hardcoded credentials found (SOC 2 violation): {matches}"
        )

    def test_compliance_student_names_not_leaked_in_model_output(self, prepared_system, sample_marks):
        """Compliance (GDPR Art.5): Student names from training data are not present in model outputs."""
        seed_module.main()
        students = get_all_students()
        real_names = set(students["name"].str.lower())

        result = predict_2026(sample_marks, "LinearRegression")
        output_str = json.dumps(result).lower()
        leaked_names = [name for name in real_names if name in output_str]
        assert not leaked_names, (
            f"Student names leaked in prediction output: {leaked_names}"
        )
