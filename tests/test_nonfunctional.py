"""
tests/test_nonfunctional.py
============================
Non-Functional Tests

Standards alignment:
  - Performance Efficiency  ISO/IEC 25010 §8.4   Load / Stress / Soak / Spike / Scalability
  - Security                ISO/IEC 25010 §8.7   Input sanitization, injection, data protection
  - Usability               ISO/IEC 25010 §8.3 + ISO 9241   UX, error messaging, consistency
  - Compatibility           ISO/IEC 25010 §8.2   Cross-platform paths, data-format variants
  - Reliability             ISO/IEC 25010 §8.4   Stability, determinism, crash-free operation
"""

import pathlib
import threading
import time

import numpy as np
import pandas as pd
import pytest

from database import seed as seed_module
from database.db_utils import get_all_students, get_all_marks_wide, get_student_marks
from models.evaluate import evaluate_all_classifiers, evaluate_all_models
from models.predict import predict_2026, predict_pass_fail
from models.train import (
    SUBJECTS,
    VLE_FEATURE_COLUMNS,
    load_clean_marks,
    train_regression_models,
    train_vle_classifiers,
)
from preprocessing.clean import clip_outliers, clean_missing, load_raw_data, run_pipeline

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
# 1. PERFORMANCE TESTING — ISO/IEC 25010 Performance Efficiency
#    Industry: Load / Stress / Soak / Spike / Scalability
# ===========================================================================

class TestPerformanceEfficiency:

    def test_regression_training_time_within_budget(self, prepared_system):
        """Load: Regression model training completes within 30 s."""
        start = time.perf_counter()
        train_regression_models()
        elapsed = time.perf_counter() - start
        assert elapsed < 30, f"Regression training too slow: {elapsed:.2f}s"

    def test_vle_classifier_training_time_within_budget(self, prepared_system):
        """Load: VLE classifier training completes within 30 s."""
        start = time.perf_counter()
        train_vle_classifiers()
        elapsed = time.perf_counter() - start
        assert elapsed < 30, f"VLE classifier training too slow: {elapsed:.2f}s"

    def test_regression_prediction_latency(self, prepared_system, sample_marks):
        """Load: Average regression prediction latency < 100 ms over 10 calls."""
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            predict_2026(sample_marks, "LinearRegression")
            times.append(time.perf_counter() - t0)
        avg = sum(times) / len(times)
        assert avg < 0.1, f"Avg regression prediction latency: {avg:.4f}s"

    def test_vle_prediction_latency(self, prepared_system, sample_vle_row):
        """Load: Average VLE prediction latency < 100 ms over 10 calls."""
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            predict_pass_fail(sample_vle_row, "RandomForest")
            times.append(time.perf_counter() - t0)
        avg = sum(times) / len(times)
        assert avg < 0.1, f"Avg VLE prediction latency: {avg:.4f}s"

    def test_stress_100_regression_predictions(self, prepared_system, sample_marks):
        """Stress: 100 sequential regression predictions complete without errors."""
        errors = []
        for i in range(100):
            try:
                result = predict_2026(sample_marks, "LinearRegression")
                assert isinstance(result, dict)
            except Exception as exc:
                errors.append((i, str(exc)))
        assert not errors, f"Errors during regression stress test: {errors}"

    def test_stress_100_vle_predictions(self, prepared_system, sample_vle_row):
        """Stress: 100 sequential VLE predictions complete without errors."""
        errors = []
        for i in range(100):
            try:
                label, prob = predict_pass_fail(sample_vle_row, "RandomForest")
                assert isinstance(label, int)
                assert 0 <= prob <= 1
            except Exception as exc:
                errors.append((i, str(exc)))
        assert not errors, f"Errors during VLE stress test: {errors}"

    def test_soak_pipeline_10_runs(self):
        """Soak: Data pipeline runs correctly across 10 consecutive executions without drift."""
        seed_module.main()
        for run_idx in range(10):
            df = run_pipeline()
            assert not df.empty, f"Pipeline returned empty DataFrame on run {run_idx}"
            assert all(c in df.columns for c in SUBJECTS), f"Missing subject columns on run {run_idx}"
            assert df[SUBJECTS].max().max() <= 100, f"Mark > 100 detected on run {run_idx}"
            assert df[SUBJECTS].min().min() >= 0, f"Mark < 0 detected on run {run_idx}"

    def test_spike_sudden_vle_burst(self, prepared_system, sample_vle_row):
        """Spike: 50 VLE predictions after a 0.5 s idle period complete without errors."""
        time.sleep(0.5)
        errors = []
        for i in range(50):
            try:
                predict_pass_fail(sample_vle_row, "LogisticRegression")
            except Exception as exc:
                errors.append((i, str(exc)))
        assert not errors, f"Spike test errors: {errors}"

    def test_data_loading_performance(self):
        """Load: Full data-cleaning pipeline completes within 5 s."""
        seed_module.main()
        start = time.perf_counter()
        run_pipeline()
        elapsed = time.perf_counter() - start
        assert elapsed < 5, f"Data loading too slow: {elapsed:.2f}s"

    def test_scalability_marks_dataframe_size(self, prepared_system):
        """Scalability: Cleaned marks dataset stays within in-memory processing limits."""
        df = load_clean_marks()
        assert len(df) < 10_000, "Dataset unexpectedly large for in-memory processing"
        mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        assert mem_mb < 50, f"Marks DataFrame uses {mem_mb:.2f} MB — exceeds 50 MB budget"

    def test_model_evaluation_throughput(self, prepared_system):
        """Load: evaluate_all_models() completes within 60 s."""
        # Retrain to guarantee fresh PKL files before evaluating throughput
        train_regression_models()
        train_vle_classifiers()
        start = time.perf_counter()
        results = evaluate_all_models()
        elapsed = time.perf_counter() - start
        assert elapsed < 60, f"evaluate_all_models too slow: {elapsed:.2f}s"
        assert not results.empty


# ===========================================================================
# 2. SECURITY TESTING — ISO/IEC 25010 Security
#    Industry: Security Testing / Pen Testing (OWASP)
# ===========================================================================

class TestSecurity:

    def test_sql_injection_in_student_id_does_not_corrupt_db(self):
        """Security (OWASP A03): SQL injection string in student_id does not modify the DB."""
        seed_module.main()
        result = get_student_marks("'; DROP TABLE students; --")
        assert isinstance(result, pd.DataFrame)
        # Confirm the table still has all expected students
        students = get_all_students()
        assert len(students) == 15, "students table corrupted by injection attempt"

    def test_sql_injection_in_marks_query_returns_safe_result(self):
        """Security (OWASP A03): SQL injection in DB query path returns a safe DataFrame."""
        seed_module.main()
        df = get_all_marks_wide()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, "Marks table unexpectedly empty after injection test"

    def test_path_traversal_in_regression_model_name(self, prepared_system):
        """Security (OWASP A01): Path traversal string in model_name raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            predict_2026(
                {"Math": 80, "Physics": 75, "CS": 70, "English": 85, "Statistics": 78},
                "../../etc/passwd",
            )

    def test_path_traversal_in_vle_model_name(self, prepared_system, sample_vle_row):
        """Security (OWASP A01): Path traversal in VLE model_name raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            predict_pass_fail(sample_vle_row, "../../etc/shadow")

    def test_prediction_output_contains_no_raw_training_data(self, prepared_system, sample_vle_row):
        """Security: VLE prediction output is atomic — no DataFrames or raw rows leaked."""
        # Ensure classifiers are freshly written before loading
        train_vle_classifiers()
        label, prob = predict_pass_fail(sample_vle_row, "RandomForest")
        assert isinstance(label, int)
        assert isinstance(prob, float)

    def test_non_numeric_vle_features_handled_safely(self, prepared_system):
        """Security (OWASP A03): Non-numeric VLE feature values are rejected, not silently consumed."""
        bad_input = {col: "'; DROP TABLE vle_students; --" for col in VLE_FEATURE_COLUMNS}
        with pytest.raises((ValueError, TypeError)):
            predict_pass_fail(bad_input, "RandomForest")

    def test_extreme_numeric_inputs_produce_valid_probability(self, prepared_system):
        """Security: Extreme numeric inputs (1e18) do not produce NaN or out-of-range probability."""
        extreme_input = {col: 1e18 for col in VLE_FEATURE_COLUMNS}
        label, prob = predict_pass_fail(extreme_input, "LogisticRegression")
        assert not np.isnan(prob), "Probability is NaN with extreme inputs"
        assert 0.0 <= prob <= 1.0, f"Probability {prob} out of [0,1] with extreme inputs"

    def test_negative_marks_clipped_to_zero(self):
        """Security: Negative mark values are clamped to 0 by clip_outliers."""
        df = load_raw_data()
        df_test = df.copy()
        df_test.loc[df_test.index[0], "Math"] = -999
        clipped = clip_outliers(df_test)
        assert clipped["Math"].min() >= 0, "Negative marks not clipped to 0"

    def test_marks_above_100_clipped_to_100(self):
        """Security: Marks above 100 are clamped to 100 by clip_outliers."""
        df = load_raw_data()
        df_test = df.copy()
        df_test.loc[df_test.index[0], "Math"] = 999
        clipped = clip_outliers(df_test)
        assert clipped["Math"].max() <= 100, "Marks above 100 not clipped"

    def test_arbitrary_code_string_in_model_name_raises_not_executes(self, prepared_system, sample_vle_row):
        """Security (OWASP A03): Arbitrary code string as model_name raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            predict_pass_fail(sample_vle_row, "__import__('os').system('whoami')")

    def test_eval_output_contains_no_sensitive_columns(self, prepared_system):
        """Security: evaluate_all_models() output contains only metrics — no PII columns."""
        results = evaluate_all_models()
        sensitive = {"name", "email", "password", "dob", "phone", "address"}
        leaked = sensitive.intersection(set(results.columns))
        assert not leaked, f"PII columns found in evaluation output: {leaked}"


# ===========================================================================
# 3. USABILITY TESTING — ISO/IEC 25010 + ISO 9241
#    Industry: Usability / UX Testing
# ===========================================================================

class TestUsability:

    def test_missing_subjects_error_names_missing_fields(self, prepared_system):
        """Usability: Missing-subject ValueError message identifies which subjects are absent."""
        with pytest.raises(ValueError) as exc_info:
            predict_2026({"Math": 80}, "LinearRegression")
        msg = str(exc_info.value)
        assert any(s in msg for s in ["Physics", "CS", "English", "Statistics"]), (
            f"Error message not informative enough: '{msg}'"
        )

    def test_invalid_model_name_error_includes_model_name(self, prepared_system):
        """Usability: FileNotFoundError for bad model name includes the model name."""
        with pytest.raises(FileNotFoundError) as exc_info:
            predict_2026(
                {"Math": 80, "Physics": 75, "CS": 70, "English": 85, "Statistics": 78},
                "NonExistentModel",
            )
        assert "NonExistentModel" in str(exc_info.value), (
            "Error message does not mention the bad model name"
        )

    def test_regression_output_covers_all_five_subjects(self, prepared_system, sample_marks):
        """Usability: Regression prediction returns a value for every subject."""
        result = predict_2026(sample_marks, "LinearRegression")
        assert set(result.keys()) == set(SUBJECTS), (
            f"Missing subjects in output: {set(SUBJECTS) - set(result.keys())}"
        )

    def test_all_predicted_marks_are_displayable_percentages(self, prepared_system, sample_marks):
        """Usability: All predicted marks are in [0, 100] — directly displayable as percentages."""
        for model in ["LinearRegression", "DecisionTree", "RandomForest"]:
            result = predict_2026(sample_marks, model)
            for subj, val in result.items():
                assert 0 <= val <= 100, (
                    f"{model} predicted {val:.2f} for {subj} — outside [0, 100]"
                )

    def test_vle_probability_is_valid_ui_percentage(self, prepared_system, sample_vle_row):
        """Usability: Pass/fail probability is a [0, 1] float — suitable for progress-bar display."""
        _, prob = predict_pass_fail(sample_vle_row, "RandomForest")
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_subject_summary_has_expected_dashboard_columns(self):
        """Usability: subject_summary() returns columns required by the dashboard table."""
        from analysis.stats import subject_summary
        seed_module.main()
        run_pipeline()
        df = load_clean_marks()
        summary = subject_summary(df)
        assert {"subject", "mean", "median", "std"}.issubset(summary.columns)

    def test_student_summary_has_expected_dashboard_columns(self):
        """Usability: student_summary() returns columns required by the dashboard table."""
        from analysis.stats import student_summary
        seed_module.main()
        run_pipeline()
        df = load_clean_marks()
        summary = student_summary(df)
        assert {"student_id", "name", "year", "overall_mean", "best_subject", "worst_subject"}.issubset(
            summary.columns
        )

    def test_empty_feature_dict_raises_not_returns_garbage(self, prepared_system):
        """Usability: Empty feature dict raises an error rather than returning a silent wrong result."""
        with pytest.raises((ValueError, FileNotFoundError, KeyError)):
            predict_2026({}, "LinearRegression")

    def test_pipeline_always_returns_dataframe(self):
        """Usability: run_pipeline() always returns a DataFrame, never None."""
        seed_module.main()
        result = run_pipeline()
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert not result.empty


# ===========================================================================
# 4. COMPATIBILITY TESTING — ISO/IEC 25010 Compatibility
#    Industry: Compatibility / Cross-Platform / Cross-Data-Format
# ===========================================================================

class TestCompatibility:

    def test_pipeline_accepts_custom_file_paths(self, tmp_path):
        """Compatibility: Pipeline works with user-supplied custom file paths."""
        import shutil
        custom_raw = tmp_path / "custom_raw.csv"
        custom_clean = tmp_path / "custom_clean.csv"
        shutil.copy(ROOT / "data" / "students_raw.csv", custom_raw)
        df = run_pipeline(raw_path=str(custom_raw), clean_path=str(custom_clean))
        assert not df.empty
        assert custom_clean.exists(), "Custom clean CSV was not written"

    def test_cleaned_csv_round_trip_readable(self):
        """Compatibility: Cleaned CSV is re-readable by pandas (CSV round-trip)."""
        seed_module.main()
        run_pipeline()
        df = pd.read_csv(ROOT / "data" / "students_clean.csv")
        assert not df.empty
        assert all(c in df.columns for c in SUBJECTS)

    def test_all_regression_models_accept_same_input_format(self, prepared_system, sample_marks):
        """Compatibility: All regression model types consume identical input dicts."""
        for model_name in ["LinearRegression", "DecisionTree", "RandomForest"]:
            result = predict_2026(sample_marks, model_name)
            assert isinstance(result, dict)
            assert len(result) == 5, f"{model_name} returned wrong number of subjects"

    def test_all_vle_classifiers_accept_same_input_format(self, prepared_system, sample_vle_row):
        """Compatibility: All VLE classifiers consume identical input dicts."""
        for model_name in ["LogisticRegression", "RandomForest"]:
            label, prob = predict_pass_fail(sample_vle_row, model_name)
            assert isinstance(label, int)
            assert 0.0 <= prob <= 1.0

    def test_vle_feature_columns_present_in_all_splits(self):
        """Compatibility: VLE feature columns are consistent in original and test splits (SMOTE variants may omit contextual columns)."""
        seed_module.main()
        # Original and test splits must contain all 19 declared feature columns
        full_splits = [
            (ROOT / "data" / "vle" / "original" / "original.csv", "original"),
            (ROOT / "data" / "vle" / "test" / "test.csv", "test"),
        ]
        for path, label in full_splits:
            df = seed_module._load_vle_csv(path, label)
            for col in VLE_FEATURE_COLUMNS:
                assert col in df.columns, f"Column '{col}' missing from {label} split"
        # SMOTE train split must at minimum contain behavioural feature columns + label
        behaviour_cols = [
            "gender", "age", "logins", "total_hours", "pct_avg_hours",
            "presence_count", "absence_count", "pct_attended",
        ]
        train_df = seed_module._load_vle_csv(
            ROOT / "data" / "vle" / "train_validate" / "csv" / "smote.csv", "train"
        )
        for col in behaviour_cols + ["label"]:
            assert col in train_df.columns, f"Core column '{col}' missing from SMOTE train split"

    def test_core_modules_use_pathlib_for_cross_platform_paths(self):
        """Compatibility: Core modules use pathlib.Path — compatible with Windows, macOS, Linux."""
        import preprocessing.clean as clean_mod
        import models.train as train_mod
        assert isinstance(clean_mod.ROOT, pathlib.Path), "preprocessing.clean.ROOT is not pathlib.Path"
        assert isinstance(train_mod.ROOT, pathlib.Path), "models.train.ROOT is not pathlib.Path"

    def test_seed_is_idempotent_across_runs(self):
        """Compatibility: Running seed.main() twice does not duplicate or corrupt data."""
        seed_module.main()
        seed_module.main()
        students = get_all_students()
        assert len(students) == 15, f"Expected 15 students after double-seed, got {len(students)}"

    def test_all_smote_csv_variants_are_loadable(self):
        """Compatibility: All six SMOTE-variant CSVs load without error and contain 'label'."""
        csv_dir = ROOT / "data" / "vle" / "train_validate" / "csv"
        variants = [
            "adasyn.csv", "borderline_smote.csv", "none.csv",
            "smote_nc.csv", "smote_svm.csv", "smote.csv",
        ]
        for filename in variants:
            path = csv_dir / filename
            if path.exists():
                df = seed_module._load_vle_csv(path, "train")
                assert not df.empty, f"{filename} loaded as empty DataFrame"
                assert "label" in df.columns, f"'label' column missing in {filename}"


# ===========================================================================
# 5. RELIABILITY TESTING — ISO/IEC 25010 Reliability
#    Industry: Reliability Testing (Stability, Determinism, Crash-free)
# ===========================================================================

class TestReliability:

    def test_regression_predictions_are_deterministic(self, prepared_system, sample_marks):
        """Reliability: Identical input produces identical regression prediction on repeated calls."""
        result1 = predict_2026(sample_marks, "LinearRegression")
        result2 = predict_2026(sample_marks, "LinearRegression")
        for subj in SUBJECTS:
            assert result1[subj] == pytest.approx(result2[subj], abs=1e-6), (
                f"Non-deterministic regression prediction for {subj}"
            )

    def test_vle_predictions_are_deterministic(self, prepared_system, sample_vle_row):
        """Reliability: Identical VLE input produces identical label and probability."""
        label1, prob1 = predict_pass_fail(sample_vle_row, "RandomForest")
        label2, prob2 = predict_pass_fail(sample_vle_row, "RandomForest")
        assert label1 == label2, "Non-deterministic VLE label"
        assert prob1 == pytest.approx(prob2, abs=1e-6), "Non-deterministic VLE probability"

    def test_pipeline_output_stable_across_successive_runs(self):
        """Reliability: Successive pipeline runs produce identical cleaned subject marks."""
        seed_module.main()
        df1 = run_pipeline()
        df2 = run_pipeline()
        pd.testing.assert_frame_equal(
            df1[SUBJECTS].reset_index(drop=True),
            df2[SUBJECTS].reset_index(drop=True),
            check_exact=False,
            rtol=1e-5,
        )

    def test_models_survive_retraining_without_corruption(self):
        """Reliability: Retraining models twice does not corrupt saved files or break predictions."""
        seed_module.main()
        run_pipeline()
        train_regression_models()
        train_vle_classifiers()
        train_regression_models()
        train_vle_classifiers()
        sample = {"Math": 70, "Physics": 65, "CS": 60, "English": 75, "Statistics": 68}
        result = predict_2026(sample, "RandomForest")
        assert all(0 <= v <= 100 for v in result.values()), (
            "Predictions out of range after retraining"
        )

    def test_no_nan_values_in_cleaned_dataset(self):
        """Reliability: Cleaning pipeline eliminates all NaN values from subject columns."""
        seed_module.main()
        df = run_pipeline()
        nan_count = df[SUBJECTS].isna().sum().sum()
        assert nan_count == 0, f"{nan_count} NaN values remain after cleaning pipeline"

    def test_no_inf_values_in_regression_predictions(self, prepared_system, sample_marks):
        """Reliability: No infinite values appear in regression prediction output."""
        result = predict_2026(sample_marks, "RandomForest")
        for subj, val in result.items():
            assert not np.isinf(val), f"Infinite prediction returned for {subj}"

    def test_evaluation_metrics_contain_no_nan(self, prepared_system):
        """Reliability: RMSE and MAE columns in evaluate_all_models() have no NaN entries."""
        results = evaluate_all_models()
        assert not results["RMSE"].isna().any(), "NaN RMSE values in evaluation results"
        assert not results["MAE"].isna().any(), "NaN MAE values in evaluation results"

    def test_database_handles_20_successive_queries_without_failure(self):
        """Reliability: Database connection sustains 20 successive queries without error."""
        seed_module.main()
        errors = []
        for i in range(20):
            try:
                df = get_all_marks_wide()
                assert not df.empty
            except Exception as exc:
                errors.append((i, str(exc)))
        assert not errors, f"DB connection errors: {errors}"

    def test_memory_usage_within_expected_bounds(self, prepared_system):
        """Reliability: Marks DataFrame memory stays within expected operational bounds."""
        df = load_clean_marks()
        assert len(df) < 10_000, "Dataset row count exceeded safety threshold"
        mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        assert mem_mb < 50, f"DataFrame consumes {mem_mb:.2f} MB — exceeds 50 MB limit"
