import pathlib
import streamlit as st
from pandas import DataFrame

from analysis.stats import (
    correlation_matrix,
    failing_students,
    student_summary,
    subject_summary,
    top_performers,
    year_on_year_change,
)
from models.evaluate import evaluate_all_classifiers, evaluate_all_models
from models.predict import predict_2026, predict_pass_fail
from models.train import (
    CLASSIFIERS,
    SUBJECTS,
    build_regression_dataset,
    get_vle_feature_columns,
    load_clean_marks,
    load_vle_train_test,
)
from report.generate_pdf import generate_report
from visualisation.charts import (
    bar_predictions_2026,
    bar_student_averages,
    bar_subject_averages,
    bar_vle_class_distribution,
    bar_feature_importance,
    comparison_bar,
    heatmap_correlation,
    heatmap_vle_correlation,
    line_class_progress,
    line_progress,
    plot_confusion_matrix,
    plot_roc_curve,
    scatter_actual_vs_predicted,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _safe_load_vle_original() -> DataFrame:
    try:
        train_df, test_df = load_vle_train_test()
        return train_df
    except Exception:
        return DataFrame()


def _build_vle_feature_row() -> dict:
    feature_row = {}
    feature_row["gender"] = st.selectbox("Gender", [0, 1], index=0)
    feature_row["age"] = st.number_input("Age", min_value=16, max_value=60, value=20)
    feature_row["logins"] = st.number_input("Logins", min_value=0, max_value=200, value=10)
    feature_row["total_hours"] = st.number_input("Total hours in module area", min_value=0.0, max_value=500.0, value=10.0)
    feature_row["pct_avg_hours"] = st.number_input("% of average hours", min_value=0.0, max_value=5.0, value=0.5)
    feature_row["presence_count"] = st.number_input("Presence count", min_value=0, max_value=20, value=8)
    feature_row["absence_count"] = st.number_input("Absence count", min_value=0, max_value=20, value=2)
    feature_row["pct_attended"] = st.number_input("% attended", min_value=0.0, max_value=100.0, value=80.0)
    feature_row["attending_from_home"] = st.selectbox("Attending from home?", [1, 0], index=0)
    feature_row["distance_to_uni_km"] = st.number_input("Distance to university (km)", min_value=0.0, max_value=200.0, value=10.0)
    feature_row["polar4_quintile"] = st.selectbox("POLAR4 quintile", [1, 2, 3, 4, 5], index=2)
    feature_row["polar3_quintile"] = st.selectbox("POLAR3 quintile", [1, 2, 3, 4, 5], index=2)
    feature_row["adult_he_2001_quintile"] = st.selectbox("Adult HE 2001 quintile", [1, 2, 3, 4, 5], index=2)
    feature_row["adult_he_2011_quintile"] = st.selectbox("Adult HE 2011 quintile", [1, 2, 3, 4, 5], index=2)
    feature_row["tundra_msoa_quintile"] = st.selectbox("TUNDRA MSOA quintile", [1, 2, 3, 4, 5], index=2)
    feature_row["tundra_lsoa_quintile"] = st.selectbox("TUNDRA LSOA quintile", [1, 2, 3, 4, 5], index=2)
    feature_row["gaps_gcse_quintile"] = st.selectbox("GCSE gap quintile", [1, 2, 3, 4, 5], index=2)
    feature_row["gaps_gcse_ethnicity_quintile"] = st.selectbox("GCSE ethnicity gap quintile", [1, 2, 3, 4, 5], index=2)
    feature_row["uni_connect_target_ward"] = st.selectbox("Uni Connect target ward", [0, 1], index=0)
    return feature_row


def main() -> None:
    st.set_page_config(page_title="Student Analytics Dashboard", layout="wide")
    st.title("AI-Powered Student Performance Analytics")

    dataset_option = st.sidebar.selectbox("Select dataset", ["Marks Cohort", "VLE Pass/Fail"])

    marks_df = load_clean_marks()
    students = sorted(marks_df["student_id"].unique())
    vle_df = _safe_load_vle_original()

    if dataset_option == "Marks Cohort":
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Student Profile", "2026 Predictions", "Export"])

        with tab1:
            st.header("Marks cohort overview")

            # KPI cards
            summ_df = student_summary(marks_df)
            class_avg = round(marks_df[SUBJECTS].mean().mean(), 2)
            top_df = top_performers(marks_df, n=1)
            top_name = top_df.iloc[0]["name"] if not top_df.empty else "—"
            top_avg = top_df.iloc[0]["overall_mean"] if not top_df.empty else 0

            yoy_df = year_on_year_change(marks_df)
            if not yoy_df.empty:
                best_imp_idx = yoy_df["overall_delta"].idxmax()
                most_improved = yoy_df.loc[best_imp_idx, "name"]
                most_improved_delta = yoy_df.loc[best_imp_idx, "overall_delta"]
            else:
                most_improved = "—"
                most_improved_delta = 0

            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Class average", f"{class_avg:.1f}")
            kpi2.metric("Top student", top_name, f"avg {top_avg:.1f}")
            kpi3.metric("Most improved", str(most_improved), f"Δ {most_improved_delta:+.2f}")

            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Subject averages")
                fig = bar_subject_averages(marks_df)
                st.pyplot(fig)
            with col2:
                st.subheader("Subject correlation")
                corr = correlation_matrix(marks_df)
                fig2 = heatmap_correlation(corr)
                st.pyplot(fig2)

        with tab2:
            student_id = str(st.selectbox("Select student", students))
            st.header(f"Profile: {student_id}")
            student_rows = marks_df[marks_df["student_id"] == student_id].sort_values("year")

            # Stats tables
            s_summ = summ_df[summ_df["student_id"] == student_id][
                ["year", "overall_mean", "best_subject", "worst_subject"]
            ]
            st.subheader("Year summary")
            st.table(s_summ.set_index("year"))

            st.subheader("Subject-level stats (all cohort)")
            st.table(subject_summary(marks_df).set_index("subject"))

            st.subheader("Mark history")
            st.table(student_rows[["year"] + SUBJECTS].set_index("year"))

            col_a, col_b = st.columns(2)
            with col_a:
                fig3 = comparison_bar(marks_df, student_id)
                st.pyplot(fig3)
            with col_b:
                fig4 = line_progress(marks_df, student_id)
                st.pyplot(fig4)

            # Failing alert
            fail_df = failing_students(marks_df, threshold=50.0)
            student_fails = fail_df[fail_df["student_id"] == student_id]
            if not student_fails.empty:
                subjects_below = ", ".join(student_fails["subject"].tolist())
                st.error(f"Warning: {student_id} has average below 50 in: {subjects_below}")

        with tab3:
            st.header("Predict 2026 marks")
            student_id = str(st.selectbox("Student for prediction", students, key="pred_student"))
            model_name = st.selectbox("Regression model", ["LinearRegression", "DecisionTree", "RandomForest"], key="reg_model")
            row_2025 = marks_df[(marks_df["student_id"] == student_id) & (marks_df["year"] == 2025)]
            if row_2025.empty:
                st.warning("No 2025 row found for this student.")
            else:
                features = {subject: float(row_2025.iloc[0][subject]) for subject in SUBJECTS}
                predictions = predict_2026(features, model_name)
                pred_df = DataFrame({"subject": list(predictions.keys()), "predicted_mark": list(predictions.values())})
                fig = bar_predictions_2026(pred_df, student_id)
                st.pyplot(fig)
                st.subheader("Predicted 2026 marks")
                st.table(pred_df.set_index("subject"))
                try:
                    eval_df = evaluate_all_models()
                    st.subheader("Regression model evaluation")
                    st.dataframe(eval_df)
                except Exception as exc:
                    st.warning(f"Could not load evaluation metrics: {exc}")

        with tab4:
            st.header("Export Report")
            st.write("Generate a full PDF report covering all students, statistics, charts, and model evaluation.")
            if st.button("Generate PDF Report"):
                with st.spinner("Building report…"):
                    try:
                        # Save charts first so they are embedded in the PDF
                        from visualisation.charts import _save
                        import matplotlib
                        matplotlib.use("Agg")
                        _save(bar_subject_averages(marks_df), "bar_subject_averages")
                        _save(heatmap_correlation(correlation_matrix(marks_df)), "heatmap_correlation")
                        pdf_path = generate_report()
                        st.success(f"Report saved to: {pdf_path}")
                        with open(pdf_path, "rb") as fh:
                            st.download_button(
                                label="Download PDF",
                                data=fh.read(),
                                file_name="student_performance_report.pdf",
                                mime="application/pdf",
                            )
                    except Exception as exc:
                        st.error(f"Report generation failed: {exc}")

    else:
        st.header("VLE pass/fail analytics")
        tab1, tab2, tab3 = st.tabs(["Dataset Overview", "At-Risk Predictor", "Classifier Evaluation"])

        with tab1:
            st.subheader("Dataset overview")
            if vle_df.empty:
                st.warning("VLE dataset is not available.")
            else:
                counts = vle_df["label"].value_counts().sort_index()
                passes = int(counts.get(0, 0))
                fails = int(counts.get(1, 0))
                st.metric("Total students", len(vle_df))
                st.metric("Pass count", passes)
                st.metric("Fail count", fails)
                st.metric("Fail rate", f"{fails / max(1, len(vle_df)):.1%}")
                fig_dist = bar_vle_class_distribution(vle_df)
                st.pyplot(fig_dist)
                fig_corr = heatmap_vle_correlation(vle_df)
                st.pyplot(fig_corr)

        with tab2:
            st.subheader("Predict student risk")
            model_name = st.selectbox("Classifier", ["LogisticRegression", "RandomForest"], key="clf_model")
            feature_row = _build_vle_feature_row()
            if st.button("Predict pass/fail"):
                try:
                    label, proba = predict_pass_fail(feature_row, model_name)
                    if label == 0:
                        st.success(f"Prediction: PASS (fail probability {proba:.1%})")
                    else:
                        st.error(f"Prediction: FAIL (fail probability {proba:.1%})")
                    rf_model = None
                    if model_name == "RandomForest":
                        import joblib
                        rf_model = joblib.load(ROOT / "saved_models" / "classifier_RandomForest.pkl")
                    else:
                        rf_model = None
                    if rf_model is not None and hasattr(rf_model, "feature_importances_"):
                        feature_names = list(feature_row.keys())
                        fig_fi = bar_feature_importance(rf_model.feature_importances_, feature_names)
                        st.pyplot(fig_fi)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

        with tab3:
            st.subheader("Classifier evaluation")
            try:
                eval_df = evaluate_all_classifiers()
                st.dataframe(eval_df)
                for model_name in ["LogisticRegression", "RandomForest"]:
                    model = None
                    try:
                        import joblib
                        model = joblib.load(ROOT / "saved_models" / f"classifier_{model_name}.pkl")
                    except Exception:
                        model = None
                    if model is not None:
                        train_df, test_df = load_vle_train_test()
                        X_test = test_df[get_vle_feature_columns(test_df)].fillna(0).values
                        scaler = joblib.load(ROOT / "saved_models" / "vle_scaler.pkl")
                        X_test_scaled = scaler.transform(X_test)
                        y_test = test_df["label"].to_numpy(dtype=int)
                        from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
                        y_pred = model.predict(X_test_scaled)
                        proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
                        cm = confusion_matrix(y_test, y_pred)
                        fig_cm = plot_confusion_matrix(cm, model_name)
                        st.pyplot(fig_cm)
                        if proba is not None:
                            fpr, tpr, _ = roc_curve(y_test, proba)
                            auc = float(roc_auc_score(y_test, proba))
                            fig_roc = plot_roc_curve(fpr, tpr, auc, model_name)
                            st.pyplot(fig_roc)
            except Exception as exc:
                st.warning(f"Evaluation metrics unavailable: {exc}")


if __name__ == "__main__":
    main()
