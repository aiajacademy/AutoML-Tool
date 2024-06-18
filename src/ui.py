import streamlit as st
from src.feature_engineering import FeatureEngineer
from src.model_selection import ModelSelector
from src.hyperparameter_tuning import HyperparameterTuner
from src.performance_evaluation import PerformanceEvaluator

def main():
    st.title("Automated Machine Learning (AutoML) Tool")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        target = st.selectbox("Select the target column", data.columns)

        if st.button("Run AutoML"):
            feature_engineer = FeatureEngineer()
            model_selector = ModelSelector()
            hyperparameter_tuner = HyperparameterTuner()
            performance_evaluator = PerformanceEvaluator()

            features = feature_engineer.extract_features(data.drop(columns=[target]))
            target_data = data[target]
            best_model = model_selector.select_model(features, target_data)
            tuned_model = hyperparameter_tuner.tune_model(best_model, features, target_data)
            performance = performance_evaluator.evaluate_model(tuned_model, features, target_data)

            st.subheader("Model Performance")
            st.text(performance['report'])
            st.text(performance['confusion_matrix'])

if __name__ == "__main__":
    main()
