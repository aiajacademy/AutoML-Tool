# Usage Guide

## Basic Usage

1. Import the AutoML tool:
   ```python
   from src.feature_engineering import FeatureEngineer
   from src.model_selection import ModelSelector
   from src.hyperparameter_tuning import HyperparameterTuner
   from src.performance_evaluation import PerformanceEvaluator

   # Initialize components
   feature_engineer = FeatureEngineer()
   model_selector = ModelSelector()
   hyperparameter_tuner = HyperparameterTuner()
   performance_evaluator = PerformanceEvaluator()

   # Use the components
   features = feature_engineer.extract_features(data)
   best_model = model_selector.select_model(features, target)
   tuned_model = hyperparameter_tuner.tune_model(best_model, features, target)
   performance = performance_evaluator.evaluate_model(tuned_model, features, target)
   print(performance)
