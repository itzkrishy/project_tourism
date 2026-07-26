# for data manipulation
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer # Added this import

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# for model serialization
import joblib
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("project-tourism-training-experiment")

# Xtrain/Xtest/ytrain/ytest are downloaded from the previous job's artifact
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv")
ytest = pd.read_csv("ytest.csv")

numeric_features = ["CityTier", "NumberOfPersonVisiting", "Passport", "PitchSatisfactionScore", "OwnCar", "Age", "DurationOfPitch", "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "NumberOfChildrenVisiting", "MonthlyIncome"]
categorical_features = ["TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus", "Designation"]

# Set the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Define XGBoost Classifier (since ProdTaken is a binary target 0/1)
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid for classification
param_grid = {
    'xgbclassifier__n_estimators': [50, 100, 150],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__subsample': [0.6, 0.8, 1.0],
    'xgbclassifier__colsample_bytree': [0.6, 0.8, 1.0],
    'xgbclassifier__gamma': [0, 0.1, 0.2]
}

# Create pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="recall", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

# Best model
best_model = grid_search.best_estimator_
print("Best Params:")
print(grid_search.best_params_)

# Predictions
y_pred_train = best_model.predict(Xtrain)
y_proba_train = best_model.predict_proba(Xtrain)[:, 1]
y_pred_test = best_model.predict(Xtest)
y_proba_test = best_model.predict_proba(Xtest)[:, 1]

train_report = classification_report(ytrain, y_pred_train, output_dict=True)
test_report = classification_report(ytest, y_pred_test, output_dict=True)
print(classification_report(ytest, y_pred_test))

# Log the metrics for the best model
mlflow.log_metrics({
    "train_accuracy": train_report["accuracy"],
    "train_precision": train_report["1"]["precision"],
    "train_recall": train_report["1"]["recall"],
    "train_f1-score": train_report["1"]["f1-score"],
    "test_accuracy": test_report["accuracy"],
    "test_precision": test_report["1"]["precision"],
    "test_recall": test_report["1"]["recall"],
    "test_f1-score": test_report["1"]["f1-score"]
})

# Save best model
model_path = "project_tourism/deployment/best_tourism_prediction_model.joblib"
joblib.dump(best_model, model_path)
mlflow.log_artifact(model_path, artifact_path="model")
print(f"Best model saved to: {model_path}")
