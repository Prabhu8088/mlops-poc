import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"
MODEL_FILE = "model.pkl"


def train_model(algorithm="random_forest"):

    # Load training data
    train_data = pd.read_csv(TRAIN_FILE)

    X_train = train_data.drop("fraud", axis=1)
    y_train = train_data["fraud"]

    # Create ML model
    if algorithm == "random_forest":
        model = RandomForestClassifier(
          n_estimators=100,
          max_depth=10,
          random_state=42,
          class_weight="balanced",
        )
    
    elif algorithm == "logistic_regression":
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
    
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # MLflow experiment
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run():

         # Train the model
        model.fit(X_train, y_train)

         # Log algorithm
        mlflow.log_param("algorithm", algorithm)

        # Log model-specific parameters
        if algorithm == "random_forest":
            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("max_depth", 10)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("class_weight", "balanced")
        
        elif algorithm == "logistic_regression":
            mlflow.log_param("max_iter", 1000)
            mlflow.log_param("class_weight", "balanced")
            mlflow.log_param("random_state", 42)

        # Load test data
        test_data = pd.read_csv(TEST_FILE)

        # Separate features and target
        X_test = test_data.drop("fraud", axis=1)
        y_test = test_data["fraud"]

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
    
        # Save model
        joblib.dump(model, MODEL_FILE)

        # Log model to MLflow
        if algorithm == "random_forest":
            mlflow.sklearn.log_model(
              model,
              name="model",
              skops_trusted_types=["sklearn.tree._tree.Tree"]
            )
        else:
            mlflow.sklearn.log_model(
                model,
                name="model"
            )
        
        # Print results
        print("\nModel Evaluation")
        print("----------------")
        print(f"Algorithm : {algorithm}")
        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")

    print("Model training completed")
    print(f"Model saved to: {MODEL_FILE}")
    print("Model logged to MLflow")


if __name__ == "__main__":
    train_model("logistic_regression")