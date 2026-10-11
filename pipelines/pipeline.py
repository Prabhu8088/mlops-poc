from kfp import dsl
from kfp.compiler import Compiler

@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=[
        "numpy",
        "pandas",
    ],
)
def generate_data(
    output_file: dsl.Output[dsl.Dataset],
):
    import numpy as np
    import pandas as pd

    np.random.seed(42)

    NUM_RECORDS = 10000

    data = pd.DataFrame({
        "transaction_amount": np.random.exponential(200, NUM_RECORDS),
        "transaction_hour": np.random.randint(0, 24, NUM_RECORDS),
        "customer_age": np.random.randint(18, 80, NUM_RECORDS),
        "account_age_days": np.random.randint(30, 3000, NUM_RECORDS),
        "transactions_last_24h": np.random.poisson(3, NUM_RECORDS),
        "foreign_transaction": np.random.randint(0, 2, NUM_RECORDS),
        "distance_from_home": np.random.exponential(20, NUM_RECORDS),
        "previous_fraud_count": np.random.poisson(0.2, NUM_RECORDS),
    })

    fraud_score = (
        (data["transaction_amount"] > 500) * 1.5
        + (data["transaction_hour"].isin([0, 1, 2, 3, 4])) * 1.0
        + (data["transactions_last_24h"] > 8) * 1.2
        + (data["foreign_transaction"] == 1) * 1.0
        + (data["distance_from_home"] > 50) * 1.3
        + (data["previous_fraud_count"] > 0) * 1.5
    )

    probability = 1 / (1 + np.exp(-(fraud_score - 2.5)))

    data["fraud"] = np.random.binomial(1, probability)

    data.to_csv(output_file.path, index=False)

    print(f"Generated {len(data)} records")
    print(f"Saved dataset to: {output_file.path}")

    print("\nClass distribution:")
    print(data["fraud"].value_counts())


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=[
        "pandas",
        "scikit-learn",
    ],
)
def preprocess_data(
    input_file: dsl.Input[dsl.Dataset],
    train_file: dsl.Output[dsl.Dataset],
    test_file: dsl.Output[dsl.Dataset],
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    data = pd.read_csv(input_file.path)

    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data["fraud"],
    )

    train_data.to_csv(train_file.path, index=False)
    test_data.to_csv(test_file.path, index=False)

    print(f"Training records: {len(train_data)}")
    print(f"Testing records: {len(test_data)}")


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "joblib",
        "mlflow",
    ],
)

def train_model(
    train_file: dsl.Input[dsl.Dataset],
    test_file: dsl.Input[dsl.Dataset],
    model_file: dsl.Output[dsl.Model],
):
    import os
    import tempfile
    import joblib
    import pandas as pd
    import mlflow
    import mlflow.sklearn

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    # MLflow server running on EC2
    mlflow.set_tracking_uri("http://172.31.15.216:5000")
    mlflow.set_experiment("fraud-detection")

    # Load training data
    train_data = pd.read_csv(train_file.path)
    X_train = train_data.drop("fraud", axis=1)
    y_train = train_data["fraud"]

    # Load test data
    test_data = pd.read_csv(test_file.path)
    X_test = test_data.drop("fraud", axis=1)
    y_test = test_data["fraud"]

    # Algorithms to compare
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    }

    results = []
    os.makedirs(os.path.dirname(model_file.path) or ".", exist_ok=True)


    # Train and log each algorithm as a separate MLflow run
    with tempfile.TemporaryDirectory() as temp_dir:
        for algorithm, model in models.items():
            with mlflow.start_run(run_name=algorithm):
                mlflow.log_param("algorithm", algorithm)
                mlflow.log_param("class_weight", "balanced")
                mlflow.log_param("random_state", 42)

                if algorithm == "random_forest":
                    mlflow.log_param("n_estimators", 100)
                    mlflow.log_param("max_depth", 10)
                else:
                    mlflow.log_param("max_iter", 1000)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(
                        y_test, y_pred, zero_division=0
                    ),
                    "recall": recall_score(
                        y_test, y_pred, zero_division=0
                    ),
                    "f1_score": f1_score(
                        y_test, y_pred, zero_division=0
                    ),
                }

                mlflow.log_metrics(metrics)

                # Log a native MLflow model for registry compatibility
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="mlflow-model",
                )

                run_id = mlflow.active_run().info.run_id

                results.append({
                    "algorithm": algorithm,
                    "f1_score": metrics["f1_score"],
                    "run_id": run_id,
                    "metrics": metrics,
                })

                print(f"\nAlgorithm: {algorithm}")
                print(f"MLflow Run ID: {run_id}")
                for metric_name, value in metrics.items():
                    print(f"{metric_name}: {value:.4f}")

        # Both algorithms have now completed. Select the winner.
        best_result = max(results, key=lambda result: result["f1_score"])

        # Register only the winning model
        model_uri = f"runs:/{best_result['run_id']}/mlflow-model"
        registered_model_name = "fraud-detection-model"

        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name,
        )

        print("\n========== Model Registry ==========")
        print(f"Registered model: {registered_model_name}")
        print(f"Model version: {registered_model.version}")
        print(f"Selected algorithm: {best_result['algorithm']}")
        print(f"Selected run ID: {best_result['run_id']}")

        # Preserve the existing Kubeflow output artifact
        joblib.dump(models[best_result["algorithm"]], model_file.path)

        print("\n========== Model Comparison ==========")
        for result in results:
            print(
                f"{result['algorithm']}: "
                f"F1={result['f1_score']:.4f}"
            )

        print(f"\nBest algorithm: {best_result['algorithm']}")
        print(f"Best F1 score: {best_result['f1_score']:.4f}")
        print(f"Selected model saved to: {model_file.path}")


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "joblib",
    ],
)
def evaluate_model(
    model_file: dsl.Input[dsl.Model],
    test_file: dsl.Input[dsl.Dataset],
):
    import pandas as pd
    import joblib

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    model = joblib.load(model_file.path)

    test_data = pd.read_csv(test_file.path)

    X_test = test_data.drop("fraud", axis=1)
    y_test = test_data["fraud"]

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nModel Evaluation")
    print("----------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


@dsl.pipeline(
    name="fraud-detection-pipeline",
    description="Fraud detection MLOps pipeline",
)
def fraud_detection_pipeline():

    generate_task = generate_data()

    preprocess_task = preprocess_data(
        input_file=generate_task.outputs["output_file"],
    )

    train_task = train_model(
        train_file=preprocess_task.outputs["train_file"],
        test_file=preprocess_task.outputs["test_file"],
    )

    evaluate_task = evaluate_model(
        model_file=train_task.outputs["model_file"],
        test_file=preprocess_task.outputs["test_file"],
    )


if __name__ == "__main__":

    Compiler().compile(
        pipeline_func=fraud_detection_pipeline,
        package_path="fraud_detection_pipeline.yaml",
    )

    print("Pipeline compiled successfully!")