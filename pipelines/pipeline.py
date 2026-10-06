from kfp import dsl
from kfp.compiler import Compiler


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "joblib",
    ],
)
def generate_data(
    output_file: dsl.OutputPath(str),
):
    import pandas as pd
    import random

    random.seed(42)

    data = []

    for _ in range(1000):
        amount = random.uniform(10, 10000)
        age = random.randint(18, 80)
        transaction_count = random.randint(1, 20)

        # Simple synthetic fraud rule
        fraud = 1 if amount > 7000 and transaction_count > 10 else 0

        data.append({
            "amount": amount,
            "age": age,
            "transaction_count": transaction_count,
            "fraud": fraud,
        })

    df = pd.DataFrame(data)

    df.to_csv(output_file, index=False)

    print(f"Generated {len(df)} records")
    print(f"Data saved to: {output_file}")


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=[
        "pandas",
        "scikit-learn",
    ],
)
def preprocess_data(
    input_file: dsl.InputPath(str),
    train_file: dsl.OutputPath(str),
    test_file: dsl.OutputPath(str),
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    data = pd.read_csv(input_file)

    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data["fraud"],
    )

    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

    print(f"Training records: {len(train_data)}")
    print(f"Testing records: {len(test_data)}")


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "joblib",
    ],
)
def train_model(
    train_file: dsl.InputPath(str),
    model_file: dsl.OutputPath(str),
):
    import pandas as pd
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    train_data = pd.read_csv(train_file)

    X_train = train_data.drop("fraud", axis=1)
    y_train = train_data["fraud"]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    joblib.dump(model, model_file)

    print("Model training completed")
    print(f"Model saved to: {model_file}")


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "joblib",
    ],
)
def evaluate_model(
    model_file: dsl.InputPath(str),
    test_file: dsl.InputPath(str),
):
    import pandas as pd
    import joblib

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    model = joblib.load(model_file)

    test_data = pd.read_csv(test_file)

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