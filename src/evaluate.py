import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


TEST_FILE = "data/test.csv"
MODEL_FILE = "model.pkl"


def evaluate_model():

    test_data = pd.read_csv(TEST_FILE)

    X_test = test_data.drop("fraud", axis=1)
    y_test = test_data["fraud"]

    model = joblib.load(MODEL_FILE)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print("\nModel Evaluation")
    print("================")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report")
    print("====================")
    print(classification_report(y_test, predictions, zero_division=0))


if __name__ == "__main__":
    evaluate_model()