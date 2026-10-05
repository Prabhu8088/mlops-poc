import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


TRAIN_FILE = "data/train.csv"
MODEL_FILE = "model.pkl"


def train_model():

    train_data = pd.read_csv(TRAIN_FILE)

    X_train = train_data.drop("fraud", axis=1)
    y_train = train_data["fraud"]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILE)

    print("Model training completed")
    print(f"Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    train_model()