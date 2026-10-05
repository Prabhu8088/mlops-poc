import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_FILE = "data/fraud_data.csv"

TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"


def preprocess_data():
    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} records")

    X = df.drop("fraud", axis=1)
    y = df["fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    train_data = X_train.copy()
    train_data["fraud"] = y_train

    test_data = X_test.copy()
    test_data["fraud"] = y_test

    train_data.to_csv(TRAIN_FILE, index=False)
    test_data.to_csv(TEST_FILE, index=False)

    print(f"Training data: {len(train_data)} records")
    print(f"Testing data: {len(test_data)} records")


if __name__ == "__main__":
    preprocess_data()