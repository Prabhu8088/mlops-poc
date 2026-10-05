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

# Create a synthetic fraud probability.
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

output_file = "data/fraud_data.csv"

data.to_csv(output_file, index=False)

print(f"Generated {len(data)} records")
print(f"Saved dataset to: {output_file}")
print("\nClass distribution:")
print(data["fraud"].value_counts())