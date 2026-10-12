import os

import requests
from flask import Flask, render_template, request

app = Flask(__name__)

KSERVE_URL = os.getenv(
    "KSERVE_URL",
    "http://fraud-predictor-predictor.churn-model.svc.cluster.local"
    "/v2/models/fraud-predictor/infer",
)

FEATURES = [
    "transaction_amount",
    "transaction_hour",
    "customer_age",
    "account_age_days",
    "transactions_last_24h",
    "foreign_transaction",
    "distance_from_home",
    "previous_fraud_count",
]

DEFAULTS = {
    "transaction_amount": "59.28",
    "transaction_hour": "7",
    "customer_age": "24",
    "account_age_days": "2009",
    "transactions_last_24h": "3",
    "foreign_transaction": "0",
    "distance_from_home": "0.90",
    "previous_fraud_count": "0",
}


@app.route("/", methods=["GET", "POST"])
def index():
    values = DEFAULTS.copy()
    prediction = None
    error = None

    if request.method == "POST":
        try:
            values = {key: request.form[key] for key in FEATURES}
            features = [float(values[key]) for key in FEATURES]

            if not all(map(lambda x: x == x and abs(x) != float("inf"), features)):
                raise ValueError("Please enter valid finite numbers.")

            if features[5] not in (0, 1):
                raise ValueError("foreign_transaction must be 0 or 1.")

            payload = {
                "inputs": [
                    {
                        "name": "input-0",
                        "shape": [1, len(FEATURES)],
                        "datatype": "FP64",
                        "data": features,
                    }
                ]
            }

            response = requests.post(
                KSERVE_URL,
                json=payload,
                timeout=15,
            )
            response.raise_for_status()
            result = response.json()

            outputs = result.get("outputs", [])
            if not outputs or not outputs[0].get("data"):
                raise ValueError(
                    "KServe returned no prediction. Check the model response format."
                )

            prediction = outputs[0]["data"][0]

        except (ValueError, KeyError) as exc:
            error = str(exc)
        except requests.RequestException as exc:
            app.logger.exception("KServe inference request failed")
            error = (
                "Could not reach the model inference service. "
                "Check KServe connectivity and application logs."
            )

    return render_template(
        "index.html",
        features=FEATURES,
        values=values,
        prediction=prediction,
        error=error,
    )


@app.get("/health")
def health():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)