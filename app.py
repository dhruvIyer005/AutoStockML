
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load("model.joblib")


@app.get("/")
def home():
    return jsonify({
        "project": "AutoStockML",
        "status": "running",
        "endpoint": "/predict"
    })


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object"}), 400

    prices = data.get("prices")

    if not isinstance(prices, list) or len(prices) != 5:
        return jsonify({
            "error": "Provide exactly five closing prices"
        }), 400

    try:
        values = np.asarray(prices, dtype=float)
    except (ValueError, TypeError):
        return jsonify({"error": "Prices must be numeric"}), 400

    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        return jsonify({
            "error": "Prices must be finite positive numbers"
        }), 400

    prediction = model.predict(values.reshape(1, -1))[0]

    return jsonify({
        "predicted_next_close": round(float(prediction), 4),
        "model": "LinearRegression"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
