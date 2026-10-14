
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)
X, y = [], []

for _ in range(5000):
    prices = [100.0]

    for _ in range(5):
        prices.append(prices[-1] * (1 + rng.normal(0, 0.01)))

    X.append(prices[:5])
    y.append(prices[5])

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.joblib")
print("Model trained and saved to model.joblib")
