
from app import app


def test_home():
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert response.json["status"] == "running"


def test_health():
    response = app.test_client().get("/health")
    assert response.status_code == 200


def test_prediction():
    response = app.test_client().post(
        "/predict",
        json={"prices": [100, 101, 102, 101.5, 103]}
    )
    assert response.status_code == 200
    assert "predicted_next_close" in response.json


def test_invalid_prices():
    response = app.test_client().post(
        "/predict",
        json={"prices": [100, 101]}
    )
    assert response.status_code == 400
