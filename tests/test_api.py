from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict(monkeypatch):
    from src.api import main

    expected = {
        "model": "test-model",
        "churn_probability": 0.6,
        "threshold": 0.35,
        "churn_prediction": 1,
        "risk_label": "High Risk (Churn)",
    }
    monkeypatch.setattr(main, "predict_churn", lambda _: expected)

    response = client.post("/predict", json={})

    assert response.status_code == 200
    assert response.json() == expected


def test_metrics():
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "http_requests_total" in response.text