import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app import app
    return TestClient(app)


def test_status_endpoint(client):
    r = client.get("/status")
    assert r.status_code == 200
    assert r.json() == "I'm OK"


def test_version_endpoint(client):
    r = client.get("/version")
    assert r.status_code == 200
    data = r.json()
    assert "service" in data
    assert "version" in data
    assert "model_loaded_at" in data
    assert "shadow_enabled" in data


def test_predict_returns_valid_shape(client):
    payload = {
        "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
        "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
        "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
        "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
        "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
        "V26": 0.17, "V27": 0.06, "V28": 0.02, "Amount": 149.62, "Hour": 0,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["pred"] in (0, 1)
    assert 0.0 <= data["prob"] <= 1.0


def test_predict_rejects_missing_fields(client):
    r = client.post("/predict", json={"Amount": 100})
    assert r.status_code == 422