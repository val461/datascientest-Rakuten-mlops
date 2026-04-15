import pytest
import requests

BASE_URL = "http://localhost:8000"


def test_health():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    # assert data["class_name"] in ["setosa", "versicolor", "virginica"]


def test_train():
    response = requests.post(f"{BASE_URL}/train")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
