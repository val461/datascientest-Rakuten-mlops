import pytest
import requests
import logging

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"


def test_health():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["service"] == "inference-api"


def test_predict():
    payload = {
        "designation": "Folkmanis Puppets - Marionnette Et Theatre - Mini Turtle",
        "description": "Marionnette tortue miniature en tissu",
        "productid": 516376098,
        "imageid":1019294171
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data


def test_train():
    logger.info('May take 10mn.')
    response = requests.post(f"{BASE_URL}/train")
    assert response.status_code == 200
    assert response.json()["status"] == "success"

