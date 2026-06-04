from dotenv import load_dotenv
load_dotenv()  # loads .env

import pytest
import requests
import logging
import os

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("API_KEY")


@pytest.fixture
def correct_payload():
    return {
        "designation": "Folkmanis Puppets - Marionnette Et Theatre - Mini Turtle",
        "description": "Marionnette tortue miniature en tissu",
        "productid": 516376098,
        "imageid":1019294171
    }


def test_health():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["service"] == "inference-api"


def test_auth(correct_payload):
    # check api key is loaded from .env by pytest
    assert API_KEY is not None

    # try missing key
    response = requests.post(f"{BASE_URL}/predict", json=correct_payload)
    assert response.status_code == 401
    response = requests.post(f"{BASE_URL}/train")
    assert response.status_code == 401

    # try wrong key
    wrong_key="wrong_key"
    response = requests.post(f"{BASE_URL}/predict", headers={"X-API-Key": wrong_key}, json=correct_payload)
    assert response.status_code == 403
    response = requests.post(f"{BASE_URL}/train", headers={"X-API-Key": wrong_key})
    assert response.status_code == 403


def test_predict(correct_payload):
    # try correct request
    response = requests.post(f"{BASE_URL}/predict", headers={"X-API-Key": API_KEY}, json=correct_payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data


def test_train():
    logger.info('May take 10mn.')
    response = requests.post(f"{BASE_URL}/train", headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    assert response.json()["status"] == "success"
