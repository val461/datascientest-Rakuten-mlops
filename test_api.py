from dotenv import load_dotenv
load_dotenv()  # loads .env

import pytest
import httpx
import logging
import os
import asyncio

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
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
    response = httpx.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["service"] == "inference-api"
    assert "training_ongoing" in response.json()


def test_auth(correct_payload):
    # check api key is loaded from .env by pytest
    assert API_KEY is not None

    # try missing key
    response = httpx.post(f"{BASE_URL}/predict", json=correct_payload)
    assert response.status_code == 401
    response = httpx.post(f"{BASE_URL}/train")
    assert response.status_code == 401

    # try wrong key
    wrong_key="wrong_key"
    response = httpx.post(f"{BASE_URL}/predict", headers={"X-API-Key": wrong_key}, json=correct_payload)
    assert response.status_code == 403
    response = httpx.post(f"{BASE_URL}/train", headers={"X-API-Key": wrong_key})
    assert response.status_code == 403


def test_predict(correct_payload):
    # try correct request
    response = httpx.post(f"{BASE_URL}/predict", headers={"X-API-Key": API_KEY}, json=correct_payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data

    # try missing payload
    response = httpx.post(f"{BASE_URL}/predict", headers={"X-API-Key": API_KEY})
    assert response.status_code == 422

    # try wrong payload (type int instead of str; missing fields)
    wrong_payload = {"designation": 22}
    response = httpx.post(f"{BASE_URL}/predict", headers={"X-API-Key": API_KEY}, json=wrong_payload)
    assert response.status_code == 422

    # try wrong payload (empty value; missing fields)
    wrong_payload = {"designation": ""}
    response = httpx.post(f"{BASE_URL}/predict", headers={"X-API-Key": API_KEY}, json=wrong_payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data


# Commented because training is slow and also tested by `test_async_train` below
# def test_train():
#     logger.info('May take 10mn.')
#     response = httpx.post(f"{BASE_URL}/train", headers={"X-API-Key": API_KEY}, timeout=None)
#     logger.debug(f"{response.text=}")
#     assert response.status_code == 200
#     assert response.json()["status"] == "success"


@pytest.mark.asyncio  # tag for compatibility with module pytest-asyncio
async def test_async_requests(correct_payload):
    n=5
    # Make n prediction requests at nearly the same time. They are supposed to succeed.
    async with httpx.AsyncClient() as client:
        requests = [client.post(f"{BASE_URL}/predict", headers={"X-API-Key": API_KEY}, json=correct_payload) for _ in range(n)]
        responses = await asyncio.gather(*requests)
    assert len(responses) == n
    for response in responses:
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_async_train():
    logger.info('May take 10mn.')
    # Make 2 training requests at nearly the same time. Request 2 is supposed to be denied because a training lock was implemented.
    async with httpx.AsyncClient() as client:
        request1 = client.post(f"{BASE_URL}/train", headers={"X-API-Key": API_KEY}, timeout=None)
        request2 = client.post(f"{BASE_URL}/train", headers={"X-API-Key": API_KEY}, timeout=None)
        response1, response2 = await asyncio.gather(request1, request2)
    logger.debug(f"{response1.text=}\n{response2.text=}")
    assert response1.status_code == 200
    assert response1.json()["status"] == "success"
    assert response2.status_code == 409
