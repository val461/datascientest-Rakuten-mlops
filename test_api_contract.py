import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def reset_training_state():
    main.training_ongoing = False
    with main.training_lock:
        main.reset_campaign_state()
    yield
    main.training_ongoing = False
    with main.training_lock:
        main.reset_campaign_state()


@pytest.fixture
def api_headers(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "test-api-key")
    return {"X-API-Key": "test-api-key"}


def fake_simulation_result(step: int, deployed: bool) -> dict:
    return {
        "simulation_step": step,
        "available_ratio": 0.5 + step * 0.05,
        "train_rows": 100 + step,
        "accuracy": 0.8 + step * 0.001,
        "f1_macro": 0.79 + step * 0.001,
        "f1_weighted": 0.8 + step * 0.001,
        "mlflow_run_id": f"run-{step}",
        "deployed": deployed,
    }


def test_invalid_training_steps_are_rejected(api_headers):
    response = client.post("/train/simulation/-1", headers=api_headers)
    assert response.status_code == 422

    response = client.post("/train/simulation/11", headers=api_headers)
    assert response.status_code == 422


def test_full_training_uses_classic_pipeline(monkeypatch, api_headers):
    monkeypatch.setattr(
        main,
        "train_and_save_model",
        lambda: {"model_path": "models/model.joblib", "deployed": True},
    )
    reload_calls = []
    monkeypatch.setattr(
        main,
        "load_model",
        lambda **kwargs: reload_calls.append(kwargs),
    )

    response = client.post("/train", headers=api_headers)

    assert response.status_code == 200
    assert response.json()["deployed"] is True
    assert reload_calls == [{"force_reload": True}]


def test_single_step_does_not_deploy_by_default(monkeypatch, api_headers):
    calls = []

    def fake_train(step: int, deploy: bool):
        calls.append((step, deploy))
        return fake_simulation_result(step, deploy)

    monkeypatch.setattr(main, "train_and_save_simulation_model", fake_train)

    response = client.post("/train/simulation/4", headers=api_headers)

    assert response.status_code == 200
    assert response.json()["simulation_step"] == 4
    assert response.json()["deployed"] is False
    assert calls == [(4, False)]


def test_single_step_can_be_deployed(monkeypatch, api_headers):
    monkeypatch.setattr(
        main,
        "train_and_save_simulation_model",
        lambda step, deploy: fake_simulation_result(step, deploy),
    )
    reload_calls = []
    monkeypatch.setattr(
        main,
        "load_model",
        lambda **kwargs: reload_calls.append(kwargs),
    )

    response = client.post(
        "/train/simulation/3?deploy=true",
        headers=api_headers,
    )

    assert response.status_code == 200
    assert response.json()["deployed"] is True
    assert reload_calls == [{"force_reload": True}]


def test_complete_campaign_runs_all_steps_and_deploys_only_last(
    monkeypatch,
    api_headers,
):
    calls = []

    def fake_train(step: int, deploy: bool):
        calls.append((step, deploy))
        return fake_simulation_result(step, deploy)

    monkeypatch.setattr(main, "train_and_save_simulation_model", fake_train)
    monkeypatch.setattr(main, "load_model", lambda **kwargs: None)

    response = client.post("/train/simulation", headers=api_headers)
    campaign = client.get(
        "/train/simulation/status",
        headers=api_headers,
    )

    assert response.status_code == 202
    assert calls == [(step, step == 10) for step in range(11)]
    assert campaign.status_code == 200
    assert campaign.json()["status"] == "completed"
    assert campaign.json()["completed_steps"] == 11
    assert campaign.json()["results"][-1]["deployed"] is True
    assert all(
        result["deployed"] is False
        for result in campaign.json()["results"][:-1]
    )


def test_campaign_failure_is_reported_and_releases_lock(monkeypatch):
    calls = []

    def fake_train(step: int, deploy: bool):
        calls.append((step, deploy))
        if step == 3:
            raise RuntimeError("step 3 failed")
        return fake_simulation_result(step, deploy)

    monkeypatch.setattr(main, "train_and_save_simulation_model", fake_train)
    main.training_ongoing = True
    with main.training_lock:
        main.reset_campaign_state()
        main.campaign_state["status"] = "running"

    main.run_simulation_campaign()
    state = main.campaign_snapshot()

    assert calls == [(0, False), (1, False), (2, False), (3, False)]
    assert state["status"] == "failed"
    assert state["completed_steps"] == 3
    assert state["current_step"] == 3
    assert state["error"] == "step 3 failed"
    assert state["finished_at"] is not None
    assert main.training_ongoing is False


def test_second_training_request_is_rejected(monkeypatch, api_headers):
    monkeypatch.setattr(
        main,
        "train_and_save_simulation_model",
        lambda step, deploy: fake_simulation_result(step, deploy),
    )
    main.training_ongoing = True

    response = client.post("/train/simulation/2", headers=api_headers)

    assert response.status_code == 409
    assert response.json()["detail"] == "Un entrainement est deja en cours"


def test_data_status_before_split(monkeypatch, api_headers):
    monkeypatch.setattr(main, "load_split_metadata", lambda: None)
    monkeypatch.setattr(main, "is_model_available", lambda: False)

    response = client.get("/data-status", headers=api_headers)

    assert response.status_code == 200
    assert response.json() == {
        "split_initialized": False,
        "model_available": False,
    }
