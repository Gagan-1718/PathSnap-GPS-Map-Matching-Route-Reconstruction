"""API tests for PathSnap. Run them against a running API (Laptop 1).

Start the API first:   deployment\\run_api.bat
Then run:              python -m pytest api/test_api.py -v
Other machine/port:    set PATHSNAP_API=http://192.168.43.120:8000   (then run pytest)
"""
import os

import pytest
import requests

API = os.environ.get("PATHSNAP_API", "http://localhost:8000").rstrip("/")


def get(path):
    return requests.get(API + path, timeout=30)


def post(path, body):
    return requests.post(API + path, json=body, timeout=120)


@pytest.fixture(scope="module", autouse=True)
def api_is_running():
    try:
        get("/health")
    except requests.exceptions.ConnectionError:
        pytest.skip(f"API not running at {API}. Start it with deployment\\run_api.bat")


def test_health_is_ok():
    r = get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["default_model"] in body["models"]


def test_trips_list_has_ten_trips():
    r = get("/trips")
    assert r.status_code == 200
    assert len(r.json()["trips"]) == 10


def test_predict_returns_route_and_metrics():
    trip = get("/trips").json()["trips"][0]
    r = post("/predict", {"trip_id": trip})
    assert r.status_code == 200
    body = r.json()
    assert body["trip_id"] == trip
    assert len(body["gps_points"]) > 0
    assert len(body["predicted_route"]) > 0
    assert 0.0 <= body["metrics"]["top1_accuracy"] <= 1.0
    assert 0.0 <= body["metrics"]["route_continuity"] <= 1.0


def test_reference_route_never_leaves_the_server():
    # Rule: ground truth must stay on Laptop 1. Only summary metrics may be returned.
    trip = get("/trips").json()["trips"][0]
    body = post("/predict", {"trip_id": trip}).json()
    forbidden = {"route", "reference_route", "ref_arc", "ref_arcs", "labels", "ground_truth"}
    assert forbidden.isdisjoint(body.keys())
    assert forbidden.isdisjoint(body["metrics"].keys())


def test_every_trip_works():
    # The final evaluation must work for all ten trips, not just one.
    for trip in get("/trips").json()["trips"]:
        r = post("/predict", {"trip_id": trip})
        assert r.status_code == 200, f"trip {trip} failed: {r.text[:200]}"


def test_unknown_trip_gives_404():
    assert post("/predict", {"trip_id": "does_not_exist"}).status_code == 404


def test_unknown_model_gives_400():
    trip = get("/trips").json()["trips"][0]
    assert post("/predict", {"trip_id": trip, "model": "no_such_model"}).status_code == 400
