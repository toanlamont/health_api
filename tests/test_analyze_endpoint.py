"""End-to-end checks for the analyse endpoint, over the real app and database."""

import pytest
from fastapi.testclient import TestClient

from main import app
from tests.test_ppg import synth


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth(client, sign_up):
    return sign_up(client, "ppg@example.com")


def payload(times, regions, source="finger", **extra):
    return {
        "source": source,
        "times": [float(t) for t in times],
        "region_count": regions.shape[1],
        "values": [float(v) for v in regions.reshape(-1)],
        **extra,
    }


def test_reads_a_real_pulse(client, auth):
    times, regions = synth(bpm=72)
    response = client.post(
        "/api/v1/heart-rate/analyze", json=payload(times, regions), headers=auth
    )
    assert response.status_code == 200
    body = response.json()
    assert body["bpm"] == pytest.approx(72, abs=3)
    assert body["snr_db"] > 0
    assert body["method"] in {"red", "green", "pos", "chrom"}
    assert body["fps"] == pytest.approx(30, abs=1)


def test_reports_a_rate_even_for_a_weak_capture(client, auth):
    # The endpoint no longer refuses. Whether a capture is worth believing is
    # decided by comparing snr_db against a reference heart rate, not by a
    # threshold picked from the synthetic signals in this repository.
    times, regions = synth(bpm=72)
    pulse = client.post(
        "/api/v1/heart-rate/analyze", json=payload(times, regions), headers=auth
    ).json()

    times, regions = synth(strength=0.0, noise=3.0)
    noise = client.post(
        "/api/v1/heart-rate/analyze", json=payload(times, regions), headers=auth
    ).json()

    assert noise["bpm"] is not None
    assert noise["snr_db"] < pulse["snr_db"] - 6.0


def test_stores_the_reading_when_asked(client, auth):
    before = len(client.get("/api/v1/heart-rate", headers=auth).json())

    times, regions = synth(bpm=88)
    body = client.post(
        "/api/v1/heart-rate/analyze",
        json=payload(times, regions, save=True),
        headers=auth,
    ).json()
    assert body["bpm"] is not None

    after = client.get("/api/v1/heart-rate", headers=auth).json()
    assert len(after) == before + 1
    assert any(record["bpm"] == body["bpm"] for record in after)


def test_does_not_store_when_told_not_to(client, auth):
    before = len(client.get("/api/v1/heart-rate", headers=auth).json())
    times, regions = synth(bpm=95)
    client.post(
        "/api/v1/heart-rate/analyze",
        json=payload(times, regions, save=False),
        headers=auth,
    )
    after = len(client.get("/api/v1/heart-rate", headers=auth).json())
    assert after == before


def test_a_capture_too_short_to_analyse_is_never_stored(client, auth):
    before = len(client.get("/api/v1/heart-rate", headers=auth).json())
    times, regions = synth(seconds=2.0)
    body = client.post(
        "/api/v1/heart-rate/analyze",
        json=payload(times, regions, save=True),
        headers=auth,
    ).json()
    assert body["bpm"] is None
    after = len(client.get("/api/v1/heart-rate", headers=auth).json())
    assert after == before


def test_handles_a_multi_region_face_capture(client, auth):
    times, regions = synth(bpm=75, regions=9, noise=0.3)
    body = client.post(
        "/api/v1/heart-rate/analyze",
        json=payload(times, regions, source="face"),
        headers=auth,
    ).json()
    assert body["bpm"] == pytest.approx(75, abs=4)
    assert 0 <= body["region"] < 9


def test_rejects_a_mismatched_payload(client, auth):
    times, regions = synth(bpm=72)
    bad = payload(times, regions)
    bad["values"] = bad["values"][:-3]
    response = client.post("/api/v1/heart-rate/analyze", json=bad, headers=auth)
    assert response.status_code == 422


def test_requires_authentication(client):
    times, regions = synth(bpm=72)
    response = client.post("/api/v1/heart-rate/analyze", json=payload(times, regions))
    assert response.status_code in (401, 403)


def test_payload_size_is_practical_for_a_face_capture():
    # 15s at 30fps over a 5x5 grid: this is what replaces uploading video.
    times, regions = synth(seconds=15, regions=25)
    body = payload(times, regions, source="face")
    import json

    megabytes = len(json.dumps(body).encode()) / 1e6
    assert megabytes < 2.0, f"payload is {megabytes:.1f} MB"
