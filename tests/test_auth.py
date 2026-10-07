"""Sign-up email verification."""

import pytest
from fastapi.testclient import TestClient

from app.api import auth_router
from main import app

CREDS = {"password": "secret123", "full_name": "V"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def register(client, email):
    return client.post("/api/v1/auth/register", json={"email": email, **CREDS})


def login(client, email):
    return client.post("/api/v1/auth/login", json={"email": email, "password": CREDS["password"]})


def test_cannot_log_in_until_verified(client, sent_codes):
    assert register(client, "v1@example.com").status_code == 201
    code = sent_codes["v1@example.com"]

    refused = login(client, "v1@example.com")
    assert refused.status_code == 403
    assert refused.json()["detail"] == auth_router.NOT_VERIFIED

    wrong = "000000" if code != "000000" else "111111"
    assert client.post("/api/v1/auth/verify-email", json={"email": "v1@example.com", "code": wrong}).status_code == 400

    ok = client.post("/api/v1/auth/verify-email", json={"email": "v1@example.com", "code": code})
    assert ok.status_code == 200 and ok.json()["access_token"]
    assert login(client, "v1@example.com").status_code == 200
    # Verified now: the address can't be signed up again.
    assert register(client, "v1@example.com").status_code == 400


def test_code_dies_after_too_many_wrong_guesses(client, sent_codes):
    register(client, "v2@example.com")
    code = sent_codes["v2@example.com"]
    wrong = "000000" if code != "000000" else "111111"
    for _ in range(auth_router.CODE_MAX_ATTEMPTS):
        client.post("/api/v1/auth/verify-email", json={"email": "v2@example.com", "code": wrong})
    late = client.post("/api/v1/auth/verify-email", json={"email": "v2@example.com", "code": code})
    assert late.status_code == 400 and "hết hạn" in late.json()["detail"]


def test_resend_has_a_cooldown_and_replaces_the_code(client, sent_codes, monkeypatch):
    register(client, "v3@example.com")
    first = sent_codes["v3@example.com"]
    assert client.post("/api/v1/auth/resend-code", json={"email": "v3@example.com"}).status_code == 429

    monkeypatch.setattr(auth_router, "RESEND_COOLDOWN", auth_router.timedelta(0))
    assert client.post("/api/v1/auth/resend-code", json={"email": "v3@example.com"}).status_code == 202
    second = sent_codes["v3@example.com"]
    if second != first:
        assert client.post("/api/v1/auth/verify-email", json={"email": "v3@example.com", "code": first}).status_code == 400
    assert client.post("/api/v1/auth/verify-email", json={"email": "v3@example.com", "code": second}).status_code == 200


def test_resend_does_not_reveal_unknown_addresses(client):
    assert client.post("/api/v1/auth/resend-code", json={"email": "nobody@example.com"}).status_code == 202
