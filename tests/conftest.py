"""Point the app at a throwaway database before it is imported.

app.config reads DATABASE_URL at import time, so this has to happen before any
test module pulls in the app — which is exactly what conftest is loaded for.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///" + os.path.join(
    tempfile.mkdtemp(), "test.db"
).replace("\\", "/")
os.environ.setdefault("SECRET_KEY", "test-secret")

# The endpoint archives every capture it is asked to store, so a test run would
# otherwise bury the real measurements in captures/ under its own synthetic ones.
os.environ["HEARTBEAT_SAVE_CAPTURES"] = "0"

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def sent_codes(monkeypatch):
    """Capture verification emails instead of sending them: {email: code}."""
    from app import mailer

    codes = {}

    async def capture(to, name, code, minutes):
        codes[to] = code

    monkeypatch.setattr(mailer, "send_verification_code", capture)
    return codes


@pytest.fixture
def sign_up(sent_codes):
    """Register, verify and log in; returns auth headers. Safe to call twice
    for the same email."""

    def go(client, email, password="secret123"):
        client.post("/api/v1/auth/register", json={"email": email, "password": password, "full_name": email})
        if email in sent_codes:
            client.post("/api/v1/auth/verify-email", json={"email": email, "code": sent_codes.pop(email)})
        token = client.post("/api/v1/auth/login", json={"email": email, "password": password}).json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    return go
