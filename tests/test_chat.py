"""Chat endpoints and tools, with the model replaced by LiteLLM's mock."""

import asyncio
import json

import litellm
import pytest
from fastapi.testclient import TestClient

from app.api import chat_router
from app.chat_tools import run_tool
from app.database import async_session
from app.models import HeartRateRecord, User
from main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_model(monkeypatch):
    sent = []
    real = litellm.acompletion

    async def fake(**kwargs):
        sent.append(kwargs["messages"])
        return await real(**kwargs, mock_response="Chào bạn")

    monkeypatch.setattr(chat_router.litellm, "acompletion", fake)
    return sent


def events(text):
    out = []
    for frame in text.strip().split("\n\n"):
        event, data = None, None
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        out.append((event, data if data == "[DONE]" else json.loads(data)))
    return out


def test_conversation_is_saved_and_history_capped(client, mock_model, sign_up):
    auth = sign_up(client, "chat@example.com")

    first = events(client.post("/api/v1/chat/stream", json={"message": "câu 1"}, headers=auth).text)
    assert first[0][0] == "meta"
    cid = first[0][1]["conversation_id"]
    assert first[-1] == (None, "[DONE]")
    assert "".join(d["delta"] for e, d in first if e is None and d != "[DONE]") == "Chào bạn"

    for i in range(2, 6):
        client.post("/api/v1/chat/stream", json={"conversation_id": cid, "message": f"câu {i}"}, headers=auth)

    stored = client.get(f"/api/v1/chat/conversations/{cid}/messages", headers=auth).json()
    assert [m["role"] for m in stored] == ["user", "assistant"] * 5

    # System prompt + at most HISTORY_LIMIT earlier turns + the new message,
    # and the history never opens on an assistant turn.
    last = mock_model[-1]
    assert last[0]["role"] == "system"
    assert last[-1] == {"role": "user", "content": "câu 5"}
    assert len(last) - 2 <= chat_router.HISTORY_LIMIT
    assert last[1]["role"] == "user"

    convs = client.get("/api/v1/chat/conversations", headers=auth).json()
    assert convs[0]["id"] == cid and convs[0]["title"] == "câu 1"


def test_other_users_cannot_touch_a_conversation(client, mock_model, sign_up):
    owner = sign_up(client, "owner@example.com")
    other = sign_up(client, "other@example.com")
    cid = events(client.post("/api/v1/chat/stream", json={"message": "bí mật"}, headers=owner).text)[0][1]["conversation_id"]

    assert client.get(f"/api/v1/chat/conversations/{cid}/messages", headers=other).status_code == 404
    assert client.delete(f"/api/v1/chat/conversations/{cid}", headers=other).status_code == 404
    assert client.post("/api/v1/chat/stream", json={"conversation_id": cid, "message": "x"}, headers=other).status_code == 404

    assert client.delete(f"/api/v1/chat/conversations/{cid}", headers=owner).status_code == 204
    assert client.get("/api/v1/chat/conversations", headers=owner).json() == []


def test_heart_rate_tool_reads_only_the_callers_data(client, sign_up):
    sign_up(client, "hr-a@example.com")
    sign_up(client, "hr-b@example.com")

    async def go():
        async with async_session() as db:
            from sqlalchemy import select
            a = (await db.execute(select(User).where(User.email == "hr-a@example.com"))).scalar_one()
            b = (await db.execute(select(User).where(User.email == "hr-b@example.com"))).scalar_one()
            db.add_all([HeartRateRecord(user_id=a.id, bpm=bpm) for bpm in (60, 70, 80)])
            db.add(HeartRateRecord(user_id=b.id, bpm=150))
            await db.commit()
            return await run_tool("get_heart_rate", {"days": 7}, db, a)

    result = asyncio.run(go())
    assert result["count"] == 3
    assert (result["min_bpm"], result["max_bpm"], result["avg_bpm"]) == (60, 80, 70)


def test_rate_limit_survives_deleting_conversations(client, mock_model, sign_up, monkeypatch):
    monkeypatch.setattr(chat_router, "RATE_LIMITS", [(chat_router.timedelta(hours=1), 2, "giờ")])
    auth = sign_up(client, "limited@example.com")

    for _ in range(2):
        cid = events(client.post("/api/v1/chat/stream", json={"message": "hi"}, headers=auth).text)[0][1]["conversation_id"]
        client.delete(f"/api/v1/chat/conversations/{cid}", headers=auth)

    blocked = client.post("/api/v1/chat/stream", json={"message": "hi"}, headers=auth)
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) > 0
    # Refused before reaching the model.
    assert len(mock_model) == 2
