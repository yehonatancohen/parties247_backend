from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import app


class FakeAnalyticsCollection:
    def __init__(self):
        self.docs: list[dict] = []

    def insert_one(self, doc):
        self.docs.append(doc)
        return SimpleNamespace(inserted_id="id")

    def find(self, *args, **kwargs):
        return list(self.docs)


def make_request(payload, headers=None):
    headers = headers or {}
    req = SimpleNamespace(headers=headers)
    req.get_json = lambda silent=True: payload
    req.remote_addr = headers.get("Remote-Addr")
    return req


def test_create_analytics_event_success(monkeypatch):
    collection = FakeAnalyticsCollection()
    headers = {
        "User-Agent": "pytest-agent",
        "Referer": "https://example.com/page",
        "X-Forwarded-For": "203.0.113.5",
    }
    payload = {
        "category": "page",
        "action": "view",
        "label": "Home",
        "path": "/",
        "context": {"foo": "bar", "skip": None},
        "sessionId": "abc123",
    }

    monkeypatch.setattr(app, "analytics_collection", collection)
    monkeypatch.setattr(app, "request", make_request(payload, headers))

    response, status = app.create_analytics_event()

    assert status == 201
    assert response["message"] == "Recorded"
    assert len(collection.docs) == 1
    doc = collection.docs[0]
    assert doc["category"] == "page"
    assert doc["action"] == "view"
    assert doc["label"] == "Home"
    assert doc["path"] == "/"
    assert doc["context"] == {"foo": "bar"}
    assert doc["sessionId"] == "abc123"
    assert doc["clientIp"] == "203.0.113.5"
    assert isinstance(doc["createdAt"], datetime)


def test_create_analytics_event_validation(monkeypatch):
    collection = FakeAnalyticsCollection()
    monkeypatch.setattr(app, "analytics_collection", collection)
    monkeypatch.setattr(app, "request", make_request({"category": "page"}))

    response, status = app.create_analytics_event()

    assert status == 400
    assert response["message"] == "Invalid analytics event."
    assert collection.docs == []


def test_analytics_summary(monkeypatch):
    now = datetime.now(timezone.utc)
    collection = FakeAnalyticsCollection()
    collection.docs.extend(
        [
            {
                "category": "page",
                "action": "view",
                "label": "Home",
                "path": "/",
                "createdAt": now,
            },
            {
                "category": "party",
                "action": "enter",
                "label": "After Hours",
                "path": "/event/after-hours",
                "context": {"partyId": "party-123", "partySlug": "after-hours"},
                "createdAt": now - timedelta(days=5),
            },
            {
                "category": "page",
                "action": "view",
                "label": "Home",
                "path": "/",
                "createdAt": now - timedelta(days=40),
            },
        ]
    )

    monkeypatch.setattr(app, "analytics_collection", collection)

    response, status = app.analytics_summary()

    assert status == 200
    assert response["totalEvents"] == 3
    assert response["recentEvents"] == 2
    assert any(item["category"] == "page" and item["action"] == "view" for item in response["actions"])
    assert response["topLabels"][0]["label"] == "Home"
    assert response["topPaths"][0]["path"] == "/"
    top_party = response["topPartyEntries"][0]
    assert top_party["label"] == "After Hours"
    assert top_party["partyId"] == "party-123"
