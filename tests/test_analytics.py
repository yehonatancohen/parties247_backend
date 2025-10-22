from datetime import datetime, timedelta, timezone
from collections import Counter
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


def test_party_click_endpoint_records_event(monkeypatch):
    collection = FakeAnalyticsCollection()

    request_obj = SimpleNamespace(
        headers={"User-Agent": "pytest-agent"},
        path="/api/analytics/party-click",
        remote_addr="198.51.100.42",
    )
    request_obj.get_json = lambda silent=True: {
        "partyId": "p1",
        "carouselId": "c1",
        "carouselTitle": "Featured",
        "source": "homepage",
        "page": "/",
        "context": {"widget": "hero"},
    }

    monkeypatch.setattr(app, "analytics_collection", collection)
    monkeypatch.setattr(app, "request", request_obj)

    response, status = app.create_party_click_event()

    assert status == 201
    assert response["message"] == "Recorded"
    assert len(collection.docs) == 1
    doc = collection.docs[0]
    assert doc["category"] == "party"
    assert doc["action"] == "click"
    assert doc["label"] == "p1"
    assert doc["path"] == "/"
    assert doc["context"]["carouselId"] == "c1"
    assert doc["context"]["widget"] == "hero"


def test_party_click_endpoint_requires_identifier(monkeypatch):
    collection = FakeAnalyticsCollection()

    request_obj = SimpleNamespace(
        headers={},
        path="/api/analytics/party-click",
        remote_addr="198.51.100.43",
    )
    request_obj.get_json = lambda silent=True: {"context": {"ignored": "value"}}

    monkeypatch.setattr(app, "analytics_collection", collection)
    monkeypatch.setattr(app, "request", request_obj)

    response, status = app.create_party_click_event()

    assert status == 400
    assert response["message"] == "Invalid party click payload."
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


def test_get_carousels_records_analytics(monkeypatch):
    collection = FakeAnalyticsCollection()

    class DummyCursor:
        def __init__(self, items):
            self._items = list(items)

        def sort(self, key, direction):
            return list(self._items)

    class DummyCarousels:
        def find(self):
            return DummyCursor([
                {"_id": "1", "title": "Featured", "partyIds": ["p1", "p2"]},
                {"_id": "2", "title": "Fresh", "partyIds": []},
            ])

    request_obj = SimpleNamespace(
        headers={"User-Agent": "pytest-agent"},
        path="/api/carousels",
        args={"page": "/"},
        remote_addr="198.51.100.10",
    )

    monkeypatch.setattr(app, "analytics_collection", collection)
    monkeypatch.setattr(app, "carousels_collection", DummyCarousels())
    monkeypatch.setattr(app, "request", request_obj)

    payload, status = app.get_carousels()
    assert status == 200
    assert len(collection.docs) == 5
    categories = Counter(doc["category"] for doc in collection.docs)
    assert categories["website"] == 1
    assert categories["carousel"] == 2
    assert categories["party"] == 2
    party_docs = [doc for doc in collection.docs if doc["category"] == "party"]
    assert all(doc["action"] == "impression" for doc in party_docs)
    assert any(doc.get("context", {}).get("carouselId") == "1" for doc in party_docs)


def test_get_carousels_skips_admin_session(monkeypatch):
    collection = FakeAnalyticsCollection()

    class DummyCursor:
        def __init__(self, items):
            self._items = list(items)

        def sort(self, key, direction):
            return list(self._items)

    class DummyCarousels:
        def find(self):
            return DummyCursor([{"_id": "1", "title": "Featured", "partyIds": ["p1"]}])

    request_obj = SimpleNamespace(
        headers={"Authorization": "Bearer token"},
        path="/api/carousels",
        args={},
        remote_addr="198.51.100.20",
    )

    monkeypatch.setattr(app, "analytics_collection", collection)
    monkeypatch.setattr(app, "carousels_collection", DummyCarousels())
    monkeypatch.setattr(app, "request", request_obj)
    monkeypatch.setattr(app, "JWT_SECRET", "secret")

    payload, status = app.get_carousels()
    assert status == 200
    assert collection.docs == []


def test_event_detail_records_analytics(monkeypatch):
    collection = FakeAnalyticsCollection()
    request_obj = SimpleNamespace(
        headers={"User-Agent": "pytest-agent"},
        path="/api/events/demo",
        args={},
        remote_addr="203.0.113.9",
    )

    def fake_find_event(slug):
        return ({"slug": slug, "id": "event-1", "name": "Demo"}, {"originalUrl": "https://example.com/event"})

    monkeypatch.setattr(app, "analytics_collection", collection)
    monkeypatch.setattr(app, "request", request_obj)
    monkeypatch.setattr(app, "find_event_by_slug", fake_find_event)
    monkeypatch.setattr(app, "default_referral_code", lambda: None)
    monkeypatch.setattr(app, "apply_default_referral", lambda payload, referral: None)

    response, status, headers = app.event_detail_api("demo")

    assert status == 200
    assert any(doc["category"] == "party" and doc["action"] == "view" for doc in collection.docs)
    party_doc = next(doc for doc in collection.docs if doc["category"] == "party")
    assert party_doc["label"] == "demo"
    assert party_doc.get("context", {}).get("slug") == "demo"
