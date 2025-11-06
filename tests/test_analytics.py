from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import app


class FakeVisitorCollection:
    def __init__(self):
        self.docs: list[dict] = []

    def update_one(self, filter, update, upsert=False):
        session_id = filter.get("sessionId")
        existing = next((doc for doc in self.docs if doc.get("sessionId") == session_id), None)
        payload = update.get("$set", {}).copy()
        if existing:
            existing.update(payload)
        elif upsert:
            self.docs.append(payload)
        return SimpleNamespace(upserted_id=None)

    def find(self, *args, **kwargs):
        return list(self.docs)


class FakePartyCollection:
    def __init__(self, docs: list[dict]):
        self.docs = docs

    def find_one(self, query):
        for doc in self.docs:
            match = True
            for key, value in query.items():
                if doc.get(key) != value:
                    match = False
                    break
            if match:
                return doc.copy()
        return None

    def find(self, *args, **kwargs):
        return list(self.docs)


class FakePartyAnalyticsCollection:
    def __init__(self):
        self.docs: list[dict] = []

    def update_one(self, filter, update, upsert=False):
        party_id = filter.get("partyId")
        existing = next((doc for doc in self.docs if doc.get("partyId") == party_id), None)
        if not existing:
            existing = {"partyId": party_id, "views": 0, "redirects": 0}
            self.docs.append(existing)
        inc = update.get("$inc", {})
        for key, value in inc.items():
            existing[key] = existing.get(key, 0) + value
        for key, value in update.get("$set", {}).items():
            existing[key] = value
        return SimpleNamespace(upserted_id=None)

    def find(self, *args, **kwargs):
        return list(self.docs)

    def delete_one(self, filter):
        party_id = filter.get("partyId")
        before = len(self.docs)
        self.docs = [doc for doc in self.docs if doc.get("partyId") != party_id]
        return SimpleNamespace(deleted_count=before - len(self.docs))


def make_request(payload, headers=None):
    headers = headers or {}
    req = SimpleNamespace(headers=headers)
    req.get_json = lambda silent=True: payload
    req.remote_addr = headers.get("Remote-Addr")
    return req


def test_record_unique_visitor(monkeypatch):
    visitors = FakeVisitorCollection()
    headers = {
        "User-Agent": "pytest-agent",
        "Referer": "https://example.com",
        "X-Forwarded-For": "198.51.100.5",
    }
    payload = {"sessionId": "abc123"}

    monkeypatch.setattr(app, "visitor_analytics_collection", visitors)
    monkeypatch.setattr(app, "request", make_request(payload, headers))

    response, status = app.record_unique_visitor()

    assert status == 202
    assert response["message"] == "Recorded"
    assert len(visitors.docs) == 1
    stored = visitors.docs[0]
    assert stored["sessionId"] == "abc123"
    assert stored["referer"] == "https://example.com"
    assert stored["clientIp"] == "198.51.100.5"


def test_record_party_metrics(monkeypatch):
    now = datetime.now(timezone.utc)
    party_doc = {"_id": "party-1", "name": "After Hours", "slug": "after-hours", "date": (now + timedelta(days=1)).isoformat()}
    parties = FakePartyCollection([party_doc])
    metrics = FakePartyAnalyticsCollection()
    payload = {"partyId": "party-1"}

    monkeypatch.setattr(app, "parties_collection", parties)
    monkeypatch.setattr(app, "party_analytics_collection", metrics)
    monkeypatch.setattr(app, "request", make_request(payload))

    response, status = app.record_party_view()
    assert status == 202
    assert response["message"] == "Recorded"

    response, status = app.record_party_redirect()
    assert status == 202
    doc = metrics.docs[0]
    assert doc["views"] == 1
    assert doc["redirects"] == 1
    assert doc["partySlug"] == "after-hours"


def test_analytics_summary(monkeypatch):
    now = datetime.now(timezone.utc)
    live_party = {"_id": "party-1", "name": "After Hours", "slug": "after-hours", "date": (now + timedelta(hours=12)).isoformat()}
    archived_party = {"_id": "party-old", "name": "Old Party", "slug": "old", "date": (now - timedelta(days=2)).isoformat()}
    parties = FakePartyCollection([live_party, archived_party])

    metrics = FakePartyAnalyticsCollection()
    metrics.docs = [
        {"partyId": "party-1", "views": 5, "redirects": 2},
        {"partyId": "party-old", "views": 10, "redirects": 4},
    ]

    visitors = FakeVisitorCollection()
    visitors.docs = [
        {"sessionId": "new", "createdAt": now},
        {"sessionId": "old", "createdAt": now - timedelta(days=2)},
    ]

    monkeypatch.setattr(app, "parties_collection", parties)
    monkeypatch.setattr(app, "party_analytics_collection", metrics)
    monkeypatch.setattr(app, "visitor_analytics_collection", visitors)

    response, status = app.analytics_summary()

    assert status == 200
    assert response["uniqueVisitors24h"] == 1
    assert len(response["parties"]) == 1
    entry = response["parties"][0]
    assert entry["partyId"] == "party-1"
    assert entry["views"] == 5
    assert entry["redirects"] == 2
