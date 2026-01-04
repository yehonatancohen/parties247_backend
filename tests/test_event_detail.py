import types

import app


class _DummyCollection:
    def __init__(self, items):
        self._items = list(items)

    def find(self):
        return list(self._items)


class _DummyAnalytics:
    def __init__(self):
        self.last_update = None

    def update_one(self, filter, update, upsert=False):
        self.last_update = {"filter": filter, "update": update, "upsert": upsert}

    def delete_one(self, filter):  # pragma: no cover - not used in tests
        self.last_update = {"deleted": filter}


def test_event_detail_api_returns_purchase_url(monkeypatch):
    docs = [
        {
            "_id": "1",
            "slug": "should-be-ignored",
            "name": "Thursday Moon 02/10",
            "date": "2099-02-10T20:00:00",
            "goOutUrl": "https://tickets.example.com/buy",
            "originalUrl": "https://tickets.example.com/info",
        }
    ]

    monkeypatch.setattr(app, "parties_collection", _DummyCollection(docs))

    analytics = _DummyAnalytics()
    monkeypatch.setattr(app, "party_analytics_collection", analytics)

    class _DummySettings:
        def find_one(self, filter):
            return {"value": "buy-now"}

    monkeypatch.setattr(app, "settings_collection", _DummySettings())

    payload, status, headers = app.event_detail_api("thursday-moon-02-10")

    assert status == 200
    event = payload["event"]
    assert event["slug"] == "thursday-moon-02-10"
    assert event["purchaseUrl"].endswith("ref=buy-now")
    assert event["originalUrl"].endswith("ref=buy-now")
    assert event["referralCode"] == "buy-now"
    assert headers["Cache-Control"] == f"public, max-age={app.EVENT_CACHE_SECONDS}"
    assert headers["X-Robots-Tag"] == "noindex"
    assert analytics.last_update is not None
    update = analytics.last_update["update"]
    assert update["$inc"]["views"] == 1
    assert update["$set"]["partyId"] == "1"


def test_event_detail_api_not_found(monkeypatch):
    monkeypatch.setattr(app, "parties_collection", _DummyCollection([]))

    class _EmptySettings:
        def find_one(self, filter):
            return None

    monkeypatch.setattr(app, "settings_collection", _EmptySettings())

    payload, status = app.event_detail_api("missing")

    assert status == 404
    assert payload["message"] == "Event not found."
