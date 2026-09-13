import sys
from datetime import datetime, timedelta, timezone

import app


def _authorize(monkeypatch, token="test-token"):
    flask_mod = sys.modules["flask"]
    monkeypatch.setattr(app, "SERVICE_TOKEN", token)
    flask_mod.request.headers = {"X-Service-Token": token}


class FakeCampaigns:
    def __init__(self, docs):
        self.docs = docs

    def find(self, query):
        return list(self.docs)


def _campaign(**overrides):
    base = {
        "_id": "camp1",
        "partyId": "p1",
        "status": "done",
        "features": {"goOutEventId": "111"},
        "targets": [],
    }
    base.update(overrides)
    return base


def test_wa_sales_watchlist_requires_service_token(monkeypatch):
    flask_mod = sys.modules["flask"]
    flask_mod.request.headers = {}
    monkeypatch.setattr(app, "SERVICE_TOKEN", "test-token")
    res, status = app.wa_sales_watchlist()
    assert status == 401


def test_wa_sales_watchlist_handles_naive_datetimes_from_real_pymongo_reads(monkeypatch):
    """Regression test: pymongo hands back naive UTC datetimes by default
    (no tz_aware=True on this app's MongoClient — see app.py's MongoClient
    construction), while this route's own `cutoff` is timezone-aware
    (datetime.now(timezone.utc) - timedelta(...)). Comparing them directly
    raised "can't compare offset-naive and offset-aware datetimes" for
    every real campaign with a sent target — confirmed live in production
    on 2026-09-13 via parties247_fetcher's wa_sales_watch.py hitting a 500
    within minutes of this route's first deploy. Every doc here uses naive
    datetimes on purpose, matching that real shape exactly."""
    _authorize(monkeypatch)
    now = datetime.now(timezone.utc)
    recent_sent = datetime.utcnow() - timedelta(hours=2)  # naive, like a real pymongo read
    campaigns = FakeCampaigns([
        _campaign(targets=[{"chatId": "c1", "sentAt": recent_sent}]),
    ])
    monkeypatch.setattr(app, "wa_campaigns_collection", campaigns)

    res, status = app.wa_sales_watchlist()

    assert status == 200
    assert res["watchlist"] == [{"goOutEventId": "111", "partyId": "p1", "reasons": ["recentlySent"]}]


def test_wa_sales_watchlist_flags_queued_campaigns_without_a_sent_target(monkeypatch):
    _authorize(monkeypatch)
    campaigns = FakeCampaigns([
        _campaign(status="queued", targets=[{"chatId": "c1", "sentAt": None}]),
    ])
    monkeypatch.setattr(app, "wa_campaigns_collection", campaigns)

    res, status = app.wa_sales_watchlist()

    assert status == 200
    assert res["watchlist"][0]["reasons"] == ["queued"]


def test_wa_sales_watchlist_drops_campaigns_with_no_resolvable_event_id(monkeypatch):
    _authorize(monkeypatch)
    monkeypatch.setattr(app, "parties_collection", None)  # find_party_for_analytics fallback finds nothing
    campaigns = FakeCampaigns([
        _campaign(features={}, partyId=None, targets=[{"chatId": "c1", "sentAt": datetime.utcnow()}]),
    ])
    monkeypatch.setattr(app, "wa_campaigns_collection", campaigns)

    res, status = app.wa_sales_watchlist()

    assert status == 200
    assert res["watchlist"] == []


def test_wa_sales_watchlist_merges_multiple_campaigns_for_the_same_event(monkeypatch):
    _authorize(monkeypatch)
    old_sent = datetime.utcnow() - timedelta(hours=48)  # outside the 24h window
    campaigns = FakeCampaigns([
        _campaign(status="queued", targets=[{"chatId": "c1", "sentAt": None}]),
        _campaign(status="done", targets=[{"chatId": "c2", "sentAt": old_sent}]),
    ])
    monkeypatch.setattr(app, "wa_campaigns_collection", campaigns)

    res, status = app.wa_sales_watchlist()

    assert status == 200
    assert len(res["watchlist"]) == 1
    assert res["watchlist"][0]["reasons"] == ["queued"]  # the stale send doesn't add recentlySent


def test_wa_sales_watchlist_returns_500_on_unexpected_error_instead_of_crashing(monkeypatch):
    _authorize(monkeypatch)

    class ExplodingCampaigns:
        def find(self, query):
            raise RuntimeError("Atlas hiccup")

    monkeypatch.setattr(app, "wa_campaigns_collection", ExplodingCampaigns())

    res, status = app.wa_sales_watchlist()

    assert status == 500
    assert "message" in res
