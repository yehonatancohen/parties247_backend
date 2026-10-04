"""Bots (crawlers, prerenderers, Googlebot Smartphone) used to be stored and
counted as visitors and party views. They are now dropped at ingest, and
purge_bot_analytics removes the ones already stored."""

from datetime import datetime, timedelta, timezone

import app

GOOGLEBOT_SMARTPHONE = (
    "Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0.0.0 Mobile Safari/537.36 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
)
HEADLESS = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/129.0.0.0 Safari/537.36"
IPHONE = "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1"


def test_bot_detection_runs_before_mobile():
    assert app._parse_device_type(GOOGLEBOT_SMARTPHONE) == "bot"
    assert app._parse_device_type(HEADLESS) == "bot"
    assert app._parse_device_type(IPHONE) == "mobile"
    assert not app._is_bot_user_agent(IPHONE)
    assert not app._is_bot_user_agent(None)


class FakeColl:
    def __init__(self, docs):
        self.docs = docs
        self.deleted = []
        self.updates = []

    def find(self, query=None, projection=None):
        cutoff = (query or {}).get("createdAt", {}).get("$gte")
        return [d for d in self.docs if cutoff is None or d.get("createdAt", cutoff) >= cutoff]

    def delete_many(self, query):
        ids = set(query["_id"]["$in"])
        self.deleted.extend(ids)
        self.docs = [d for d in self.docs if d["_id"] not in ids]

    def update_one(self, query, update):
        self.updates.append((query, update))

    def update_many(self, query, update):
        pass


def test_purge_bot_analytics(monkeypatch):
    now = datetime.now(timezone.utc)
    visitors = FakeColl([
        {"_id": 1, "createdAt": now, "userAgent": HEADLESS},
        {"_id": 2, "createdAt": now, "userAgent": IPHONE},
        {"_id": 3, "createdAt": now - timedelta(days=5), "userAgent": HEADLESS},
    ])
    events = FakeColl([
        {"_id": "a", "createdAt": now, "userAgent": GOOGLEBOT_SMARTPHONE, "partyId": "p1", "action": "view"},
        {"_id": "b", "createdAt": now, "userAgent": GOOGLEBOT_SMARTPHONE, "partyId": "p1", "action": "redirect"},
        {"_id": "c", "createdAt": now, "userAgent": IPHONE, "partyId": "p1", "action": "view"},
    ])
    counters = FakeColl([])
    monkeypatch.setattr(app, "visitor_analytics_collection", visitors)
    monkeypatch.setattr(app, "analytics_collection", events)
    monkeypatch.setattr(app, "party_analytics_collection", counters)
    cutoff = now - timedelta(hours=48)

    dry = app.purge_bot_analytics(cutoff)
    assert (dry["visitorSessions"], dry["partyEvents"]) == (1, 2)
    assert dry["byParty"] == {"p1": {"views": 1, "redirects": 1}}
    assert visitors.deleted == [] and counters.updates == []

    app.purge_bot_analytics(cutoff, dry_run=False)
    assert visitors.deleted == [1]
    assert sorted(events.deleted) == ["a", "b"]
    assert counters.updates == [({"partyId": "p1"}, {"$inc": {"views": -1, "redirects": -1}})]
