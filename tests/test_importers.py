import sys

import app


class StubResponse:
    def __init__(self, json_data=None):
        self._json_data = json_data or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._json_data


def authenticated_headers():
    flask_mod = sys.modules["flask"]
    app.JWT_SECRET = "secret"
    flask_mod.request.headers = {"Authorization": "Bearer token"}


def test_scrape_nightlife_events_posts_api(monkeypatch):
    captured = {}

    monkeypatch.setattr(app, "_format_iso_timestamp", lambda now=None: "2025-10-14T19:04:11.691Z")

    def fake_post(url, json=None, headers=None, timeout=None, proxies=None):
        captured.update({
            "url": url,
            "json": json,
            "headers": headers,
            "timeout": timeout,
            "proxies": proxies,
        })
        return StubResponse({
            "status": True,
            "events": [{"Url": "lemuriajourney"}, {"Url": "1759921611905"}],
        })

    monkeypatch.setattr(app.requests, "post", fake_post)

    urls = app.scrape_nightlife_events("refcode")

    assert captured["url"] == "https://www.go-out.co/endOne/getEventsByTypeNew"
    assert captured["json"] == {
        "skip": 0,
        "limit": 50,
        "location": "IL",
        "Types": ["תל אביב", "מועדוני לילה"],
        "recivedDate": "2025-10-14T19:04:11.691Z",
    }
    assert captured["headers"]["Referer"] == "https://www.go-out.co/tickets/nightlife"
    assert captured["proxies"] == {"http": None, "https": None}
    assert urls == [
        "https://www.go-out.co/event/lemuriajourney?aff=refcode",
        "https://www.go-out.co/event/1759921611905?aff=refcode",
    ]


def test_scrape_weekend_events_gets_api(monkeypatch):
    captured = {}

    monkeypatch.setattr(app, "_format_weekend_recived_date", lambda now=None: "Tue Oct 14 2025 22:16:13 GMT+0300 (Israel Daylight Time)")

    def fake_get(url, params=None, headers=None, timeout=None, proxies=None):
        captured.update({
            "url": url,
            "params": params,
            "headers": headers,
            "timeout": timeout,
            "proxies": proxies,
        })
        return StubResponse({
            "status": True,
            "events": [{"Url": "1760287558015"}],
        })

    monkeypatch.setattr(app.requests, "get", fake_get)

    urls = app.scrape_weekend_events("refcode")

    assert captured["url"] == "https://www.go-out.co/endOne/getWeekendEvents"
    assert captured["params"] == {
        "limit": 50,
        "skip": 0,
        "recivedDate": "Tue Oct 14 2025 22:16:13 GMT+0300 (Israel Daylight Time)",
        "location": "IL",
    }
    assert captured["headers"]["Referer"] == "https://www.go-out.co/weekend"
    assert urls == ["https://www.go-out.co/event/1760287558015?aff=refcode"]


def test_import_nightlife_carousel_success(monkeypatch):
    authenticated_headers()

    monkeypatch.setattr(app, "default_referral_code", lambda: "refcode")

    captured = {}

    def fake_scrape(referral):
        captured["scrape_referral"] = referral
        return ["https://www.go-out.co/event/example?aff=refcode"]

    def fake_import(name, urls, referral):
        captured.update({
            "carousel_name": name,
            "import_urls": urls,
            "import_referral": referral,
        })
        return ({"_id": "c1", "partyIds": ["p1"]}, 1, [{"url": "bad", "error": "oops"}])

    monkeypatch.setattr(app, "scrape_nightlife_events", fake_scrape)
    monkeypatch.setattr(app, "_import_carousel_from_urls", fake_import)
    monkeypatch.setattr(app, "serialize_carousel", lambda doc: {"id": str(doc.get("_id")), "partyIds": doc.get("partyIds", [])})

    response, status = app.import_nightlife_carousel()

    assert status == 200
    assert response["message"].startswith("Nightlife carousel")
    assert response["carousel"]["id"] == "c1"
    assert response["addedCount"] == 1
    assert response["sourceEventCount"] == 1
    assert response["warnings"] == [{"url": "bad", "error": "oops"}]
    assert captured["carousel_name"] == app.NIGHTLIFE_CAROUSEL_TITLE
    assert captured["import_referral"] == "refcode"
    assert captured["scrape_referral"] == "refcode"


def test_import_weekend_carousel_handles_empty(monkeypatch):
    authenticated_headers()

    monkeypatch.setattr(app, "default_referral_code", lambda: "refcode")

    def fake_scrape(referral):
        assert referral == "refcode"
        return []

    monkeypatch.setattr(app, "scrape_weekend_events", fake_scrape)

    response, status = app.import_weekend_carousel()
    assert status == 404
    assert response["message"].startswith("No weekend events")
