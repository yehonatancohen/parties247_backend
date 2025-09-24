import types

import app


def sample_event_doc():
    return {
        "slug": "mega-party",
        "name": "מסיבה",
        "nameHe": "מסיבה",
        "nameEn": "Mega Party",
        "descriptionHe": "תיאור",
        "descriptionEn": "Description",
        "summaryHe": "תקציר",
        "summaryEn": "Summary",
        "startsAt": "2099-05-01T19:00:00Z",
        "updatedAt": "2099-04-01T00:00:00Z",
        "city": {
            "slug": "tel-aviv",
            "name": {"he": "תל אביב", "en": "Tel Aviv"},
            "geo": {"lat": 32.08, "lon": 34.78, "address": "Tel Aviv", "postalCode": "61000"},
        },
        "venue": {
            "slug": "the-block",
            "name": {"he": "הבלוק", "en": "The Block"},
        },
        "genres": [
            {"slug": "techno", "name": {"he": "טכנו", "en": "Techno"}},
            "Mainstream",
        ],
        "geo": {"lat": 32.08, "lon": 34.78, "address": "Somewhere"},
    }


class DummyCollection:
    def __init__(self, docs):
        self.docs = docs

    def find(self):
        return list(self.docs)


class DummySettings:
    def __init__(self):
        self.calls = []

    def update_one(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return types.SimpleNamespace()


def configure_collections(monkeypatch, docs=None):
    docs = docs or [sample_event_doc()]
    monkeypatch.setattr(app, "parties_collection", DummyCollection(docs))
    monkeypatch.setattr(app, "settings_collection", DummySettings())


def test_normalize_event_builds_bilingual_fields():
    normalized = app.normalize_event(sample_event_doc())
    assert normalized["title"]["he"] == "מסיבה"
    assert normalized["title"]["en"] == "Mega Party"
    assert normalized["status"] == "scheduled"
    assert normalized["city"]["slug"] == "tel-aviv"
    assert normalized["canonicalUrl"].endswith("/event/mega-party")


def test_events_api_returns_expected_structure(monkeypatch):
    configure_collections(monkeypatch)
    payload, status, headers = app.list_events_api()
    assert status == 200
    assert headers["Cache-Control"] == f"public, max-age={app.EVENT_CACHE_SECONDS}"
    assert payload["items"][0]["title"]["en"] == "Mega Party"


def test_dimension_endpoints(monkeypatch):
    configure_collections(monkeypatch)
    city_payload, _, _ = app.list_cities_api()
    venue_payload, _, _ = app.list_venues_api()
    genre_payload, _, _ = app.list_genres_api()
    assert city_payload["items"][0]["slug"] == "tel-aviv"
    assert venue_payload["items"][0]["slug"] == "the-block"
    genre_slugs = {item["slug"] for item in genre_payload["items"]}
    assert "techno" in genre_slugs


def test_sitemap_and_feeds(monkeypatch):
    configure_collections(monkeypatch)
    body, status, headers = app.sitemap_index()
    assert status == 200
    assert "<sitemapindex" in body
    assert app.build_canonical("sitemap_child", name="events-0.xml") in body

    feed_body, feed_status, feed_headers = app.feeds_events("rss")
    assert feed_status == 200
    assert feed_headers["Content-Type"].startswith("application/rss+xml")
    assert "<rss" in feed_body


def test_ics_and_robots(monkeypatch):
    configure_collections(monkeypatch)
    ics_body, ics_status, ics_headers = app.ics_event("mega-party")
    assert ics_status == 200
    assert ics_body.startswith("BEGIN:VCALENDAR")
    robots_body, robots_status, _ = app.robots_txt()
    assert robots_status == 200
    assert "Sitemap:" in robots_body


def test_event_related_paths(monkeypatch):
    configure_collections(monkeypatch)
    event = app.serialize_events(include_past=False)[0]
    paths = app.event_related_paths(event)
    assert "/event/mega-party" in paths
    assert any(path.startswith("/city/") for path in paths)
