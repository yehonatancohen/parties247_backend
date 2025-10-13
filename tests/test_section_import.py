import importlib
import json
import sys
import types
from pathlib import Path

import pytest

requests_stub = sys.modules.get("requests")
if isinstance(requests_stub, types.SimpleNamespace):
    sys.modules.pop("requests", None)
    importlib.invalidate_caches()

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app

setattr(app, "requests", requests)


class FakeCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    def sort(self, key, direction):
        reverse = direction == -1
        return FakeCursor(sorted(self._docs, key=lambda d: d.get(key, 0), reverse=reverse))

    def limit(self, count):
        return iter(self._docs[:count])

    def __iter__(self):
        return iter(self._docs)


def _matches(doc, query):
    if not query:
        return True
    if "$or" in query:
        return any(_matches(doc, clause) for clause in query["$or"])
    for key, value in query.items():
        if key == "$or":
            continue
        if isinstance(value, dict) and "$in" in value:
            if doc.get(key) not in value["$in"]:
                return False
        elif doc.get(key) != value:
            return False
    return True


class FakePartiesCollection:
    def __init__(self):
        self.docs: list[dict] = []

    def find_one(self, query, projection=None):
        for doc in self.docs:
            if _matches(doc, query):
                if projection:
                    return {key: doc[key] for key, enabled in projection.items() if enabled and key in doc}
                return doc.copy()
        return None

    def update_one(self, query, update, upsert=False):
        for doc in self.docs:
            if _matches(doc, query):
                return types.SimpleNamespace(matched_count=1, upserted_id=None)
        if not upsert:
            return types.SimpleNamespace(matched_count=0, upserted_id=None)
        payload = update.get("$setOnInsert", {}).copy()
        payload.setdefault("_id", f"p{len(self.docs) + 1}")
        self.docs.append(payload)
        return types.SimpleNamespace(matched_count=0, upserted_id=payload["_id"])

    def find(self, *args, **kwargs):
        return FakeCursor(self.docs)


class FakeCarouselsCollection:
    def __init__(self):
        self.docs: list[dict] = []

    def find(self, *args, **kwargs):
        return FakeCursor(self.docs)

    def insert_one(self, doc):
        stored = doc.copy()
        stored.setdefault("_id", f"c{len(self.docs) + 1}")
        self.docs.append(stored)
        return types.SimpleNamespace(inserted_id=stored["_id"])

    def find_one(self, query):
        for doc in self.docs:
            if _matches(doc, query):
                return doc.copy()
        return None

    def update_one(self, query, update):
        for doc in self.docs:
            if _matches(doc, query):
                if "$push" in update and "partyIds" in update["$push"]:
                    doc.setdefault("partyIds", [])
                    doc["partyIds"].append(update["$push"]["partyIds"])
                if "$set" in update:
                    doc.update(update["$set"])
                return types.SimpleNamespace(matched_count=1)
        return types.SimpleNamespace(matched_count=0)


class DummyResponse:
    def __init__(self, text: str = "", status_code: int = 200, json_data=None):
        self.text = text
        self.status_code = status_code
        self._json_data = json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"Status code: {self.status_code}")
        return None

    def json(self):
        if self._json_data is not None:
            return self._json_data
        return json.loads(self.text)


def fetch_live_page(url: str) -> requests.Response:
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        pytest.skip(f"Unable to fetch live data from {url}: {exc}")


flask_mod = sys.modules["flask"]


def authenticated_headers():
    app.JWT_SECRET = "secret"
    flask_mod.request.headers = {"Authorization": "Bearer token"}


def test_extract_event_urls_from_ticket_category_uses_api(monkeypatch):
    sample_html = """
    <html><head></head><body>
    <script id="__NEXT_DATA__" type="application/json">
    {"props":{"pageProps":{"pageInitialParams":{"ticketsRequest":{"skip":0,"limit":8,"Types":["אירועים","מועדוני לילה"],"location":"IL","recivedDate":"2025-10-13T13:08:47.589Z"}}}}}
    </script>
    </body></html>
    """

    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return DummyResponse(json_data={
            "status": True,
            "events": [{"Url": "1759874720952"}, {"Url": "1760017344268"}],
        })

    monkeypatch.setattr(app.requests, "post", fake_post)
    monkeypatch.setattr(app, "default_referral_code", lambda: "refcode")

    urls = app.extract_event_urls_from_page("https://www.go-out.co/tickets/nightlife", sample_html)

    assert captured["url"] == "https://www.go-out.co/endOne/getEventsByTypeNew"
    assert captured["json"] == {
        "skip": 0,
        "limit": 8,
        "location": "IL",
        "Types": ["אירועים", "מועדוני לילה"],
        "recivedDate": "2025-10-13T13:08:47.589Z",
    }
    assert set(captured["headers"].keys()) == {"Content-Type", "Accept", "Origin", "Referer", "User-Agent"}
    assert captured["headers"]["Referer"] == "https://www.go-out.co/tickets/nightlife"

    expected_urls = {
        "https://www.go-out.co/event/1759874720952?ref=refcode",
        "https://www.go-out.co/event/1760017344268?ref=refcode",
    }
    numeric_urls = {
        url
        for url in urls
        if url.rsplit("/event/", 1)[-1].split("?")[0].isdigit()
    }
    assert numeric_urls == expected_urls


def test_add_section_imports_ticket_category_via_api(monkeypatch):
    authenticated_headers()
    section_url = "https://www.go-out.co/tickets/nightlife"

    sample_html = """
    <html><head></head><body>
    <script id="__NEXT_DATA__" type="application/json">
    {"props":{"pageProps":{"pageInitialParams":{"ticketsRequest":{"skip":0,"limit":4,"Types":["אירועים","מועדוני לילה"],"location":"IL","recivedDate":"2025-10-13T13:08:47.589Z"}}}}}
    </script>
    </body></html>
    """

    def fake_get(url, headers=None, timeout=None):
        if url == section_url:
            return DummyResponse(sample_html, 200)
        raise AssertionError(f"Unexpected GET {url}")

    posted = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        posted["url"] = url
        posted["json"] = json
        posted["headers"] = headers
        return DummyResponse(json_data={
            "status": True,
            "events": [
                {"Url": "1759874720952"},
                {"Url": "1760017344268"},
            ],
        })

    monkeypatch.setattr(app.requests, "get", fake_get)
    monkeypatch.setattr(app.requests, "post", fake_post)
    monkeypatch.setattr(app, "default_referral_code", lambda: "refcode")
    monkeypatch.setattr(app, "notify_indexers", lambda urls: None)
    monkeypatch.setattr(app, "trigger_revalidation", lambda paths: None)

    captured_urls: list[str] = []
    valid_slugs = {"1759874720952", "1760017344268"}

    def fake_ensure_party_for_url(event_url, referral):
        slug = event_url.rsplit("/event/", 1)[-1].split("?")[0]
        if slug not in valid_slugs:
            return None, False
        captured_urls.append(event_url)
        return {"_id": f"party-{len(captured_urls)}", "goOutUrl": event_url}, True

    carousel_state = {"partyIds": [], "title": ""}

    def fake_ensure_carousel_contains_party(title, party_id):
        carousel_state["title"] = title
        added = False
        if party_id not in carousel_state["partyIds"]:
            carousel_state["partyIds"].append(party_id)
            added = True
        doc = {"_id": "carousel-1", "title": carousel_state["title"], "partyIds": list(carousel_state["partyIds"]) }
        return doc, added

    monkeypatch.setattr(app, "ensure_party_for_url", fake_ensure_party_for_url)
    monkeypatch.setattr(app, "ensure_carousel_contains_party", fake_ensure_carousel_contains_party)
    monkeypatch.setattr(app, "serialize_carousel", lambda doc: doc)
    monkeypatch.setattr(app, "parties_collection", object())
    monkeypatch.setattr(app, "carousels_collection", object())

    flask_mod.request.get_json = lambda silent=True: {
        "url": section_url,
        "carouselName": "Nightlife Highlights",
    }

    payload, status = app.add_section()

    assert status == 201
    assert payload["partyCount"] == 2
    assert payload["addedCount"] == 2
    assert payload["carousel"]["title"] == "Nightlife Highlights"

    assert posted["url"] == "https://www.go-out.co/endOne/getEventsByTypeNew"
    assert posted["json"]["Types"] == ["אירועים", "מועדוני לילה"]
    assert captured_urls == [
        "https://www.go-out.co/event/1759874720952?ref=refcode",
        "https://www.go-out.co/event/1760017344268?ref=refcode",
    ]

def test_add_section_creates_carousel_and_parties(monkeypatch):
    authenticated_headers()
    parties = FakePartiesCollection()
    carousels = FakeCarouselsCollection()
    monkeypatch.setattr(app, "parties_collection", parties)
    monkeypatch.setattr(app, "carousels_collection", carousels)
    monkeypatch.setattr(app, "default_referral_code", lambda: "refcode")
    monkeypatch.setattr(app, "notify_indexers", lambda urls: None)
    monkeypatch.setattr(app, "trigger_revalidation", lambda paths: None)
    section_url = "https://www.go-out.co/s/sukkot2025"
    section_response = fetch_live_page(section_url)

    live_event_urls = app.extract_event_urls_from_page(section_url, section_response.text)
    if not live_event_urls:
        pytest.skip("No events discovered from live section page")
    selected_event_urls = live_event_urls[:2]

    event_html_map: dict[str, str] = {}
    for event_url in selected_event_urls:
        response = fetch_live_page(event_url)
        event_html_map[event_url] = response.text

    monkeypatch.setattr(app, "extract_event_urls_from_page", lambda url, html: selected_event_urls)

    def fake_get(url, headers=None, timeout=None):
        if url == section_url:
            return DummyResponse(section_response.text, section_response.status_code)
        if url in event_html_map:
            return DummyResponse(event_html_map[url], 200)
        base_url = url.split("?", 1)[0]
        if base_url in event_html_map:
            return DummyResponse(event_html_map[base_url], 200)
        raise requests.HTTPError(f"Unexpected URL requested during test: {url}")

    monkeypatch.setattr(app.requests, "get", fake_get)

    flask_mod.request.get_json = lambda silent=True: {
        "url": section_url,
        "carouselName": "Weekend Specials",
    }

    payload, status = app.add_section()
    assert status == 201
    assert payload["carousel"]["title"] == "Weekend Specials"
    assert payload["partyCount"] >= 1
    assert payload["addedCount"] >= 1

    stored_ids = carousels.docs[0]["partyIds"]
    assert len(stored_ids) == payload["partyCount"]
    go_out_urls = [doc["goOutUrl"] for doc in parties.docs]
    assert all("ref=refcode" in url for url in go_out_urls)


def test_add_section_uses_next_data_events(monkeypatch):
    authenticated_headers()
    parties = FakePartiesCollection()
    carousels = FakeCarouselsCollection()
    monkeypatch.setattr(app, "parties_collection", parties)
    monkeypatch.setattr(app, "carousels_collection", carousels)
    monkeypatch.setattr(app, "default_referral_code", lambda: "affid")
    monkeypatch.setattr(app, "notify_indexers", lambda urls: None)
    monkeypatch.setattr(app, "trigger_revalidation", lambda paths: None)
    section_url = "https://www.go-out.co/tickets/festivals"
    section_response = fetch_live_page(section_url)

    live_event_urls = app.extract_event_urls_from_page(section_url, section_response.text)
    if not live_event_urls:
        pytest.skip("No events discovered from festivals page")
    selected_event_urls = live_event_urls[:2]

    event_html_map: dict[str, str] = {}
    for event_url in selected_event_urls:
        response = fetch_live_page(event_url)
        event_html_map[event_url] = response.text

    monkeypatch.setattr(app, "extract_event_urls_from_page", lambda url, html: selected_event_urls)

    def fake_get(url, headers=None, timeout=None):
        if url == section_url:
            return DummyResponse(section_response.text, section_response.status_code)
        if url in event_html_map:
            return DummyResponse(event_html_map[url], 200)
        base_url = url.split("?", 1)[0]
        if base_url in event_html_map:
            return DummyResponse(event_html_map[base_url], 200)
        raise requests.HTTPError(f"Unexpected URL requested during test: {url}")

    monkeypatch.setattr(app.requests, "get", fake_get)

    scraped_urls: list[str] = []
    original_scrape = app.scrape_party_details

    def tracking_scrape(url: str):
        scraped_urls.append(url)
        return original_scrape(url)

    monkeypatch.setattr(app, "scrape_party_details", tracking_scrape)

    flask_mod.request.get_json = lambda silent=True: {
        "url": section_url,
        "carouselName": "Verified Events",
    }

    payload, status = app.add_section()
    assert status == 201
    assert payload["partyCount"] >= 1
    assert payload["addedCount"] >= 1
    assert scraped_urls
    assert all("ref=affid" in url for url in scraped_urls)
    assert parties.docs
    assert all("ref=affid" in doc.get("goOutUrl", "") for doc in parties.docs)


def test_add_section_without_links_returns_not_found(monkeypatch):
    authenticated_headers()
    parties = FakePartiesCollection()
    carousels = FakeCarouselsCollection()
    monkeypatch.setattr(app, "parties_collection", parties)
    monkeypatch.setattr(app, "carousels_collection", carousels)
    monkeypatch.setattr(app, "default_referral_code", lambda: None)
    empty_url = "https://www.go-out.co/s/thispagedoesnotexist"
    empty_response = fetch_live_page(empty_url)

    discovered = app.extract_event_urls_from_page(empty_url, empty_response.text)
    if discovered:
        pytest.skip("Expected empty search results, but events were found")

    monkeypatch.setattr(
        app.requests,
        "get",
        lambda url, headers=None, timeout=None: DummyResponse(empty_response.text, empty_response.status_code),
    )

    flask_mod.request.get_json = lambda silent=True: {
        "url": empty_url,
        "carouselName": "Empty",
    }

    payload, status = app.add_section()
    assert status == 404
    assert "No parties" in payload["message"]
