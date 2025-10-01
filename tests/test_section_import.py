import importlib
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
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"Status code: {self.status_code}")
        return None


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
