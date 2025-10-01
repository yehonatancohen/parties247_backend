import sys
import types
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app


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
    def __init__(self, text):
        self.text = text
        self.status_code = 200

    def raise_for_status(self):
        return None


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

    html = """
    <html><body>
    <a href="https://example.com/event/first">First</a>
    <a href="/event/second">Second</a>
    </body></html>
    """
    monkeypatch.setattr(app.requests, "get", lambda url, headers=None, timeout=None: DummyResponse(html))

    def fake_scrape(url):
        slug = url.rsplit("/", 1)[-1]
        return {
            "name": slug,
            "date": "2024-01-01T00:00:00Z",
            "location": "Tel Aviv",
            "description": "desc",
            "imageUrl": "https://example.com/img.jpg",
            "region": "מרכז",
            "musicType": "טכנו",
            "eventType": "מסיבת מועדון",
            "age": "18+",
            "tags": [],
            "originalUrl": url,
            "canonicalUrl": url,
            "goOutUrl": url,
        }

    monkeypatch.setattr(app, "scrape_party_details", fake_scrape)

    flask_mod.request.get_json = lambda silent=True: {
        "url": "https://example.com/collection",
        "carouselName": "Weekend Specials",
    }

    payload, status = app.add_section()
    assert status == 201
    assert payload["carousel"]["title"] == "Weekend Specials"
    assert payload["partyCount"] == 2
    assert payload["addedCount"] == 2

    stored_ids = carousels.docs[0]["partyIds"]
    assert len(stored_ids) == 2
    go_out_urls = [doc["goOutUrl"] for doc in parties.docs]
    assert all(url.endswith("refcode") for url in go_out_urls)


def test_add_section_uses_next_data_events(monkeypatch):
    authenticated_headers()
    parties = FakePartiesCollection()
    carousels = FakeCarouselsCollection()
    monkeypatch.setattr(app, "parties_collection", parties)
    monkeypatch.setattr(app, "carousels_collection", carousels)
    monkeypatch.setattr(app, "default_referral_code", lambda: "affid")
    monkeypatch.setattr(app, "notify_indexers", lambda urls: None)
    monkeypatch.setattr(app, "trigger_revalidation", lambda paths: None)

    html = """
    <html><head>
    <script id="__NEXT_DATA__" type="application/json">{"props":{"pageProps":{"pageInitialParams":{"events":[{"Url":"awesome-slug"}]}}}}</script>
    </head><body></body></html>
    """
    monkeypatch.setattr(app.requests, "get", lambda url, headers=None, timeout=None: DummyResponse(html))

    scraped_urls: list[str] = []

    def fake_scrape(url):
        scraped_urls.append(url)
        slug = urlparse(url).path.rsplit("/", 1)[-1]
        return {
            "name": slug,
            "date": "2024-01-01T00:00:00Z",
            "location": "Tel Aviv",
            "description": "desc",
            "imageUrl": "https://example.com/img.jpg",
            "region": "מרכז",
            "musicType": "טכנו",
            "eventType": "מסיבת מועדון",
            "age": "18+",
            "tags": [],
            "originalUrl": url,
            "canonicalUrl": url,
            "goOutUrl": url,
        }

    monkeypatch.setattr(app, "scrape_party_details", fake_scrape)

    flask_mod.request.get_json = lambda silent=True: {
        "url": "https://www.go-out.co/s/gooutverified",
        "carouselName": "Verified Events",
    }

    payload, status = app.add_section()
    assert status == 201
    assert payload["partyCount"] == 1
    assert scraped_urls == ["https://www.go-out.co/event/awesome-slug?ref=affid"]
    assert payload["addedCount"] == 1

    stored = parties.docs[0]
    assert stored["goOutUrl"].endswith("ref=affid")


def test_add_section_without_links_returns_not_found(monkeypatch):
    authenticated_headers()
    parties = FakePartiesCollection()
    carousels = FakeCarouselsCollection()
    monkeypatch.setattr(app, "parties_collection", parties)
    monkeypatch.setattr(app, "carousels_collection", carousels)
    monkeypatch.setattr(app, "default_referral_code", lambda: None)
    monkeypatch.setattr(app.requests, "get", lambda url, headers=None, timeout=None: DummyResponse("<html></html>"))

    flask_mod.request.get_json = lambda silent=True: {
        "url": "https://example.com/collection",
        "carouselName": "Empty",
    }

    payload, status = app.add_section()
    assert status == 404
    assert "No parties" in payload["message"]
