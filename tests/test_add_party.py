import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app


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


class FakeCarouselsCollection:
    def __init__(self):
        self.docs: list[dict] = []

    def find(self, *args, **kwargs):
        return FakeCursor(self.docs)

    def find_one(self, query):
        for doc in self.docs:
            if _matches(doc, query):
                return doc.copy()
        return None

    def insert_one(self, doc):
        stored = doc.copy()
        stored.setdefault("_id", f"c{len(self.docs) + 1}")
        self.docs.append(stored)
        return types.SimpleNamespace(inserted_id=stored["_id"])

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


flask_mod = sys.modules["flask"]


def authenticated_headers():
    app.JWT_SECRET = "secret"
    flask_mod.request.headers = {"Authorization": "Bearer token"}


def test_add_party_creates_carousel(monkeypatch):
    authenticated_headers()
    parties = FakePartiesCollection()
    carousels = FakeCarouselsCollection()
    monkeypatch.setattr(app, "parties_collection", parties)
    monkeypatch.setattr(app, "carousels_collection", carousels)
    monkeypatch.setattr(app, "default_referral_code", lambda: "refcode")
    monkeypatch.setattr(app, "notify_indexers", lambda urls: None)
    monkeypatch.setattr(app, "trigger_revalidation", lambda paths: None)
    monkeypatch.setattr(
        app,
        "scrape_party_details",
        lambda url: {
            "name": "Great Event",
            "canonicalUrl": url,
            "originalUrl": url,
            "goOutUrl": url,
        },
    )

    flask_mod.request.get_json = lambda silent=True: {
        "url": "https://example.com/event/awesome",
        "carouselName": "Highlights",
    }

    payload, status = app.add_party()

    assert status == 201
    assert payload["carousel"]["title"] == "Highlights"
    assert payload["addedToCarousel"] is True
    assert len(carousels.docs) == 1
    assert carousels.docs[0]["partyIds"]


def test_add_party_existing_appends_to_carousel(monkeypatch):
    authenticated_headers()
    parties = FakePartiesCollection()
    parties.docs.append(
        {
            "_id": "p1",
            "canonicalUrl": "https://example.com/event/awesome",
            "goOutUrl": "https://example.com/event/awesome",
        }
    )
    carousels = FakeCarouselsCollection()
    carousels.docs.append({"_id": "c1", "title": "Highlights", "partyIds": []})

    monkeypatch.setattr(app, "parties_collection", parties)
    monkeypatch.setattr(app, "carousels_collection", carousels)
    monkeypatch.setattr(app, "default_referral_code", lambda: "refcode")
    monkeypatch.setattr(app, "notify_indexers", lambda urls: None)
    monkeypatch.setattr(app, "trigger_revalidation", lambda paths: None)
    monkeypatch.setattr(
        app,
        "scrape_party_details",
        lambda url: {
            "name": "Great Event",
            "canonicalUrl": url,
            "originalUrl": url,
            "goOutUrl": url,
        },
    )

    flask_mod.request.get_json = lambda silent=True: {
        "url": "https://example.com/event/awesome",
        "carouselName": "Highlights",
    }

    payload, status = app.add_party()

    assert status == 200
    assert payload["addedToCarousel"] is True
    assert payload["carousel"]["partyIds"] == ["p1"]
    assert carousels.docs[0]["partyIds"] == ["p1"]
