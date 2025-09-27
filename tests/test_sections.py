import sys
import types

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


class FakeSectionsCollection:
    def __init__(self):
        self.docs: list[dict] = []

    def find(self, filter=None, projection=None):
        filter = filter or {}
        projection = projection or {}
        items = []
        wanted_ids = None
        if "_id" in filter and isinstance(filter["_id"], dict) and "$in" in filter["_id"]:
            wanted_ids = {str(value) for value in filter["_id"]["$in"]}
        for doc in self.docs:
            if wanted_ids is not None:
                if str(doc.get("_id")) not in wanted_ids:
                    continue
            else:
                match = True
                for key, value in filter.items():
                    if key == "_id" and not isinstance(value, dict):
                        match = str(doc.get("_id")) == str(value)
                    elif key != "_id" and doc.get(key) != value:
                        match = False
                    if not match:
                        break
                if not match:
                    continue
            items.append(doc.copy())
        if projection:
            include_keys = [key for key, enabled in projection.items() if enabled]
            if include_keys:
                trimmed = []
                for doc in items:
                    trimmed.append({key: doc[key] for key in include_keys if key in doc})
                items = trimmed
        return FakeCursor(items)

    def find_one(self, filter):
        filter = filter or {}
        if "slug" in filter:
            for doc in self.docs:
                if doc.get("slug") == filter["slug"]:
                    return doc.copy()
        if "_id" in filter:
            wanted = str(filter["_id"])
            for doc in self.docs:
                if str(doc.get("_id")) == wanted:
                    return doc.copy()
        return None

    def insert_one(self, doc):
        stored = doc.copy()
        stored.setdefault("_id", str(len(self.docs) + 1))
        self.docs.append(stored)
        return types.SimpleNamespace(inserted_id=stored["_id"])

    def update_one(self, filter, update, **kwargs):
        updates = update.get("$set", {})
        for doc in self.docs:
            if str(doc.get("_id")) == str(filter.get("_id")):
                doc.update(updates)
                return types.SimpleNamespace(matched_count=1)
        return types.SimpleNamespace(matched_count=0)

    def delete_one(self, filter):
        for index, doc in enumerate(self.docs):
            if str(doc.get("_id")) == str(filter.get("_id")):
                self.docs.pop(index)
                return types.SimpleNamespace(deleted_count=1)
        return types.SimpleNamespace(deleted_count=0)


flask_mod = sys.modules["flask"]


def authenticated_headers():
    app.JWT_SECRET = "secret"
    flask_mod.request.headers = {"Authorization": "Bearer token"}


def test_admin_add_section_assigns_slug_and_order(monkeypatch):
    authenticated_headers()
    fake = FakeSectionsCollection()
    monkeypatch.setattr(app, "sections_collection", fake)

    flask_mod.request.get_json = lambda silent=True: {
        "title": "Weekend Picks",
        "content": "Lots of parties",
    }
    created, status = app.add_section()
    assert status == 201
    assert created["slug"] == "weekend-picks"
    assert created["order"] == 0

    flask_mod.request.get_json = lambda silent=True: {
        "title": "Late Night",
        "content": "After hours",
    }
    second, status = app.add_section()
    assert status == 201
    assert second["order"] == 1


def test_admin_update_section_allows_slug_change(monkeypatch):
    authenticated_headers()
    fake = FakeSectionsCollection()
    seed = {
        "title": "Original",
        "content": "Body",
        "slug": "original",
        "order": 0,
        "createdAt": "2024-01-01T00:00:00Z",
        "updatedAt": "2024-01-01T00:00:00Z",
    }
    inserted = fake.insert_one(seed)
    monkeypatch.setattr(app, "sections_collection", fake)

    flask_mod.request.get_json = lambda silent=True: {
        "slug": "Weekend Specials",
        "content": " Updated body ",
    }
    payload, status = app.update_section(inserted.inserted_id)
    assert status == 200
    assert payload["section"]["slug"] == "weekend-specials"
    stored = fake.find_one({"_id": inserted.inserted_id})
    assert stored["slug"] == "weekend-specials"
    assert stored["content"].startswith(" Updated body")


def test_reorder_sections_updates_order(monkeypatch):
    authenticated_headers()
    fake = FakeSectionsCollection()
    first = fake.insert_one({
        "title": "First",
        "content": "One",
        "slug": "first",
        "order": 0,
        "createdAt": "2024-01-01T00:00:00Z",
        "updatedAt": "2024-01-01T00:00:00Z",
    }).inserted_id
    second = fake.insert_one({
        "title": "Second",
        "content": "Two",
        "slug": "second",
        "order": 1,
        "createdAt": "2024-01-01T00:00:00Z",
        "updatedAt": "2024-01-01T00:00:00Z",
    }).inserted_id
    monkeypatch.setattr(app, "sections_collection", fake)

    flask_mod.request.get_json = lambda silent=True: {
        "orderedIds": [second, first],
    }
    payload, status = app.reorder_sections()
    assert status == 200
    orders = {doc["_id"]: doc["order"] for doc in fake.docs}
    assert orders[second] == 0
    assert orders[first] == 1


def test_public_sections_sorted(monkeypatch):
    fake = FakeSectionsCollection()
    fake.insert_one({"title": "C", "content": "desc", "slug": "c", "order": 2})
    fake.insert_one({"title": "A", "content": "desc", "slug": "a", "order": 0})
    fake.insert_one({"title": "B", "content": "desc", "slug": "b", "order": 1})
    monkeypatch.setattr(app, "sections_collection", fake)

    payload, status = app.list_sections()
    assert status == 200
    titles = [item["title"] for item in payload]
    assert titles == ["A", "B", "C"]
