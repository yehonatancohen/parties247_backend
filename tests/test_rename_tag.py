import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import app


def test_rename_tag_with_string_id(monkeypatch):
    tag_id = "0123456789abcdef01234567"
    tags = {tag_id: {"_id": tag_id, "name": "Old", "slug": "old"}}
    parties = [{"tags": ["Old", "Other"]}]

    class FakeTags:
        def find_one(self, query):
            if "_id" in query:
                return tags.get(query["_id"])
            if "slug" in query:
                for doc in tags.values():
                    if doc["slug"] == query["slug"]:
                        return doc
            return None

        def update_one(self, filt, update):
            doc = tags.get(filt["_id"])
            if doc:
                doc.update(update["$set"])

    class FakeParties:
        def update_many(self, filt, update, array_filters=None):
            old = filt["tags"]
            new = update["$set"]["tags.$[elem]"]
            for p in parties:
                p["tags"] = [new if t == old else t for t in p["tags"]]

    monkeypatch.setattr(app, "tags_collection", FakeTags())
    monkeypatch.setattr(app, "parties_collection", FakeParties())

    flask_mod = sys.modules["flask"]
    flask_mod.request.headers = {"Authorization": "Bearer token"}
    flask_mod.request.get_json = lambda silent=True: {"tagId": tag_id, "newName": "New"}

    res, status = app.rename_tag()
    assert status == 200
    assert tags[tag_id]["name"] == "New"
    assert parties[0]["tags"][0] == "New"
