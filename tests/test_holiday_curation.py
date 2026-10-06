import app


class FakeSettings:
    def __init__(self):
        self.docs = {}

    def find_one(self, query):
        return self.docs.get(query["key"])

    def update_one(self, query, update, upsert=False):
        self.docs.setdefault(query["key"], {}).update(update["$set"])


def test_parse_dedupes_and_pinned_wins_over_hidden():
    out = app.parse_holiday_curation({"partyIds": ["a1", "b2", "a1"], "hiddenIds": ["b2", "c3"]})
    assert out == {"partyIds": ["a1", "b2"], "hiddenIds": ["c3"]}


def test_parse_defaults_missing_lists_to_empty():
    assert app.parse_holiday_curation({}) == {"partyIds": [], "hiddenIds": []}


def test_parse_rejects_bad_bodies():
    assert app.parse_holiday_curation(None) is None
    assert app.parse_holiday_curation({"partyIds": "a1"}) is None
    assert app.parse_holiday_curation({"partyIds": [1]}) is None
    assert app.parse_holiday_curation({"partyIds": ["$where"]}) is None
    assert app.parse_holiday_curation({"partyIds": [], "extra": 1}) is None
    assert app.parse_holiday_curation({"partyIds": [f"p{i}" for i in range(301)]}) is None


def test_load_returns_empty_when_unset_and_saved_lists_when_set(monkeypatch):
    settings = FakeSettings()
    monkeypatch.setattr(app, "settings_collection", settings)
    assert app.load_holiday_curation("halloween")["partyIds"] == []

    settings.update_one({"key": "holidayPage:halloween"}, {"$set": {"partyIds": ["b", "a"], "hiddenIds": ["c"]}})
    loaded = app.load_holiday_curation("halloween")
    assert loaded["partyIds"] == ["b", "a"] and loaded["hiddenIds"] == ["c"]
    assert app.load_holiday_curation("sylvester")["partyIds"] == []


def test_slug_pattern():
    assert app.HOLIDAY_SLUG_RE.fullmatch("yom-haatzmaut")
    assert not app.HOLIDAY_SLUG_RE.fullmatch("../etc")
