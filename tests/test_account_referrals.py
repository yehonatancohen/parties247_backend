import app


class FakeSettings:
    def __init__(self):
        self.docs = {}

    def find_one(self, query):
        return self.docs.get(query["key"])

    def update_one(self, query, update, upsert=False):
        self.docs.setdefault(query["key"], {}).update(update["$set"])


def test_parse_accepts_two_different_codes_and_trims():
    assert app.parse_account_referrals({"account1": " a1 ", "account2": "b-2"}) == {"account1": "a1", "account2": "b-2"}


def test_parse_rejects_bad_bodies():
    assert app.parse_account_referrals(None) is None
    assert app.parse_account_referrals({"account1": "a1"}) is None
    assert app.parse_account_referrals({"account1": "a1", "account2": "b2", "x": "c"}) is None
    assert app.parse_account_referrals({"account1": "", "account2": "b2"}) is None
    assert app.parse_account_referrals({"account1": "a 1", "account2": "b2"}) is None
    assert app.parse_account_referrals({"account1": 1, "account2": "b2"}) is None
    assert app.parse_account_referrals({"account1": "same", "account2": "same"}) is None


def test_load_falls_back_to_defaults_then_returns_saved(monkeypatch):
    settings = FakeSettings()
    monkeypatch.setattr(app, "settings_collection", settings)
    assert app.load_account_referrals() == app.DEFAULT_ACCOUNT_REFERRALS

    settings.update_one({"key": app.ACCOUNT_REFERRALS_KEY},
                        {"$set": {"value": {"account1": "new1", "account2": "new2"}}})
    assert app.load_account_referrals() == {"account1": "new1", "account2": "new2"}
