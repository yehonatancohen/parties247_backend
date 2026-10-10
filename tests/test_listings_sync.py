"""Listing Guard wiring in app.py: the sync diff, public/hidden reads, the
single ingest helper and the audit's handling of merges and issues."""
import os
import sys
import types

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import app

NOW = "2026-10-09T09:00:00+00:00"

EVENT = {
    "Title": "FRIDAY MAINSTREAM | 16.10",
    "StartingDate": "2026-10-16T23:00:00.000",
    "Adress": "Moonchild, Tel Aviv-Yafo, ישראל",
    "Location": {"lat": 32.0681, "lng": 34.7622, "city": "Tel Aviv-Yafo"},
    "EventPublicity": "פומבי",
    "EventSerial": 48000,
    "Url": "1790000000000",
    "Description": "line one\nline two",
    # The public page's placeholder — must never become the price.
    "Tickets": [{"Title": "", "Price": 200, "Commision": 5, "Amount": "150", "sold": 0, "Active": True}],
}
TIERS = [
    {"Title": "כניסה לכל הלילה", "Price": 80, "Commision": 9.25, "display": "ACTIVE"},
    {"Title": "הרשמה חינם בהגעה עד 23:30", "Price": 0, "Commision": 0, "display": "ACTIVE"},
]


def sync(party, item):
    return app.compute_listing_sync(party, item, {}, NOW)


def test_full_sync_fixes_price_name_and_region_in_one_pass():
    party = {"_id": "p1", "name": "FRIDAY MAINSTREAM | 09.10", "ticketPrice": 0, "soldOut": False,
             "date": "2026-10-16T23:00:00.000", "location": "Moonchild, Tel Aviv-Yafo, ישראל", "region": "לא ידוע"}
    outcome = sync(party, {"partyId": "p1", "event": EVENT, "tiers": TIERS})

    assert outcome["changes"]["ticketPrice"] == [0, 87.4]
    assert outcome["changes"]["name"] == ["FRIDAY MAINSTREAM | 09.10", "FRIDAY MAINSTREAM | 16.10"]
    assert outcome["changes"]["region"] == ["לא ידוע", "מרכז"]
    assert outcome["set"]["priceInfo"]["freeLabel"] == "כניסה חופשית עד 23:30"
    assert outcome["set"]["source"]["tiers"][0]["final"] == 87.4
    assert "listingStatus" not in outcome["set"]


def test_tiers_only_sync_keeps_the_page_snapshot():
    full = sync({"_id": "p1"}, {"partyId": "p1", "event": EVENT, "tiers": TIERS})["set"]
    party = {"_id": "p1", **full}
    raised = [{**TIERS[0], "Price": 100}, TIERS[1]]
    outcome = sync(party, {"partyId": "p1", "tiers": raised})

    assert outcome["changes"] == {"ticketPrice": [87.4, 109.25]}
    assert outcome["set"]["source"]["title"] == "FRIDAY MAINSTREAM | 16.10"


def test_page_only_sync_keeps_known_tiers():
    party = {"_id": "p1", **sync({"_id": "p1"}, {"partyId": "p1", "event": EVENT, "tiers": TIERS})["set"]}
    outcome = sync(party, {"partyId": "p1", "event": {**EVENT, "Title": "FRIDAY MAINSTREAM | 16.10 🔥"}})

    assert outcome["set"]["source"]["tiers"] == party["source"]["tiers"]
    assert outcome["changes"] == {"name": ["FRIDAY MAINSTREAM | 16.10", "FRIDAY MAINSTREAM | 16.10 🔥"]}


def test_failed_tier_fetches_keep_the_price_then_stop_vouching_for_it():
    party = {"_id": "p1", **sync({"_id": "p1"}, {"partyId": "p1", "event": EVENT, "tiers": TIERS})["set"]}
    for expected_verified in (True, True, False):
        outcome = sync(party, {"partyId": "p1", "tiers": None})
        assert outcome["changes"] == {}
        assert outcome["set"]["priceInfo"]["verified"] is expected_verified
        party = {**party, **outcome["set"]}
    assert party["ticketPrice"] == 87.4 and party["source"]["tiersFailCount"] == 3


def test_missing_page_counts_failures_without_touching_fields():
    party = {"_id": "p1", **sync({"_id": "p1"}, {"partyId": "p1", "event": EVENT, "tiers": TIERS})["set"]}
    outcome = sync(party, {"partyId": "p1", "pageStatus": 404})
    assert outcome["changes"] == {}
    assert outcome["set"]["source"]["failCount"] == 1 and outcome["set"]["source"]["title"] == EVENT["Title"]


def test_private_event_is_hidden_by_the_sync():
    outcome = sync({"_id": "p1", "name": "x"}, {"partyId": "p1", "event": {**EVENT, "EventPublicity": "פרטי"}})
    assert outcome["set"]["listingStatus"] == "hidden" and outcome["set"]["statusReason"] == "private"
    assert outcome["changes"]["listingStatus"] == ["live", "hidden"]


def test_admin_locked_price_survives_the_sync():
    party = {"_id": "p1", "ticketPrice": 70, "locks": ["ticketPrice"]}
    outcome = sync(party, {"partyId": "p1", "event": EVENT, "tiers": TIERS})
    assert "ticketPrice" not in outcome["changes"] and "ticketPrice" not in outcome["set"]
    assert outcome["set"]["priceInfo"]["from"] == 87.4  # the real one is still recorded


# --- reads ------------------------------------------------------------------

class _Cursor(list):
    def sort(self, *args, **kwargs):
        return self


def test_public_reads_hide_unlisted_parties_and_internal_reads_do_not(monkeypatch):
    docs = [
        {"_id": "1", "name": "Live", "date": "2099-01-01", "source": {"title": "Live"}, "locks": ["name"]},
        {"_id": "2", "name": "Private", "date": "2099-01-02", "listingStatus": "hidden"},
        {"_id": "3", "name": "Dup", "date": "2099-01-03", "listingStatus": "merged"},
    ]
    monkeypatch.setattr(app, "parties_collection",
                        types.SimpleNamespace(find=lambda *a, **k: _Cursor(dict(d) for d in docs)))
    monkeypatch.setattr(app, "settings_collection", types.SimpleNamespace(find_one=lambda q: {}))

    public = app._fetch_parties_cached({})
    assert [p["name"] for p in public] == ["Live"]
    assert "source" not in public[0] and "locks" not in public[0]

    everything = app._fetch_parties_cached({}, include_hidden=True)
    assert [p["name"] for p in everything] == ["Live", "Private", "Dup"]
    assert everything[0]["locks"] == ["name"]

    assert [d["name"] for d in app.all_events()] == ["Live"]
    assert len(app.all_events(include_hidden=True)) == 3


def test_include_hidden_needs_a_trusted_caller(monkeypatch):
    flask_request = sys.modules["flask"].request
    monkeypatch.setattr(app, "SERVICE_TOKEN", "svc")
    monkeypatch.setattr(app, "JWT_SECRET", "secret")
    monkeypatch.setattr(flask_request, "headers", {})
    assert app._request_is_trusted() is False
    monkeypatch.setattr(flask_request, "headers", {"X-Service-Token": "svc"})
    assert app._request_is_trusted() is True
    monkeypatch.setattr(flask_request, "headers", {"Authorization": "Bearer token"})
    assert app._request_is_trusted() is True
    monkeypatch.setattr(flask_request, "headers", {"Authorization": "Bearer forged"})
    assert app._request_is_trusted() is False


# --- writes -----------------------------------------------------------------

class FakeParties:
    def __init__(self, docs=()):
        self.docs = [dict(d) for d in docs]
        self.updates = []

    def _match(self, doc, query):
        if "$or" in query:
            return any(self._match(doc, clause) for clause in query["$or"])
        for key, value in query.items():
            if isinstance(value, dict) and "$in" in value:
                if doc.get(key) not in value["$in"]:
                    return False
            elif isinstance(value, dict) and "$gte" in value:
                if not (doc.get(key) or "") >= value["$gte"]:
                    return False
            elif doc.get(key) != value:
                return False
        return True

    def find(self, query=None, projection=None):
        return _Cursor(dict(d) for d in self.docs if self._match(d, query or {}))

    def find_one(self, query, projection=None):
        return next((dict(d) for d in self.docs if self._match(d, query)), None)

    def update_one(self, query, update, upsert=False):
        self.updates.append((query, update))
        for doc in self.docs:
            if self._match(doc, query):
                doc.update(update.get("$set", {}))
                return types.SimpleNamespace(matched_count=1, upserted_id=None)
        if not upsert:
            return types.SimpleNamespace(matched_count=0, upserted_id=None)
        new = dict(update.get("$setOnInsert", {}))
        new.setdefault("_id", f"p{len(self.docs) + 1}")
        self.docs.append(new)
        return types.SimpleNamespace(matched_count=0, upserted_id=new["_id"])


def test_upsert_party_doc_creates_once_and_never_overwrites(monkeypatch):
    parties = FakeParties()
    monkeypatch.setattr(app, "parties_collection", parties)
    data = {"name": "A", "canonicalUrl": "https://go-out.co/event/1", "ticketPrice": 50}

    assert app.upsert_party_doc(dict(data)) == ("p1", True)
    assert parties.docs[0]["listingStatus"] == "live"
    assert app.upsert_party_doc({**data, "name": "changed", "ticketPrice": 1}) == ("p1", False)
    assert parties.docs[0]["name"] == "A" and parties.docs[0]["ticketPrice"] == 50


class FakeIssues:
    def __init__(self, docs=()):
        self.docs = [dict(d) for d in docs]

    def find(self, query=None, projection=None):
        statuses = ((query or {}).get("status") or {}).get("$in")
        return _Cursor(d for d in self.docs if not statuses or d.get("status") in statuses)

    def update_one(self, query, update, upsert=False):
        for doc in self.docs:
            if doc["fingerprint"] == query["fingerprint"]:
                doc.update(update.get("$set", {}))
                return types.SimpleNamespace(upserted_id=None)
        self.docs.append({**query, **update.get("$setOnInsert", {}), **update.get("$set", {})})
        return types.SimpleNamespace(upserted_id="new")

    def update_many(self, query, update):
        def matches(doc):
            status, prints = query.get("status"), query.get("fingerprint") or {}
            if isinstance(status, dict) and doc.get("status") not in status["$in"]:
                return False
            if isinstance(status, str) and doc.get("status") != status:
                return False
            if "type" in query and doc.get("type") != query["type"]:
                return False
            if "$nin" in prints and doc["fingerprint"] in prints["$nin"]:
                return False
            if "$in" in prints and doc["fingerprint"] not in prints["$in"]:
                return False
            if "firstSeen" in query and not doc.get("firstSeen") < query["firstSeen"]["$lt"]:
                return False
            return True

        changed = 0
        for doc in self.docs:
            if matches(doc):
                doc.update(update["$set"])
                changed += 1
        return types.SimpleNamespace(modified_count=changed)

    def count_documents(self, query):
        return sum(1 for d in self.docs if d.get("status") == query.get("status"))


def _listed(pid, title, ref, **extra):
    source = {"title": title, "startsAt": "2099-10-17T17:00:00.000", "fetchedAt": "2099-10-16T12:00:00",
              "geo": {"lat": 32.07, "lng": 34.78}, "address": "Club 1, Tel Aviv"}
    return {"_id": pid, "name": title, "slug": f"slug-{pid}", "date": "2099-10-17T17:00:00.000",
            "referralCode": ref, "goOutEventId": f"e{pid}", "location": "Club 1, Tel Aviv",
            "source": source, **extra}


revalidated = []


def _audit_env(monkeypatch, parties, issues, recent_changes=()):
    merged = []
    revalidated.clear()
    monkeypatch.setattr(app, "listing_changes_collection", types.SimpleNamespace(
        find=lambda query, projection=None: [{"partyId": pid} for pid in recent_changes]))
    monkeypatch.setattr(app, "parties_collection", parties)
    monkeypatch.setattr(app, "listing_issues_collection", issues)
    monkeypatch.setattr(app, "listing_rules_collection", types.SimpleNamespace(find=lambda q: []))
    monkeypatch.setattr(app, "_sales_totals_by_event_id", lambda cutoff=None: {})
    monkeypatch.setattr(app, "_accounts_by_event_id", lambda: {})
    monkeypatch.setattr(app, "_revalidate_parties", lambda parties: revalidated.extend(p["_id"] for p in parties))
    monkeypatch.setattr(app, "_local_now", lambda: app.datetime(2099, 10, 16, 13))
    monkeypatch.setattr(app, "_set_listing_guard_state", lambda **fields: None)
    monkeypatch.setattr(app, "merge_listing", lambda loser, keeper, reason: merged.append((loser["_id"], keeper["_id"], reason)))
    return merged


def test_audit_merges_duplicates_into_the_best_paying_listing_and_asks_nothing(monkeypatch):
    parties = FakeParties([
        {**_listed("1", "WineNot? In The City 17.10🌅", "acc2"), "ticketPrice": 120},   # 6% = ₪7.20
        _listed("2", "WineNot In The City 17.10🌅🍷", "acc1"),                           # flat fee
        {**_listed("3", "Other Name 17.10", "acc2"), "ticketPrice": 90},                 # same door, same hour
        _listed("9", "Hidden Private 17.10", "acc2", listingStatus="hidden"),
    ])
    issues = FakeIssues([{"fingerprint": "stale:gone", "status": "open", "type": "stale_sync"}])
    merged = _audit_env(monkeypatch, parties, issues)

    summary = app.run_listing_audit("acc1", [])

    assert sorted(merged) == [("1", "2", "certain:commission"), ("3", "2", "certain:commission")]
    by_print = {doc["fingerprint"]: doc for doc in issues.docs}
    assert by_print["dup:e1:e2"]["status"] == by_print["dup:e2:e3"]["status"] == "auto_fixed"
    assert not any("e9" in fp for fp in by_print)              # hidden parties are out of it
    assert by_print["stale:gone"]["status"] == "resolved"      # cause went away
    assert summary["waiting"] == 0 and summary["new"] == 0 and summary["cleared"] == 1


def test_audit_dry_run_writes_nothing(monkeypatch):
    parties = FakeParties([_listed("1", "Same 17.10", "acc2"), _listed("2", "Same 17.10", "acc1")])
    issues = FakeIssues()
    merged = _audit_env(monkeypatch, parties, issues)

    summary = app.run_listing_audit("acc1", [], dry_run=True)

    assert merged == [] and issues.docs == [] and parties.updates == []
    assert summary["merges"][0]["keeper"] == "Same 17.10" and summary["merges"][0]["keeperId"] == "2"


def test_a_merge_undone_by_hand_is_not_repeated(monkeypatch):
    parties = FakeParties([_listed("2", "Night A 17.10", "acc2"), _listed("3", "Night B 17.10", "acc2")])
    issues = FakeIssues([{"fingerprint": "dup:e2:e3", "status": "resolved", "type": "duplicate"}])
    merged = _audit_env(monkeypatch, parties, issues)

    summary = app.run_listing_audit("acc1", [])

    assert merged == [] and summary["waiting"] == 0 and issues.docs[0]["status"] == "resolved"


def test_stale_page_is_rerendered_first_and_only_asked_about_if_it_stays_wrong(monkeypatch):
    parties = FakeParties([{**_listed("1", "Alpha 17.10", "acc2"), "ticketPrice": 87.4},
                           {**_listed("2", "Beta 19.10", "acc2", date="2099-10-19T17:00:00.000"), "ticketPrice": 50}])
    parties.docs[1]["source"]["startsAt"] = "2099-10-19T17:00:00.000"
    issues = FakeIssues()
    # Party 1 changed in the last two hours: its page is still cached, not checked.
    _audit_env(monkeypatch, parties, issues, recent_changes=["1"])
    checks = [{"partyId": "1", "status": 200, "price": "0"}, {"partyId": "2", "status": 200, "price": "0"}]

    summary = app.run_listing_audit("acc1", checks)

    assert [(doc["fingerprint"], doc["status"]) for doc in issues.docs] == [("site:e2:price", "watching")]
    assert revalidated == ["2"] and summary["waiting"] == 0

    issues.docs[0]["firstSeen"] = issues.docs[0]["firstSeen"] - app.timedelta(days=1)
    summary = app.run_listing_audit("acc1", checks)
    assert issues.docs[0]["status"] == "open" and summary["waiting"] == 1


def test_corrected_start_time_also_lands_in_starts_at():
    party = {"_id": "p1", "name": EVENT["Title"], "date": "2026-10-16T22:00:00.000",
             "startsAt": "2026-10-16T22:00:00.000"}
    outcome = sync(party, {"partyId": "p1", "event": EVENT, "tiers": TIERS})
    assert outcome["set"]["date"] == outcome["set"]["startsAt"] == EVENT["StartingDate"]
    locked = sync({**party, "locks": ["date"]}, {"partyId": "p1", "event": EVENT, "tiers": TIERS})
    assert "date" not in locked["set"] and "startsAt" not in locked["set"]
