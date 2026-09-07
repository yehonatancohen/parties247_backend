"""
The admin analytics endpoints were slow because they loaded whole collections
(full party docs incl. descriptions; full goout_sales docs incl. endOne buyer
lists) two or three times per request. These tests pin the fix: one projected
party load shared by both maps, and projected goout_sales scans.
"""

import app


class RecordingCollection:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def find(self, *args, **kwargs):
        self.calls.append(args[1] if len(args) > 1 else kwargs.get("projection"))
        return list(self.docs)

    def aggregate(self, pipeline):
        return []


def test_fetch_all_documents_passes_projection_and_falls_back_for_stubs():
    coll = RecordingCollection([{"_id": "1"}])
    assert app.fetch_all_documents(coll, projection={"name": 1}) == [{"_id": "1"}]
    assert coll.calls == [{"name": 1}]

    class NoArgs:
        def find(self):
            return [{"_id": "x"}]

    assert app.fetch_all_documents(NoArgs(), projection={"name": 1}) == [{"_id": "x"}]
    assert app.fetch_all_documents(None) == []


def test_load_party_index_is_one_projected_read(monkeypatch):
    parties = RecordingCollection([
        {"_id": "p1", "name": "One", "slug": "one", "date": "2026-09-10T22:00:00", "goOutEventId": 111},
        {"_id": "p2", "name": "Two", "slug": "two", "date": "2026-09-11T22:00:00"},
    ])
    monkeypatch.setattr(app, "parties_collection", parties)

    by_id, by_event = app._load_party_index()
    assert set(by_id) == {"p1", "p2"}
    assert by_event == {"111": {"partyId": "p1", "partyName": "One", "partySlug": "one",
                                "partyDate": app.isoformat_or_none("2026-09-10T22:00:00")}}
    assert len(parties.calls) == 1
    projection = parties.calls[0]
    assert projection == app._PARTY_INDEX_PROJECTION
    assert "description" not in projection and "images" not in projection


def test_funnel_and_sales_use_projections(monkeypatch):
    parties = RecordingCollection([
        {"_id": "p1", "name": "One", "slug": "one", "date": "2026-09-10T22:00:00", "goOutEventId": "111"},
    ])
    sales = RecordingCollection([
        {"go_out_id": "111", "account_id": "account1", "views": 5, "real_own_revenue": 300.0,
         "endone_stats": {"views": {"Views": 5}}},
    ])
    analytics = RecordingCollection([])
    monkeypatch.setattr(app, "parties_collection", parties)
    monkeypatch.setattr(app, "goout_sales_collection", sales)
    monkeypatch.setattr(app, "goout_sales_log_collection", RecordingCollection([]))
    monkeypatch.setattr(app, "analytics_collection", analytics)

    funnel = app.build_party_funnel(days=30)
    assert len(parties.calls) == 1, "funnel must load parties exactly once"
    assert sales.calls == [app._SALES_FUNNEL_PROJECTION]
    assert funnel["byParty"][0]["partyId"] == "p1"
    assert funnel["byParty"][0]["realGoOutViews"] == 5
    assert funnel["byParty"][0]["accountIds"] == ["account1"]

    parties.calls.clear()
    sales.calls.clear()
    rows = app.build_sales_by_party()
    assert parties.calls == [app._PARTY_INDEX_PROJECTION]
    assert sales.calls == [app._SALES_BY_PARTY_PROJECTION]
    assert rows[0]["partyName"] == "One" and rows[0]["realOwnRevenue"] == 300.0
