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
    assert parties.calls == [], "the party index is cached and shared between builders"
    assert sales.calls == [app._SALES_BY_PARTY_PROJECTION]
    assert rows[0]["partyName"] == "One" and rows[0]["realOwnRevenue"] == 300.0


def test_summary_and_recent_project_party_reads(monkeypatch):
    # Both used to scan full party docs: /summary took ~20s, /recent ~8s.
    parties = RecordingCollection([{"_id": "p1", "name": "One", "slug": "one", "date": "2099-01-01T22:00:00"}])
    monkeypatch.setattr(app, "parties_collection", parties)
    monkeypatch.setattr(app, "party_analytics_collection", RecordingCollection([]))
    monkeypatch.setattr(app, "visitor_analytics_collection", RecordingCollection([]))

    summary = app.build_analytics_summary()
    assert [p["partyId"] for p in summary["parties"]] == ["p1"]
    assert parties.calls == [app._SUMMARY_PARTY_PROJECTION]
    assert "description" not in app._SUMMARY_PARTY_PROJECTION


def test_time_series_counts_inside_mongo(monkeypatch):
    # Raw docs are never pulled: the 30d/month/all-time chart timed out doing that.
    from datetime import datetime, timedelta, timezone

    class AggregatingCollection:
        def __init__(self, rows):
            self.rows = rows
            self.pipelines = []

        def find(self, *args, **kwargs):
            raise AssertionError("time series must not read raw documents")

        def aggregate(self, pipeline):
            self.pipelines.append(pipeline)
            return list(self.rows)

    visitors = AggregatingCollection([{"_id": "2026-10-15", "visits": 3}])
    events = AggregatingCollection([
        {"_id": {"b": "2026-10-15", "a": "view"}, "n": 10},
        {"_id": {"b": "2026-10-15", "a": "redirect"}, "n": 2},
        {"_id": {"b": "2026-10-16", "a": "view"}, "n": 4},
    ])
    monkeypatch.setattr(app, "visitor_analytics_collection", visitors)
    monkeypatch.setattr(app, "analytics_collection", events)
    monkeypatch.setattr(app, "find_party_for_analytics", lambda *a: {"_id": "p1", "slug": "one"})

    now = datetime(2026, 10, 16, 12, tzinfo=timezone.utc)
    rows = app.build_time_series_analytics(now - timedelta(days=30), now, "day", party_slug="one")

    assert rows == [
        {"timestamp": "2026-10-15", "visits": 3, "partyViews": 10, "purchases": 2},
        {"timestamp": "2026-10-16", "visits": 0, "partyViews": 4, "purchases": 0},
    ]
    match = events.pipelines[0][0]["$match"]
    assert match["category"] == "party" and match["createdAt"]["$gte"] == now - timedelta(days=30)
    assert match["$or"] == [{"partyId": {"$in": ["one", "p1"]}}, {"label": {"$in": ["one", "p1"]}}]
    app._time_series_via_aggregation(now, now, "hour", set())
    assert events.pipelines[-1][1]["$group"]["_id"]["b"]["$dateToString"]["format"] == "%Y-%m-%dT%H:00:00Z"


def test_admin_cache_reuses_until_ttl_or_new_sale(monkeypatch):
    builds = []

    def build():
        builds.append(1)
        return len(builds)

    assert app._admin_cached("k", build, stamp="t1") == 1
    assert app._admin_cached("k", build, stamp="t1") == 1, "same sales stamp -> cached"
    assert app._admin_cached("k", build, stamp="t2") == 2, "a newer sale rebuilds at once"
    assert app._admin_cached("k", build, stamp="t2", ttl=0) == 3, "expired -> rebuilt"


def test_admin_cache_does_not_store_failures():
    def boom():
        raise RuntimeError("down")

    try:
        app._admin_cached("k2", boom)
    except RuntimeError:
        pass
    assert app._admin_cached("k2", lambda: "ok") == "ok"


def test_sales_log_stamp_is_newest_recorded_at(monkeypatch):
    class Log:
        def find_one(self, query, projection, sort=None):
            assert sort == [("recorded_at", -1)]
            return {"recorded_at": "2026-10-16T21:00:00Z"}

    monkeypatch.setattr(app, "goout_sales_log_collection", Log())
    assert app._sales_log_stamp() == "2026-10-16T21:00:00Z"
    monkeypatch.setattr(app, "goout_sales_log_collection", None)
    assert app._sales_log_stamp() is None
