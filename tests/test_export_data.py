from datetime import datetime, timedelta, timezone

import export_data

START = datetime(2026, 9, 30, 21, 0, tzinfo=timezone.utc)  # 2026-10-01 00:00 Israel
END = datetime(2026, 10, 31, 22, 0, tzinfo=timezone.utc)   # 2026-11-01 00:00 Israel
NOW = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)

PARTIES = {
    "p1": {"name": "Moon", "slug": "moon", "date": "2026-10-30T22:00:00", "goOutEventId": "111"},
    # A duplicate merged into p1: its views, clicks and sales belong to p1.
    "p1-dup": {"name": "Moon (dup)", "slug": "moon-dup", "goOutEventId": "112",
               "listingStatus": "merged", "mergedInto": "p1"},
    "p2": {"name": "Sun", "slug": "sun", "date": "2026-10-20T22:00:00", "goOutEventId": "222"},
}


def _build(**over):
    args = dict(
        start=START, end=END, parties=PARTIES,
        view_buckets=[
            {"partyId": "p1", "day": "2026-10-28", "action": "view", "channel": "instagram", "bot": False, "count": 40},
            {"partyId": "p1", "day": "2026-10-28", "action": "view", "channel": "direct", "bot": True, "count": 5},
            {"partyId": "p1-dup", "day": "2026-10-28", "action": "view", "channel": "google", "bot": False, "count": 5},
            {"partyId": "p1", "day": "2026-10-28", "action": "redirect", "channel": "instagram", "bot": False, "count": 2},
            {"partyId": "p2", "day": "2026-10-10", "action": "view", "channel": "google", "bot": False, "count": 10},
        ],
        unique_sessions={"p1": {"view": 30, "redirect": 2}, "p1-dup": {"view": 4}},
        clicks=[
            # naive UTC, like pymongo returns
            {"createdAt": datetime(2026, 10, 28, 9, 0), "partyId": "p1", "channel": "instagram",
             "firstSource": "ig", "deviceType": "mobile", "sessionId": "abcdef123456"},
            {"createdAt": datetime(2026, 10, 28, 10, 0), "partyId": "p1-dup", "channel": "google"},
            # before the window: listed nowhere, but still joined to a sale right after it opens
            {"createdAt": datetime(2026, 9, 30, 20, 0), "partyId": "p2", "channel": "whatsapp"},
        ],
        sales=[
            {"account_id": "account1", "go_out_id": "111", "delta_confirmed": 2, "revenue_earned": 30.0,
             "delta_event_revenue": 200.0, "recorded_at": datetime(2026, 10, 28, 10, 30), "source": "fast_sales_alerts"},
            {"account_id": "account2", "go_out_id": "222", "delta_confirmed": 1, "revenue_earned": 6.0,
             "delta_event_revenue": 100.0, "recorded_at": datetime(2026, 9, 30, 22, 0)},
            # a double count corrected to zero: not a sale
            {"account_id": "account2", "go_out_id": "222", "delta_confirmed": 0, "revenue_earned": 0.0,
             "delta_event_revenue": 0.0, "recorded_at": datetime(2026, 10, 29, 8, 16),
             "double_count_fix": {"revenue_earned": 26.28}},
        ],
        snapshots=[
            {"go_out_id": "111", "account_id": "account1", "views": 90, "real_own_revenue": 500,
             "endone_stats": {"ticket_stats": {"Accepted": 5, "Abandoned": 3},
                              "sales_per_date": {"dates": {"1793145600000": 2}}}},
        ],
        lifetime_by_event={"111": {"totalTicketsSold": 5, "totalRevenue": 75.0}},
        now=NOW,
    )
    args.update(over)
    return export_data.build_export(**args)


def test_merged_listing_folds_into_its_keeper():
    out = _build()
    moon = next(p for p in out["parties"] if p["partyName"] == "Moon")
    assert moon["views"] == 50 and moon["botViews"] == 5
    assert moon["clicks"] == 2
    assert moon["uniqueViewers"] == 34
    assert moon["tickets"] == 2 and moon["commission"] == 30.0
    assert moon["clickRate"] == 4.0 and moon["ticketsPerClick"] == 100.0
    assert moon["goOutViews"] == 90 and moon["goOutAbandoned"] == 3 and moon["lifetimeCommission"] == 75.0
    assert moon["siteUrl"] == "https://www.parties247.co.il/event/moon"
    assert not any(p["partyName"] == "Moon (dup)" for p in out["parties"])


def test_each_sale_carries_the_clicks_before_it():
    out = _build()
    sale = next(s for s in out["sales"] if s["account"] == "account1")
    assert sale["recordedAt"] == "2026-10-28T12:30:00"  # Israel time
    assert sale["source"] == "fast (20 min)"
    assert sale["clicks24hBefore"] == 2               # its own click and the merged duplicate's
    assert sale["lastClickChannel"] == "google" and sale["minutesFromLastClick"] == 30
    assert sale["daysBeforeParty"] == 2.4
    # the account2 sale right after the window opened still sees the click from before it
    early = next(s for s in out["sales"] if s["account"] == "account2")
    assert early["clicks24hBefore"] == 1 and early["lastClickChannel"] == "whatsapp"


def test_corrected_zero_rows_and_out_of_window_clicks_are_left_out():
    out = _build()
    assert len(out["sales"]) == 2
    # Israel is on winter time (UTC+2) from 10-25
    assert [c["clickedAt"] for c in out["clicks"]] == ["2026-10-28T11:00:00", "2026-10-28T12:00:00"]
    assert [c["saleWithin24h"] for c in out["clicks"]] == [True, True]  # the sale was at 12:30
    assert out["clicks"][0]["session"] == "abcdef12"


def test_totals_days_and_channels():
    out = _build()
    t = out["totals"]
    assert (t["views"], t["clicks"], t["tickets"], t["commission"]) == (60, 2, 3, 36.0)
    assert {a["account"]: a["commission"] for a in t["byAccount"]} == {"account1": 30.0, "account2": 6.0}
    assert out["range"] == {"from": "2026-10-01", "to": "2026-10-31", "generatedAt": "2026-10-29T14:00:00"}
    day = next(d for d in out["daily"] if d["day"] == "2026-10-28")
    assert day["views"] == 50 and day["tickets"] == 2
    ig = next(c for c in out["channels"] if c["channel"] == "instagram")
    assert ig["views"] == 40 and ig["clicks"] == 2
    assert out["goOutDaily"] == [{"day": "2026-10-28", "partyName": "Moon", "account": "account1",
                                  "tickets": 2, "goOutEventId": "111"}]


def test_all_time_has_no_lower_bound():
    out = _build(start=None)
    assert out["range"]["from"] is None
    assert len(out["clicks"]) == 3


class _FakeCollection:
    """Answers find() with its docs and aggregate() by which pipeline it was given."""

    def __init__(self, docs=None, aggregates=None):
        self.docs = docs or []
        self.aggregates = aggregates or {}
        self.projections = []

    def find(self, query=None, projection=None, *args, **kwargs):
        self.projections.append(projection)
        return [dict(d) for d in self.docs]

    def aggregate(self, pipeline, **kwargs):
        last = pipeline[-1]["$group"]["_id"]
        if "day" in last:
            return self.aggregates.get("buckets", [])
        if "sessionId" in last or last == {"partyId": "$_id.partyId", "action": "$_id.action"}:
            return self.aggregates.get("unique", [])
        return self.aggregates.get("lifetime", [])

    def find_one(self, *args, **kwargs):
        return None


def test_load_export_wires_the_reads_and_keeps_ips_out(monkeypatch):
    import app

    analytics = _FakeCollection(
        docs=[{"createdAt": datetime(2026, 10, 28, 9, 0), "partyId": "p1", "channel": "google",
               "referer": "https://www.google.com/search?q=x"}],
        aggregates={
            "buckets": [{"_id": {"partyId": "p1", "day": "2026-10-28", "action": "view",
                                 "channel": "google", "bot": False}, "count": 7}],
            "unique": [{"_id": {"partyId": "p1", "action": "view"}, "n": 5}],
        },
    )
    sales_log = _FakeCollection(docs=[{"account_id": "account1", "go_out_id": "111", "delta_confirmed": 1,
                                       "revenue_earned": 15.0, "recorded_at": datetime(2026, 10, 28, 10, 0)}])
    monkeypatch.setattr(app, "analytics_collection", analytics)
    monkeypatch.setattr(app, "parties_collection", _FakeCollection(docs=[{"_id": "p1", **PARTIES["p1"]}]))
    monkeypatch.setattr(app, "goout_sales_log_collection", sales_log)
    monkeypatch.setattr(app, "goout_sales_collection", _FakeCollection())

    out = app.load_export(START, END)

    assert out["totals"]["views"] == 7 and out["totals"]["tickets"] == 1
    assert out["clicks"][0]["referrerHost"] == "www.google.com"
    assert out["sales"][0]["clicks24hBefore"] == 1
    click_projection = next(p for p in analytics.projections if p and "createdAt" in p)
    assert "clientIp" not in click_projection and "userAgent" not in click_projection
