from datetime import datetime, timedelta, timezone

import wa_facts


ISRAEL = wa_facts.ISRAEL_TZ


# ---------------------------------------------------------------------------
# Feature snapshot
# ---------------------------------------------------------------------------

def test_build_message_features_counts_lines_emoji_price():
    text = "🎉 party\nline2\n\n💸 ₪80"
    f = wa_facts.build_message_features(text)
    assert f["lineCount"] == 3  # blank line dropped
    assert f["emojiCount"] == 2
    assert f["hasPrice"] is True
    assert f["length"] == len(text)
    # same text -> same hash, stable for dedup/analysis
    assert f["textHash"] == wa_facts.build_message_features(text)["textHash"]
    assert f["textHash"] != wa_facts.build_message_features("other")["textHash"]


def test_build_party_features_snapshots_tier_and_hours_until():
    now = datetime(2026, 9, 10, 10, 0, tzinfo=timezone.utc)  # ~13:00 Israel
    party = {
        "goOutEventId": "111",
        "referralCode": "acc1ref",
        "ticketPrice": 80,
        "musicType": "טכנו",
        "date": "2026-09-11T23:00:00",  # naive Israel wall-clock
    }
    features = wa_facts.build_party_features(
        party, account_ids={"account1"}, account1_referral="acc1ref",
        ticket_totals={"confirmed_count": 5}, now=now,
    )
    assert features["tier"] == "account1"
    assert features["expectedPerTicket"] == 25.0
    assert features["ticketsSoldBefore"] == 5
    assert features["goOutEventId"] == "111"
    assert features["hoursUntilParty"] > 0  # party is in the future


def test_build_party_features_handles_missing_date_and_sales():
    features = wa_facts.build_party_features(
        {"goOutEventId": None, "ticketPrice": None}, account_ids=(),
        account1_referral=None, ticket_totals=None, now=datetime.now(timezone.utc),
    )
    assert features["tier"] == "account2"
    assert features["goOutEventId"] is None
    assert features["hoursUntilParty"] is None
    assert features["ticketsSoldBefore"] == 0


def test_build_group_snapshot_defaults_kind_and_fatigue():
    snap = wa_facts.build_group_snapshot(
        {"memberCount": 340, "kind": "community"}, chat_id="c1", sends_last_7d=2,
    )
    assert snap == {"chatId": "c1", "memberCount": 340, "kind": "community", "sendsToGroupLast7d": 2}

    snap_missing = wa_facts.build_group_snapshot(None, chat_id="c2", sends_last_7d=0)
    assert snap_missing["kind"] == "group"
    assert snap_missing["memberCount"] is None


def test_count_recent_sends_per_group_only_counts_window_and_group():
    before = datetime(2026, 9, 10, tzinfo=timezone.utc)
    campaigns = [
        {"createdAt": before - timedelta(days=1), "targets": [{"chatId": "c1"}]},  # in window, right group
        {"createdAt": before - timedelta(days=10), "targets": [{"chatId": "c1"}]},  # too old
        {"createdAt": before - timedelta(days=2), "targets": [{"chatId": "c2"}]},  # wrong group
        {"createdAt": before + timedelta(hours=1), "targets": [{"chatId": "c1"}]},  # after `before`
    ]
    assert wa_facts.count_recent_sends_per_group(campaigns, chat_id="c1", before=before) == 1


def test_count_unique_members_reached_dedupes_across_groups():
    members = [
        {"memberHash": "m1", "groups": ["c1"]},
        {"memberHash": "m2", "groups": ["c1", "c2"]},
        {"memberHash": "m2", "groups": ["c2"]},  # duplicate hash, still counts once
        {"memberHash": "m3", "groups": ["c3"]},  # not a target group
    ]
    assert wa_facts.count_unique_members_reached(members, chat_ids={"c1", "c2"}) == 2


# ---------------------------------------------------------------------------
# Fact table
# ---------------------------------------------------------------------------

def _campaign(**overrides):
    base = {
        "_id": "camp1",
        "partyId": "p1",
        "partySlug": "s1",
        "features": {"tier": "account1", "ticketPrice": 80, "goOutEventId": "111"},
        "messageFeatures": {"lineCount": 3},
        "groupSnapshots": [{"chatId": "c1", "memberCount": 200, "kind": "group"}],
        "targets": [{"chatId": "c1", "code": "abc123", "waMsgId": "wm1",
                     "sentAt": datetime(2026, 9, 10, 18, 0, tzinfo=timezone.utc)}],
    }
    base.update(overrides)
    return base


def test_build_send_facts_skips_unsent_targets():
    campaign = _campaign(targets=[{"chatId": "c1", "code": "abc", "sentAt": None}])
    facts = wa_facts.build_send_facts(
        [campaign], clicks=[], analytics_rows=[], snapshots=[], msg_events=[],
        now=datetime.now(timezone.utc),
    )
    assert facts == []


def test_build_send_facts_buckets_clicks_and_excludes_bots():
    sent_at = datetime(2026, 9, 10, 18, 0, tzinfo=timezone.utc)
    campaign = _campaign(targets=[{"chatId": "c1", "code": "abc123", "waMsgId": "wm1", "sentAt": sent_at}])
    clicks = [
        {"campaignId": "camp1", "chatId": "c1", "at": sent_at + timedelta(minutes=30), "isBot": False, "ipHash": "h1"},
        {"campaignId": "camp1", "chatId": "c1", "at": sent_at + timedelta(hours=5), "isBot": False, "ipHash": "h2"},
        {"campaignId": "camp1", "chatId": "c1", "at": sent_at + timedelta(hours=50), "isBot": False, "ipHash": "h3"},
        {"campaignId": "camp1", "chatId": "c1", "at": sent_at + timedelta(minutes=5), "isBot": True, "ipHash": "bot"},
        {"campaignId": "camp1", "chatId": "c1", "at": sent_at - timedelta(minutes=5), "isBot": False, "ipHash": "early"},
    ]
    facts = wa_facts.build_send_facts(
        [campaign], clicks=clicks, analytics_rows=[], snapshots=[], msg_events=[],
        now=datetime.now(timezone.utc),
    )
    assert len(facts) == 1
    o = facts[0]["outcomes"]
    assert o["clicks_1h"] == 1     # only the 30-min click
    assert o["clicks_6h"] == 2     # + the 5h click
    assert o["clicks_24h"] == 2    # 50h click still outside
    assert o["clicks_72h"] == 3    # 50h click now counted
    assert facts[0]["uniqueClickers"] == 3  # bot and early (pre-send) excluded


def test_build_send_facts_buy_clicks_and_cross_party():
    sent_at = datetime(2026, 9, 10, 18, 0, tzinfo=timezone.utc)
    campaign = _campaign(targets=[{"chatId": "c1", "code": "abc123", "waMsgId": "wm1", "sentAt": sent_at}])
    analytics_rows = [
        {"waCode": "abc123", "action": "redirect", "partyId": "p1", "createdAt": sent_at + timedelta(minutes=10)},
        {"waCode": "abc123", "action": "view", "partyId": "p1", "createdAt": sent_at + timedelta(minutes=1)},
        {"waCode": "abc123", "action": "redirect", "partyId": "OTHER", "createdAt": sent_at + timedelta(minutes=20)},
        {"waCode": "different-code", "action": "redirect", "partyId": "p1", "createdAt": sent_at + timedelta(minutes=5)},
    ]
    facts = wa_facts.build_send_facts(
        [campaign], clicks=[], analytics_rows=analytics_rows, snapshots=[], msg_events=[],
        now=datetime.now(timezone.utc),
    )
    o = facts[0]["outcomes"]
    assert o["buyClicks_1h"] == 1       # same-party redirect only
    assert facts[0]["crossPartyBuyClicks"] == 1


def test_build_send_facts_reactions_and_replies():
    sent_at = datetime(2026, 9, 10, 18, 0, tzinfo=timezone.utc)
    campaign = _campaign(targets=[{"chatId": "c1", "code": "abc123", "waMsgId": "wm1", "sentAt": sent_at}])
    msg_events = [
        {"waMsgId": "wm1", "chatId": "c1", "type": "reaction", "at": sent_at + timedelta(minutes=5)},
        {"waMsgId": "wm1", "chatId": "c1", "type": "reply", "at": sent_at + timedelta(hours=2)},
        {"waMsgId": "wm1", "chatId": "OTHER_CHAT", "type": "reaction", "at": sent_at + timedelta(minutes=1)},
        {"waMsgId": "wm1", "chatId": "c1", "type": "deleted", "at": sent_at + timedelta(minutes=1)},
    ]
    facts = wa_facts.build_send_facts(
        [campaign], clicks=[], analytics_rows=[], snapshots=[], msg_events=msg_events,
        now=datetime.now(timezone.utc),
    )
    o = facts[0]["outcomes"]
    assert o["reactions_1h"] == 1
    assert o["replies_1h"] == 0
    assert o["replies_6h"] == 1


def test_build_send_facts_sales_delta_and_commission_account1():
    sent_at = datetime(2026, 9, 10, 18, 0, tzinfo=timezone.utc)
    campaign = _campaign(targets=[{"chatId": "c1", "code": "abc123", "waMsgId": "wm1", "sentAt": sent_at}])
    snapshots = [
        {"go_out_id": "111", "at": sent_at - timedelta(hours=24), "accepted": 10},
        {"go_out_id": "111", "at": sent_at, "accepted": 12},
        {"go_out_id": "111", "at": sent_at + timedelta(hours=6), "accepted": 16},
        {"go_out_id": "111", "at": sent_at + timedelta(hours=24), "accepted": 20},
    ]
    facts = wa_facts.build_send_facts(
        [campaign], clicks=[], analytics_rows=[], snapshots=snapshots, msg_events=[],
        now=datetime.now(timezone.utc),
    )
    s = facts[0]["sales"]
    assert s["attribution"] == "time-window"
    assert s["baselineAcceptedPer24h"] == 2   # 12 - 10 over the prior 24h
    assert s["acceptedDelta6h"] == 4          # 16 - 12
    assert s["acceptedDelta24h"] == 8         # 20 - 12
    assert s["estCommission6h"] == 100.0      # account1: 4 tickets * ₪25
    assert s["estCommission24h"] == 200.0


def test_build_send_facts_sales_uses_account2_percentage():
    sent_at = datetime(2026, 9, 10, 18, 0, tzinfo=timezone.utc)
    campaign = _campaign(features={"tier": "account2", "ticketPrice": 100, "goOutEventId": "222"},
                          targets=[{"chatId": "c1", "code": "abc123", "waMsgId": "wm1", "sentAt": sent_at}])
    snapshots = [
        {"go_out_id": "222", "at": sent_at, "accepted": 0},
        {"go_out_id": "222", "at": sent_at + timedelta(hours=6), "accepted": 5},
    ]
    facts = wa_facts.build_send_facts(
        [campaign], clicks=[], analytics_rows=[], snapshots=snapshots, msg_events=[],
        now=datetime.now(timezone.utc),
    )
    s = facts[0]["sales"]
    assert s["acceptedDelta6h"] == 5
    assert s["estCommission6h"] == 30.0  # 5 tickets * ₪100 * 6%


def test_build_send_facts_no_baseline_snapshot_yields_none_sales():
    sent_at = datetime(2026, 9, 10, 18, 0, tzinfo=timezone.utc)
    campaign = _campaign(targets=[{"chatId": "c1", "code": "abc123", "waMsgId": "wm1", "sentAt": sent_at}])
    # only a post-send snapshot exists -- no pre-send baseline to diff against
    snapshots = [{"go_out_id": "111", "at": sent_at + timedelta(hours=6), "accepted": 20}]
    facts = wa_facts.build_send_facts(
        [campaign], clicks=[], analytics_rows=[], snapshots=snapshots, msg_events=[],
        now=datetime.now(timezone.utc),
    )
    assert facts[0]["sales"]["acceptedDelta6h"] is None
    assert facts[0]["sales"]["estCommission6h"] is None


def test_build_send_facts_missing_event_id_has_no_sales_join():
    sent_at = datetime(2026, 9, 10, 18, 0, tzinfo=timezone.utc)
    campaign = _campaign(features={"tier": "account1", "ticketPrice": 80, "goOutEventId": None},
                          targets=[{"chatId": "c1", "code": "abc123", "waMsgId": "wm1", "sentAt": sent_at}])
    facts = wa_facts.build_send_facts(
        [campaign], clicks=[], analytics_rows=[], snapshots=[{"go_out_id": "111", "at": sent_at, "accepted": 1}],
        msg_events=[], now=datetime.now(timezone.utc),
    )
    assert facts[0]["sales"]["acceptedDelta6h"] is None


def test_build_send_facts_local_hour_dow_across_dst_exit():
    # 2026-10-25 02:30 UTC is after Israel's DST exit (Asia/Jerusalem back to UTC+2)
    sent_at = datetime(2026, 10, 25, 18, 0, tzinfo=timezone.utc)
    campaign = _campaign(targets=[{"chatId": "c1", "code": "abc123", "waMsgId": "wm1", "sentAt": sent_at}])
    facts = wa_facts.build_send_facts(
        [campaign], clicks=[], analytics_rows=[], snapshots=[], msg_events=[],
        now=datetime.now(timezone.utc),
    )
    assert facts[0]["localHour"] == 20  # UTC+2 post-DST, not UTC+3
    assert facts[0]["localDow"] == 6    # Sunday


def test_build_send_facts_is_idempotent_pure_function():
    sent_at = datetime(2026, 9, 10, 18, 0, tzinfo=timezone.utc)
    campaign = _campaign(targets=[{"chatId": "c1", "code": "abc123", "waMsgId": "wm1", "sentAt": sent_at}])
    now = datetime.now(timezone.utc)
    facts_a = wa_facts.build_send_facts([campaign], clicks=[], analytics_rows=[], snapshots=[], msg_events=[], now=now)
    facts_b = wa_facts.build_send_facts([campaign], clicks=[], analytics_rows=[], snapshots=[], msg_events=[], now=now)
    assert facts_a == facts_b


def test_build_send_facts_multiple_targets_get_own_rows():
    sent_at = datetime(2026, 9, 10, 18, 0, tzinfo=timezone.utc)
    campaign = _campaign(
        groupSnapshots=[{"chatId": "c1", "memberCount": 200}, {"chatId": "c2", "memberCount": 50}],
        targets=[
            {"chatId": "c1", "code": "code1", "waMsgId": "wm1", "sentAt": sent_at},
            {"chatId": "c2", "code": "code2", "waMsgId": "wm2", "sentAt": sent_at + timedelta(minutes=1)},
        ],
    )
    facts = wa_facts.build_send_facts(
        [campaign], clicks=[], analytics_rows=[], snapshots=[], msg_events=[],
        now=datetime.now(timezone.utc),
    )
    assert {f["chatId"] for f in facts} == {"c1", "c2"}
    assert {f["group"]["memberCount"] for f in facts} == {200, 50}


def test_build_send_facts_handles_naive_datetimes_like_real_pymongo_reads():
    """pymongo hands back naive UTC datetimes by default (no tz_aware=True
    on this app's MongoClient) -- every other test in this file uses aware
    datetimes for convenience, which doesn't match what campaigns/clicks/
    snapshots actually look like once round-tripped through Mongo in
    production. A sibling app.py route (wa_sales_watchlist) shipped with a
    naive-vs-aware comparison bug that this module's own internal logic
    does NOT have (every comparison here is naive-vs-naive or defensively
    normalized, see _local_hour_dow/build_party_features) -- this test
    pins that down against a real regression, not just code inspection."""
    sent_at = datetime(2026, 9, 10, 18, 0)  # naive
    campaign = _campaign(targets=[{"chatId": "c1", "code": "abc123", "waMsgId": "wm1", "sentAt": sent_at}])
    clicks = [{"campaignId": "camp1", "chatId": "c1", "at": sent_at + timedelta(minutes=30), "isBot": False, "ipHash": "h1"}]
    snapshots = [
        {"go_out_id": "111", "at": sent_at - timedelta(hours=24), "accepted": 10},
        {"go_out_id": "111", "at": sent_at + timedelta(hours=6), "accepted": 15},
    ]
    facts = wa_facts.build_send_facts(
        [campaign], clicks=clicks, analytics_rows=[], snapshots=snapshots, msg_events=[],
        now=datetime.now(timezone.utc),  # aware `now`, as every real caller passes
    )
    assert facts[0]["outcomes"]["clicks_1h"] == 1
    assert facts[0]["sales"]["acceptedDelta6h"] == 5
    assert facts[0]["localHour"] == 21  # 18:00 UTC -> 21:00 Israel (IDT, +3)
