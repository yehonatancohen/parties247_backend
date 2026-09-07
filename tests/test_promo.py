from datetime import datetime, timedelta, timezone

import promo
import app


ISRAEL = promo.ISRAEL_TZ


def _party(**overrides):
    base = {
        "_id": "p1",
        "slug": "thursday-moon",
        "name": "THURSDAY MOON | MAINSTREAM",
        "date": "2026-09-10T23:00:00",
        "goOutEventId": "111",
        "goOutUrl": "https://www.go-out.co/event/moon?ref=acc2",
        "location": "Moon Child, תל אביב",
        "musicType": "מיינסטרים",
        "age": "18+",
        "ticketPrice": 80,
    }
    base.update(overrides)
    return base


NOW = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)  # Tuesday morning Israel time


def test_party_datetime_treats_naive_as_israel_time():
    dt = promo.party_datetime("2026-09-10T23:00:00")
    assert dt.tzinfo is not None
    assert dt.hour == 23 and dt.day == 10  # not shifted to the 11th via UTC
    assert dt.utcoffset() == timedelta(hours=3)  # IDT in September

    aware = promo.party_datetime("2026-09-10T20:00:00Z")
    assert aware.hour == 23  # converted into Israel time

    assert promo.party_datetime("garbage") is None
    assert promo.party_datetime(None) is None


def test_hebrew_date_label_uses_israel_weekday_and_time():
    dt = promo.party_datetime("2026-09-10T23:00:00")  # Thursday
    assert promo.hebrew_date_label(dt) == "יום חמישי 10.09 · 23:00"
    midnight = promo.party_datetime("2026-09-12T00:00:00")  # Saturday, no time shown
    assert promo.hebrew_date_label(midnight) == "יום שבת 12.09"


def test_account_tier_and_commission():
    assert promo.account_tier({"account1", "account2"}) == "account1"
    assert promo.account_tier({"account2"}) == "account2"
    assert promo.account_tier((), referral_code="ref1", account1_referral="ref1") == "account1"
    assert promo.account_tier((), referral_code="ref2", account1_referral="ref1") == "account2"

    assert promo.expected_commission_per_ticket("account1", 200) == 25.0
    assert promo.expected_commission_per_ticket("account2", 200) == 12.0
    # unknown price falls back to the default estimate rather than zero
    assert promo.expected_commission_per_ticket("account2", None) == promo.DEFAULT_TICKET_PRICE * promo.ACCOUNT2_PCT


def test_rank_prefers_account1_and_demand_and_urgency():
    parties = [
        _party(_id="a2-far", slug="a2-far", date="2026-09-14T22:00:00", goOutEventId="1"),
        _party(_id="a2-soon", slug="a2-soon", date="2026-09-09T22:00:00", goOutEventId="2"),
        _party(_id="a1", slug="a1", date="2026-09-14T22:00:00", goOutEventId="3"),
        _party(_id="past", slug="past", date="2026-09-01T22:00:00", goOutEventId="4"),
        _party(_id="too-far", slug="too-far", date="2026-10-30T22:00:00", goOutEventId="5"),
        _party(_id="no-url", slug="no-url", goOutUrl=None, goOutEventId="6"),
    ]
    ranked = promo.rank_promo_candidates(
        parties,
        sales_by_event_id={"1": {"totalTicketsSold": 3, "totalRevenue": 14.4}},
        accounts_by_event_id={"1": {"account2"}, "2": {"account2"}, "3": {"account1"}},
        now=NOW,
        days=7,
        limit=10,
    )
    ids = [c["partyId"] for c in ranked]
    assert "past" not in ids and "too-far" not in ids and "no-url" not in ids
    # account1 with zero sales (₪25 × 1 × 1.0 = 25) beats account2 with 3 recent
    # sales far out (₪4.8 × 4 × 1.0 = 19.2) and account2 soon (₪4.8 × 1 × 2.0 = 9.6)
    assert ids == ["a1", "a2-far", "a2-soon"]
    a1 = ranked[0]
    assert a1["tier"] == "account1" and a1["expectedPerTicket"] == 25.0 and a1["score"] == 25.0
    assert ranked[1]["ticketsLast30d"] == 3 and ranked[1]["revenueLast30d"] == 14.4
    assert ranked[2]["daysUntil"] < 2 and ranked[2]["score"] == 9.6
    assert a1["siteUrl"] == "https://www.parties247.co.il/event/a1"


def test_rank_respects_limit_and_skips_hidden():
    parties = [_party(_id=f"p{i}", slug=f"p{i}", goOutEventId=str(i)) for i in range(5)]
    parties[0]["hidden"] = True
    ranked = promo.rank_promo_candidates(
        parties, sales_by_event_id={}, accounts_by_event_id={}, now=NOW, days=7, limit=2,
    )
    assert len(ranked) == 2
    assert all(c["partyId"] != "p0" for c in ranked)


def test_message_formats():
    ranked = promo.rank_promo_candidates(
        [_party()], sales_by_event_id={}, accounts_by_event_id={}, now=NOW, days=7,
    )
    msg = ranked[0]["message"]
    assert msg.splitlines()[0] == "🎉 *THURSDAY MOON | MAINSTREAM*"
    assert "📅 יום חמישי 10.09 · 23:00" in msg
    assert "📍 Moon Child, תל אביב" in msg
    assert "🎶 מיינסטרים · 18+" in msg
    assert "💸 החל מ-₪80" in msg
    assert msg.endswith("https://www.go-out.co/event/moon?ref=acc2")

    digest = promo.format_digest_message(ranked, days=7)
    assert digest.startswith("🔥 *המסיבות הכי חמות לשבוע הקרוב* 🔥")
    assert "1️⃣ *THURSDAY MOON | MAINSTREAM* — יום חמישי 10.09 · 23:00, Moon Child, תל אביב" in digest
    assert digest.endswith("כל המסיבות: https://www.parties247.co.il")
    assert promo.format_digest_message([], days=7) == ""


def test_build_whatsapp_promo_joins_collections(monkeypatch):
    class Coll:
        def __init__(self, docs):
            self.docs = docs
            self.projections = []

        def find(self, *args, **kwargs):
            if len(args) > 1:
                self.projections.append(args[1])
            return list(self.docs)

        def aggregate(self, pipeline):
            return [{"_id": "111", "totalRevenue": 4.8, "totalTicketsSold": 1}]

    parties = Coll([
        {"_id": "p1", "slug": "s1", "name": "N1", "date": (datetime.now(ISRAEL) + timedelta(days=1)).strftime("%Y-%m-%dT22:00:00"),
         "goOutEventId": "111", "goOutUrl": "https://www.go-out.co/event/x"},
    ])
    sales = Coll([{"go_out_id": "111", "account_id": "account1"}])

    class Settings:
        def find_one(self, filter):
            return {"value": "myref"}

    monkeypatch.setattr(app, "parties_collection", parties)
    monkeypatch.setattr(app, "goout_sales_collection", sales)
    monkeypatch.setattr(app, "goout_sales_log_collection", sales)
    monkeypatch.setattr(app, "settings_collection", Settings())

    data = app.build_whatsapp_promo(days=7, limit=5)
    assert data["days"] == 7
    assert len(data["candidates"]) == 1
    c = data["candidates"][0]
    assert c["tier"] == "account1"
    assert c["ticketsLast30d"] == 1
    assert c["url"].endswith("ref=myref")  # default referral applied to the link
    assert "ref=myref" in data["digest"]
    # the party read must be projected, never a full-document load
    assert parties.projections and "description" not in parties.projections[0]
