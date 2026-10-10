"""Listing Guard rules (listings.py). Fixtures are real cases seen on the live
site on 2026-10-09, trimmed to the fields the rules read."""
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import listings


def tier(name, price, commission=0, display="ACTIVE"):
    return {"Title": name, "Price": price, "Commision": commission, "display": display}


def price_of(*raw_tiers, rules=None):
    return listings.compute_price_info(listings.parse_tiers(list(raw_tiers)), rules)


# --- price -----------------------------------------------------------------

def test_final_price_includes_commission_like_goouts_own_faq():
    # Collabo Thursday: Price 50, Commision 13.5 -> GoOut's FAQ says 56.75₪.
    assert listings.tier_final(50, 13.5) == 56.75


def test_event_page_placeholder_ticket_is_never_a_price_source():
    """The public page's Tickets array is the same fake 200+5% on every event;
    The XXX Project showed ₪210 because of it. parse_source must not read it."""
    event = {"Title": "The XXX Project", "Url": "1",
             "Tickets": [{"Title": "", "Price": 200, "Commision": 5, "Amount": "150", "sold": 0, "Active": True}]}
    source = listings.parse_source(event)
    assert "tiers" not in source
    info = listings.compute_price_info(source.get("tiers"), previous={"from": 22.0, "verified": True})
    assert info == {"from": 22.0, "verified": False}

    real = price_of(tier("GA ticket", 20, 10), tier("Door ticket", 25, 10, "COMING_SOON"))
    assert real["from"] == 22.0 and real["salesState"] == "on_sale"


def test_free_tier_next_to_a_paid_one_is_a_badge_not_the_price():
    # FRIDAY MAINSTREAM: site showed ₪0.
    info = price_of(tier("כניסה לכל הלילה ", 80, 9.25), tier("הרשמה חינם בהגעה עד 23:30", 0))
    assert info["from"] == 87.4
    assert info["hasFree"] is True and info["onlyFree"] is False
    assert info["freeLabel"] == "כניסה חופשית עד 23:30"
    assert listings.ticket_price_from(info) == 87.4


def test_only_free_event_is_free():
    info = price_of(tier("Free entry until 23:00", 0))
    assert info["onlyFree"] is True and info["from"] is None
    assert listings.ticket_price_from(info) == 0


def test_zero_priced_tier_is_not_assumed_free():
    info = price_of(tier("משלמים במקום", 0))
    assert info["hasFree"] is False and info["onlyFree"] is False
    assert info["unknownZeroTiers"] == ["משלמים במקום"]
    assert listings.ticket_price_from(info) is None


def test_remembered_answer_classifies_a_zero_tier():
    key = listings.zero_tier_key("כרטיסים אחרונים 17.10 🔥")
    assert key == listings.zero_tier_key("כרטיסים אחרונים 24.10")
    assert price_of(tier("כרטיסים אחרונים 24.10", 0), rules={key: "free"})["onlyFree"] is True
    ignored = price_of(tier("כרטיסים אחרונים 24.10", 0), rules={key: "ignore"})
    assert ignored["hasFree"] is False and ignored["unknownZeroTiers"] == []


def test_closed_and_sold_out_rounds_are_skipped():
    # BECO972: site showed the long-closed ₪55 first round.
    info = price_of(tier("First Lote", 55, 0, "NOT_ACTIVE"), tier("Second Lote", 88, 0, "NOT_ACTIVE"),
                    tier("Last lote", 95, 10))
    assert info["from"] == 104.5
    info = price_of(tier("סבב א", 50, 8, "SOLD_OUT"), tier("סבב ב", 70, 8))
    assert info["from"] == 75.6 and listings.sold_out_from(info) is False


def test_sold_out_only_when_goout_says_sold_out():
    assert listings.sold_out_from(price_of(tier("A", 50, 0, "SOLD_OUT"), tier("B", 70, 0, "NOT_ACTIVE"))) is True
    # Every tier NOT_ACTIVE is also what a far-future event looks like before
    # its sale opens (YARKON FIELDS PURIM, March 2027) — not sold out.
    closed = price_of(tier("Entry", 150, 0, "NOT_ACTIVE"))
    assert closed["salesState"] == "closed" and listings.sold_out_from(closed) is False
    assert listings.ticket_price_from(closed) is None


def test_coming_soon_price_is_shown_before_the_sale_opens():
    info = price_of(tier("רגיל", 50, 8, "COMING_SOON"), tier("חינם עד 23:30", 0, 0, "COMING_SOON"))
    assert info["salesState"] == "coming_soon" and info["from"] == 54.0 and info["hasFree"] is True


# --- fields, locks, status --------------------------------------------------

def _source(title, starts="2026-10-17T17:00:00.000", lat=32.07, lng=34.78, **extra):
    return {"title": title, "startsAt": starts, "geo": {"lat": lat, "lng": lng, "city": "Tel Aviv"},
            "address": "Somewhere 1, Tel Aviv", **extra}


def test_name_follows_goout_and_whitespace_alone_is_not_a_change():
    source = _source("ROOM 48 | 12.11 | MISHELL", starts="2026-11-12T23:00:00.000")
    party = {"name": " ROOM 48 | 15.10 | MISHELL", "date": "2026-11-12T23:00:00.000",
             "location": "Somewhere 1, Tel Aviv"}
    changes = listings.diff_fields(listings.derive_fields(source), party)
    assert changes["name"] == "ROOM 48 | 12.11 | MISHELL"
    assert "date" not in changes and "location" not in changes
    assert "name" not in listings.diff_fields(listings.derive_fields(source), {**party, "name": " ROOM 48 | 12.11 | MISHELL "})


def test_locked_fields_are_left_alone():
    source = _source("New Title")
    info = price_of(tier("A", 100))
    party = {"name": "My Title", "ticketPrice": 80, "locks": ["name", "ticketPrice"]}
    changes = listings.diff_fields(listings.derive_fields(source, info), party)
    assert "name" not in changes and "ticketPrice" not in changes
    assert changes["location"] == "Somewhere 1, Tel Aviv"


def test_unverified_price_never_overwrites_the_stored_one():
    info = listings.compute_price_info(None, previous={"from": 90.0, "verified": True})
    assert "ticketPrice" not in listings.derive_fields(_source("X"), info)


def test_admin_edit_locks_only_what_actually_changed():
    existing = {"name": "A", "location": "Old", "ticketPrice": 80.0, "date": "2026-10-17T17:00:00.000",
                "imageUrl": "https://images.go-out.co/events/x_coverImage.jpg"}
    # The admin form re-sends every field on every save.
    update = {"name": "A", "location": "New Club", "ticketPrice": 80, "date": "2026-10-17T17:00:00",
              "imageUrl": "https://images.go-out.co/x_coverImage.jpg", "referralCode": "abc"}
    assert listings.locks_for_admin_edit(update, existing) == ["location"]


def test_same_flyer_under_another_path_is_not_an_image_change():
    party = {"imageUrl": "https://images.go-out.co/events/abc123_coverImage.jpg"}
    assert listings.diff_fields({"imageUrl": "https://images.go-out.co/abc123_coverImage.jpg"}, party) == {}


def test_geo_beats_keywords_and_centroid_placeholder_is_no_location():
    # "Hilton Bayz" had region "לא ידוע" although GoOut supplied coordinates.
    assert listings.geo_classification({"geo": {"lat": 32.0937759, "lng": 34.7709858}}) == {
        "region": "מרכז", "areas": ["tel aviv"]}
    assert listings.geo_classification({"geo": {"lat": 32.70, "lng": 35.16}})["region"] == "צפון"  # Ramat Yishai
    assert listings.geo_classification({"geo": {"lat": 29.55, "lng": 34.95}})["areas"] == ["eilat", "south"]
    assert listings.geo_classification({"geo": {"lat": 52.36, "lng": 4.88}}) == {}  # Amsterdam
    assert listings.geo_classification({"geo": {"lat": 31.046051, "lng": 34.851612}}) == {}  # "Israël"


def test_private_events_are_hidden_and_come_back_when_public():
    private = {"publicity": "פרטי"}
    assert listings.desired_status(private, {}) == ("hidden", "private")
    assert listings.desired_status({"publicity": "פומבי"}, {"listingStatus": "hidden", "statusReason": "private"}) == ("live", None)
    # Never undo a hide somebody made by hand, never touch a merge.
    assert listings.desired_status({"publicity": "פומבי"}, {"listingStatus": "hidden", "statusReason": "admin"}) is None
    assert listings.desired_status(private, {"listingStatus": "merged"}) is None
    assert listings.desired_status(private, {"listingStatus": "live", "locks": ["listingStatus"]}) is None


# --- duplicates -------------------------------------------------------------

A1 = "acc1ref"


def party(pid, title, ref="acc2ref", event_id=None, **source):
    return {"_id": pid, "name": title, "referralCode": ref, "goOutEventId": event_id or f"e{pid}",
            "source": _source(title, **source)}


def test_same_event_on_both_accounts_keeps_account1_automatically():
    a = party("1", "WineNot? In The City 17.10🌅", organizerId="o1")
    b = party("2", "WineNot In The City 17.10🌅🍷", ref=A1, organizerId="o2")
    pairs = listings.find_duplicate_pairs([a, b])
    assert pairs[0]["level"] == "certain"
    plan = listings.plan_duplicates(pairs, A1)
    assert plan["issues"] == []
    assert plan["merges"][0]["keeperId"] == "2" and plan["merges"][0]["loserId"] == "1"


def test_account2_only_duplicate_is_asked_with_the_earning_listing_suggested():
    a = party("1", "WTGL: OFF SCRIPT 10.12", starts="2026-12-10T22:00:00.000")
    b = party("2", "WTGL: OFF SCRIPT 10.12", starts="2026-12-10T22:00:00.000")
    plan = listings.plan_duplicates(listings.find_duplicate_pairs([a, b]), A1, revenue={"2": 40.0})
    assert plan["merges"] == []
    assert plan["issues"][0]["type"] == "duplicate"
    assert plan["issues"][0]["suggestion"] == {"keeperId": "2", "why": "revenue"}


def test_same_flyer_same_hour_is_certain_even_with_different_titles():
    a = party("1", "MEMORIES ELECTION FESTIVAL // 26.10🗳️", blurhash="LKO2")
    b = party("2", "THE CHOICE OF MEMORIES // 26.10", blurhash="LKO2")
    assert listings.pair_level(a, b)["level"] == "certain"


def test_same_venue_same_hour_different_promoters_is_asked_not_merged():
    # Bustan: "מיינסטרים בקבוע" (account1) vs "עד הדכא" (account2), same door, same hour.
    a = party("1", "עד הדכא • בר חופשי • אוויר הפתוח • 9.10🍺", organizerId="o1")
    b = party("2", "מיינסטרים בקבוע • בר חופשי // 09.10🍺", ref=A1, organizerId="o2")
    pairs = listings.find_duplicate_pairs([a, b])
    assert pairs[0]["level"] == "possible"
    plan = listings.plan_duplicates(pairs, A1)
    assert plan["merges"] == []
    assert plan["issues"][0]["suggestion"]["why"] == "account1"


def test_remembered_series_answer_decides_the_next_week_without_asking():
    a = party("1", "FRIDAY MAINSTREAM | 16.10", organizerId="waves")
    b = party("2", "SHTUBY x CLUB DE COMBAT • FRIDAY OPEN AIR", organizerId="waves")
    key = listings.series_rule_key(a, b)
    next_a = party("3", "FRIDAY MAINSTREAM | 23.10", organizerId="waves", starts="2026-10-24T17:00:00.000")
    next_b = party("4", "DARWISH x HELLO VERA • FRIDAY OPEN AIR", organizerId="waves", starts="2026-10-24T17:00:00.000")
    assert listings.series_rule_key(next_a, next_b) == key  # titles change weekly, the key doesn't
    pairs = listings.find_duplicate_pairs([next_a, next_b])
    assert listings.plan_duplicates(pairs, A1, series_rules={key: "different"}) == {"merges": [], "issues": []}
    same = listings.plan_duplicates(pairs, A1, series_rules={key: "same"})
    assert len(same["merges"]) == 1 and same["issues"] == []


def test_answered_pair_is_not_asked_again():
    a = party("1", "DECISION FESTIVAL | GAGARIN TLV | 15.10")
    b = party("2", "COLORFUL FESTIVAL | GAGARIN TLV | 15.10")
    pairs = listings.find_duplicate_pairs([a, b])
    assert len(listings.plan_duplicates(pairs, A1)["issues"]) == 1
    assert listings.plan_duplicates(pairs, A1, decided={listings.pair_fingerprint(a, b)})["issues"] == []


def test_different_weeks_of_a_series_are_not_duplicates():
    a = party("1", "THURSDAY MOON | MAINSTREAM | 15.10", starts="2026-10-15T23:00:00.000")
    b = party("2", "THURSDAY MOON | MAINSTREAM | 22.10", starts="2026-10-22T23:00:00.000")
    assert listings.find_duplicate_pairs([a, b]) == []


def test_venue_less_events_do_not_all_match_each_other():
    centroid = {"lat": 31.046051, "lng": 34.851612}
    a = party("1", "PURIM OF MEMORIES // 22.3", **centroid)
    b = party("2", "HATER'S FEST", **centroid)
    assert listings.find_duplicate_pairs([a, b]) == []


def test_admin_clones_are_never_merged_back_into_their_source():
    original = party("1", "Big Party", event_id="777")
    clone = {**party("2", "Big Party", event_id="777"), "cloneOf": "1"}
    legacy_clone = {**party("3", "Big Party", event_id="777"),
                    "originalUrl": "https://go-out.co/event/1?clone_ref=1759000000"}
    assert listings.find_duplicate_pairs([original, clone, legacy_clone]) == []
    plain = party("4", "Big Party", event_id="777")
    plan = listings.plan_duplicates(listings.find_duplicate_pairs([original, plain]))
    assert plan["merges"][0]["reason"].startswith("identical")


def test_one_party_is_never_merged_twice_in_one_run():
    a = party("1", "Same Party", ref=A1)
    b = party("2", "Same Party")
    c = party("3", "Same Party")
    plan = listings.plan_duplicates(listings.find_duplicate_pairs([a, b, c]), A1)
    assert sorted(m["loserId"] for m in plan["merges"]) == ["2", "3"]
    assert {m["keeperId"] for m in plan["merges"]} == {"1"}


# --- detectors --------------------------------------------------------------

def test_title_date_mismatch():
    def mismatch(title, starts, ends=None):
        return listings.title_date_mismatch({"title": title, "startsAt": starts, "endsAt": ends})

    assert mismatch("MEMORIES HANUKKAH FESTIVAL // 6.12🕎", "2026-12-09T22:00:00.000") is True
    assert mismatch("THURSDAY MOON | MAINSTREAM | 15.10", "2026-10-15T23:00:00.000") is False
    # After-midnight start is advertised under the night before.
    assert mismatch("Collabo Thursday 8.10", "2026-10-09T01:00:00.000") is False
    # Multi-day festival advertised as a range.
    assert mismatch("Revival Summer Festival 13-14.8", "2026-08-13T16:00:00.000", "2026-08-15T06:00:00.000") is False
    assert mismatch("Free until 23:30 | 18+", "2026-10-15T23:00:00.000") is False
    assert mismatch("Winenot in the city 17/10", "2026-10-17T17:00:00.000") is False


def test_party_issue_fingerprints_are_stable_and_zero_tier_is_global():
    info = price_of(tier("כרטיסים אחרונים", 0))
    one = {"_id": "1", "goOutEventId": "100", "source": _source("A 17.10"), "priceInfo": info}
    two = {"_id": "2", "goOutEventId": "200", "source": _source("B 17.10"), "priceInfo": info}
    first = [i["fingerprint"] for i in listings.detect_party_issues(one)]
    assert first == [i["fingerprint"] for i in listings.detect_party_issues(two)] == ["zero:כרטיסים אחרונים"]


def test_vague_location_is_flagged_unless_the_admin_set_one():
    vague = {"_id": "1", "location": "Israël", "source": {"title": "X", "geo": {"lat": 31.046051, "lng": 34.851612}}}
    assert [i["type"] for i in listings.detect_party_issues(vague)] == ["location_vague"]
    assert listings.detect_party_issues({**vague, "location": "Kibbutz Y", "locks": ["location"]}) == []


def test_gone_stale_and_unverified_sources_are_flagged():
    now = datetime(2026, 10, 9, 12)
    gone = {"_id": "1", "source": {**_source("X 17.10"), "failCount": 2, "pageStatus": 404}}
    assert [i["type"] for i in listings.detect_party_issues(gone, now)] == ["source_gone"]
    stale = {"_id": "2", "source": {**_source("X 17.10"), "fetchedAt": "2026-10-07T12:00:00"}}
    assert [i["type"] for i in listings.detect_party_issues(stale, now)] == ["stale_sync"]
    unverified = {"_id": "3", "priceInfo": {"verified": False},
                  "source": {**_source("X 17.10"), "fetchedAt": "2026-10-09T11:00:00", "tiersFailCount": 3}}
    assert [i["type"] for i in listings.detect_party_issues(unverified, now)] == ["price_unverified"]
    assert [i["type"] for i in listings.detect_party_issues({"_id": "4", "name": "never"}, now)] == ["stale_sync"]


def test_site_render_check_compares_the_live_page_with_the_database():
    listed = {"_id": "1", "name": "Big Party 17.10 🎉", "ticketPrice": 87.4}
    ok = {"status": 200, "name": "Big Party 17.10", "price": "87.4", "finalUrl": "https://www.parties247.co.il/event/big"}
    assert listings.detect_site_issues(listed, ok) == []
    stale = listings.detect_site_issues(listed, {**ok, "price": "0"})
    assert stale[0]["evidence"] == {"price": {"site": 0.0, "db": 87.4}}
    assert listings.detect_site_issues(listed, {"status": 404})[0]["evidence"] == {"status": 404}


# --- cases carried over from the scraper's retired dedupe_parties.py ---------

def test_same_venue_same_hour_under_two_names_is_asked_not_merged():
    a = party("1", "Organizer A presents: X", starts="2026-10-10T23:00:00.000")
    b = party("2", "Completely Different Branding", starts="2026-10-10T23:45:00.000")
    assert listings.pair_level(a, b)["level"] == "possible"
    plan = listings.plan_duplicates(listings.find_duplicate_pairs([a, b]), A1)
    assert plan["merges"] == [] and len(plan["issues"]) == 1


def test_same_venue_hours_apart_is_not_a_duplicate():
    a = party("1", "Party A", starts="2026-10-10T18:00:00.000")
    b = party("2", "Other Event", starts="2026-10-10T23:30:00.000")
    assert listings.pair_level(a, b) is None


def test_same_brand_in_another_city_on_another_night_is_not_a_duplicate():
    a = party("1", "THURSDAY MOON | MAINSTREAM | 06.10", starts="2026-10-10T23:00:00.000")
    b = party("2", "THURSDAY MOON | MAINSTREAM | 6.10", starts="2026-10-10T02:30:00.000", lat=31.77, lng=35.21)
    assert listings.pair_level(a, b) is None


def test_test_events_and_absurd_prices_are_asked_about():
    test_event = {"_id": "1", "goOutEventId": "100", "source": _source("TEST EVENT 123")}
    assert [i["fingerprint"] for i in listings.detect_party_issues(test_event)] == ["test:100"]
    pricey = {"_id": "2", "goOutEventId": "200", "source": _source("Gala 17.10"), "ticketPrice": 2500}
    assert [i["type"] for i in listings.detect_party_issues(pricey)] == ["price_suspicious"]
    assert listings.detect_party_issues({**pricey, "ticketPrice": 450}) == []
