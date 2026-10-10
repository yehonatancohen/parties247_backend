"""Listing Guard — the single set of rules for what a party listing says.

Pure logic only (no Flask, no Mongo, no network), same split as promo.py /
wa_facts.py: app.py owns the routes and the writes, this module owns every
decision. Everything that used to be spread across scheduled_price_scan,
scheduled_content_refresh, goout-scraper's dedupe_parties.py and three
separate price parsers now lives here:

  parse_source()        raw GoOut event + ticket tiers -> stored `source` snapshot
  compute_price_info()  ticket tiers -> the one price the site shows
  derive_fields()       snapshot -> display fields (name, date, location, ...)
  diff_fields()         what actually changes, honouring admin locks
  find_duplicate_pairs() / plan_duplicates()   who is the same party as whom
  detect_party_issues() everything a human has to look at

Why tiers and not the event page: the `Tickets` array inside GoOut's public
__NEXT_DATA__ is an identical placeholder on every event (Price 200,
Commision 5, Amount 150 — verified on all 103 upcoming events, 2026-10-09).
Real tiers only come from GET www.go-out.co/endOne/loadEventTickets. Nothing
in here may ever read event["Tickets"].
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta

# GoOut stores this exact point for events whose address is just "Israel" —
# the country centroid. It is "no location", not a venue: treating it as real
# would put every venue-less event 0 m away from every other one.
PLACEHOLDER_GEO = (31.046051, 34.851612)

PRIVATE_PUBLICITY = "פרטי"

# Display fields the sync may overwrite, and therefore the ones an admin edit
# can lock. Anything not listed here is never touched by the sync.
LOCKABLE_FIELDS = (
    "name", "date", "location", "imageUrl", "description",
    "ticketPrice", "soldOut", "region", "areas",
)

HIDDEN_STATUSES = ("hidden", "merged")

_FREE_KEYWORDS = ("חינם", "free", "חופשי", "חופשית", "ללא עלות", "ללא תשלום")
_TIME_RE = re.compile(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)")

_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"
    "\U00002190-\U000021FF"
    "\U00002300-\U000027BF"
    "\U00002B00-\U00002BFF"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"
    "]+",
    flags=re.UNICODE,
)
_DATE_RANGE_RE = re.compile(r"(?<!\d)(\d{1,2})\s*-\s*(\d{1,2})[./](\d{1,2})(?![\d:])")
_DATE_TOKEN_RE = re.compile(r"(?<![\d.:/])(\d{1,2})[./](\d{1,2})(?:[./]\d{2,4})?(?![\d:])")
_NON_WORD_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)


# --- small helpers ---------------------------------------------------------

def _num(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def parse_local(value) -> datetime | None:
    """GoOut dates are naive Israel wall-clock strings ("2026-10-09T01:00:00.000").
    Keep them naive: every comparison here is between two GoOut dates."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().rstrip("Z")
    try:
        return datetime.fromisoformat(text).replace(tzinfo=None)
    except ValueError:
        try:
            return datetime.strptime(text[:10], "%Y-%m-%d")
        except ValueError:
            return None


def is_hidden(party: dict) -> bool:
    return (party or {}).get("listingStatus") in HIDDEN_STATUSES


def is_clone(party: dict) -> bool:
    """Admin "clone as promotion" copies are deliberate second listings of the
    same GoOut event. clone_party() stamps cloneOf; older clones are only
    recognisable by the cache-busting marker the admin UI put in the URL."""
    if (party or {}).get("cloneOf"):
        return True
    url = str((party or {}).get("originalUrl") or "")
    return "clone_ref=" in url or "#clone-" in url


# --- source snapshot -------------------------------------------------------

def tier_final(price, commission) -> float | None:
    """Final buyer price. GoOut's own FAQ line is exactly this: 50 + 13.5% = 56.75."""
    base = _num(price)
    if base is None or base < 0:
        return None
    return round(base * (1 + (_num(commission) or 0.0) / 100), 2)


def parse_tiers(raw_tiers) -> list[dict] | None:
    """loadEventTickets `tickets` -> compact tiers. None means "we don't know"
    (fetch failed), which is different from an event that has no tiers."""
    if not isinstance(raw_tiers, list):
        return None
    tiers = []
    for raw in raw_tiers:
        if not isinstance(raw, dict):
            continue
        price = _num(raw.get("Price"))
        if price is None:
            continue
        commission = _num(raw.get("Commision")) or 0.0  # GoOut's own misspelling
        tiers.append({
            "name": _text(raw.get("Title")),
            "price": price,
            "commission": round(commission, 4),
            "final": tier_final(price, commission),
            "display": str(raw.get("display") or ("ACTIVE" if raw.get("Active", True) else "NOT_ACTIVE")),
        })
    return tiers


def pick_image(event: dict, og_image: str | None = None) -> str | None:
    if og_image:
        return og_image.replace("_whatsappImage.jpg", "_coverImage.jpg")
    schema = event.get("schemaOrg")
    for item in (schema if isinstance(schema, list) else [schema]):
        if isinstance(item, dict) and item.get("@type") == "Event":
            images = item.get("image")
            if isinstance(images, str):
                images = [images]
            if images and images[0]:
                return str(images[0])
    return None


def has_real_geo(source: dict | None) -> bool:
    geo = (source or {}).get("geo") or {}
    lat, lng = _num(geo.get("lat")), _num(geo.get("lng"))
    if lat is None or lng is None:
        return False
    return not (abs(lat - PLACEHOLDER_GEO[0]) < 1e-4 and abs(lng - PLACEHOLDER_GEO[1]) < 1e-4)


def parse_source(event: dict, tiers=None, og_image: str | None = None, fetched_at: str | None = None) -> dict:
    """Trim a GoOut `pageProps.event` (+ optional loadEventTickets tiers) down
    to the snapshot stored on the party as `source`. `tiers` omitted/None keeps
    the key out, so a page-only refresh never erases known tiers."""
    location = event.get("Location") if isinstance(event.get("Location"), dict) else {}
    # Stored already shortened: the raw text runs to several KB per party and
    # this snapshot is read on the slow Render<->Atlas link.
    description = clean_description(str(event.get("Description") or ""))
    source = {
        "title": _text(event.get("Title")),
        "startsAt": event.get("StartingDate") or None,
        "endsAt": event.get("EndingDate") or None,
        "address": _text(event.get("Adress")),  # GoOut's own misspelling
        "englishAddress": _text(event.get("EnglishAddress")),
        "geo": {
            "lat": _num(location.get("lat")),
            "lng": _num(location.get("lng")),
            "city": _text(location.get("city")),
        },
        "publicity": _text(event.get("EventPublicity")),
        "organizerId": str(event.get("OrganizerID") or "") or None,
        "creatorId": str(event.get("creatorId") or "") or None,
        "producersName": _text(event.get("ProducersName")),
        "blurhash": event.get("Blurhash") or None,
        "eventSerial": str(event.get("EventSerial") or event.get("eventSerial") or "") or None,
        "goOutUrlId": str(event.get("Url") or "") or None,
        "minimumAge": event.get("MinimumAge"),
        "image": pick_image(event, og_image),
        "description": description,
        "fetchedAt": fetched_at,
    }
    parsed = parse_tiers(tiers)
    if parsed is not None:
        source["tiers"] = parsed
        source["tiersFetchedAt"] = fetched_at
    return source


# --- price -----------------------------------------------------------------

def zero_tier_key(name: str) -> str:
    """Stable key for a ₪0 tier's name — what a remembered free/not-free answer
    is stored under, so one answer covers every week of a recurring party."""
    name = _EMOJI_RE.sub(" ", name or "")
    name = _DATE_TOKEN_RE.sub(" ", name)
    name = _NON_WORD_RE.sub(" ", name)
    return re.sub(r"\s+", " ", name).strip().lower()


def classify_zero_tier(name: str, zero_tier_rules: dict | None = None) -> str:
    """'free' | 'ignore' | 'unknown' for a tier priced ₪0. A ₪0 tier is often
    not free entry at all ("משלמים במקום", "שמירת שולחן") — only an explicit
    free keyword or a remembered answer makes it free."""
    rule = (zero_tier_rules or {}).get(zero_tier_key(name))
    if rule in ("free", "ignore"):
        return rule
    lowered = (name or "").lower()
    if any(keyword in lowered for keyword in _FREE_KEYWORDS):
        return "free"
    return "unknown"


def free_label(name: str) -> str:
    match = _TIME_RE.search(name or "")
    if match:
        return f"כניסה חופשית עד {int(match.group(1)):02d}:{match.group(2)}"
    return "כניסה חופשית בהרשמה מראש"


def compute_price_info(tiers: list[dict] | None, zero_tier_rules: dict | None = None,
                       previous: dict | None = None) -> dict:
    """The one price rule (owner decision 2026-10-09): cheapest currently
    buyable PAID tier, final incl. commission; a free tier next to it becomes a
    badge, not the price. Trust GoOut's `display`, not SaleEnd — tiers stay
    ACTIVE long past their nominal SaleEnd."""
    previous = previous or {}
    if tiers is None:
        return {**previous, "verified": False}
    if not tiers:
        return {"from": None, "hasFree": False, "freeLabel": None, "onlyFree": False,
                "salesState": "closed", "verified": True, "unknownZeroTiers": []}

    displays = {tier.get("display") for tier in tiers}
    if "ACTIVE" in displays:
        state, buyable = "on_sale", [t for t in tiers if t.get("display") == "ACTIVE"]
    elif "COMING_SOON" in displays:
        state, buyable = "coming_soon", [t for t in tiers if t.get("display") == "COMING_SOON"]
    elif "SOLD_OUT" in displays:
        state, buyable = "sold_out", []
    else:
        state, buyable = "closed", []

    paid = [t["final"] for t in buyable if (t.get("price") or 0) > 0 and t.get("final") is not None]
    free_names, unknown = [], []
    for tier in buyable:
        if (tier.get("price") or 0) > 0:
            continue
        kind = classify_zero_tier(tier.get("name", ""), zero_tier_rules)
        if kind == "free":
            free_names.append(tier.get("name", ""))
        elif kind == "unknown":
            unknown.append(tier.get("name", ""))

    has_free = bool(free_names)
    return {
        "from": min(paid) if paid else None,
        "hasFree": has_free,
        "freeLabel": free_label(free_names[0]) if has_free else None,
        "onlyFree": has_free and not paid,
        "salesState": state,
        "verified": True,
        "unknownZeroTiers": unknown,
    }


def ticket_price_from(price_info: dict):
    """Legacy `ticketPrice` (website, promo.py, WhatsApp engine all read it):
    the paid price, 0 only when the event really is free-only."""
    if price_info.get("from") is not None:
        return price_info["from"]
    return 0 if price_info.get("onlyFree") else None


def sold_out_from(price_info: dict) -> bool:
    """Only a real SOLD_OUT. "closed" (every tier NOT_ACTIVE) is also what an
    event months away looks like before its sale opens — seen 2026-10-09 on a
    March 2027 listing — so it must not read as sold out."""
    return price_info.get("salesState") == "sold_out"


# --- geography -------------------------------------------------------------

def distance_m(a: dict, b: dict) -> float | None:
    """Metres between two sources' venues; None unless both have a real location."""
    if not (has_real_geo(a) and has_real_geo(b)):
        return None
    lat1, lng1 = float(a["geo"]["lat"]), float(a["geo"]["lng"])
    lat2, lng2 = float(b["geo"]["lat"]), float(b["geo"]["lng"])
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lng2 - lng1) / 2) ** 2)
    return 2 * 6371000 * math.asin(math.sqrt(h))


def _km_from(lat: float, lng: float, point: tuple[float, float]) -> float:
    fake = {"geo": {"lat": point[0], "lng": point[1]}}
    return (distance_m({"geo": {"lat": lat, "lng": lng}}, fake) or 0.0) / 1000


def geo_classification(source: dict | None) -> dict:
    """Region + areas from GoOut's coordinates. Returns {} when there is no
    real location or it is outside Israel, so callers fall back to the old
    keyword matching on the address text."""
    if not has_real_geo(source):
        return {}
    lat, lng = float(source["geo"]["lat"]), float(source["geo"]["lng"])
    if not (29.4 <= lat <= 33.4 and 34.2 <= lng <= 35.95):
        return {}
    areas: list[str] = []
    if lat < 29.75:
        region, areas = "דרום", ["eilat", "south"]
    elif lat < 31.4 or (lat < 31.85 and lng < 34.95):
        region, areas = "דרום", ["south"]
    elif lat > 32.4:
        region, areas = "צפון", ["north"]
        if _km_from(lat, lng, (32.80, 35.00)) <= 15:
            areas = ["haifa", "north"]
    else:
        region = "מרכז"
        if _km_from(lat, lng, (32.08, 34.80)) <= 12:
            areas = ["tel aviv"]
        elif _km_from(lat, lng, (31.78, 35.21)) <= 12:
            areas = ["jerusalem"]
    return {"region": region, "areas": areas}


# --- display fields --------------------------------------------------------

def clean_description(description: str) -> str:
    cleaned = " ".join(list(filter(None, (description or "").split("\n")))[:3]).strip()
    if len(cleaned) > 250:
        cleaned = cleaned[:247] + "..."
    return cleaned


def derive_fields(source: dict, price_info: dict | None = None) -> dict:
    """Display fields as GoOut currently states them. Empty values are left
    out entirely — a hole in a scrape must never blank good stored data."""
    fields: dict = {}
    if source.get("title"):
        fields["name"] = source["title"]
    if source.get("startsAt"):
        fields["date"] = source["startsAt"]
    if source.get("address"):
        fields["location"] = source["address"]
    if source.get("image"):
        fields["imageUrl"] = source["image"]
    description = clean_description(source.get("description") or "")
    if description:
        fields["description"] = description
    if source.get("eventSerial"):
        fields["goOutEventId"] = source["eventSerial"]
    geo = geo_classification(source)
    if geo.get("region"):
        fields["region"] = geo["region"]
    if geo.get("areas"):
        fields["areas"] = geo["areas"]
    if price_info is not None and price_info.get("verified"):
        fields["ticketPrice"] = ticket_price_from(price_info)
        fields["soldOut"] = sold_out_from(price_info)
    return fields


def desired_status(source: dict, party: dict) -> tuple[str, str | None] | None:
    """Private GoOut events are never listed (owner decision). Returns the
    (status, reason) the party should move to, or None for "leave it"; only
    ever undoes a hide that this same rule made."""
    if "listingStatus" in (party.get("locks") or []):
        return None  # the admin decided this one by hand
    status = party.get("listingStatus") or "live"
    private = source.get("publicity") == PRIVATE_PUBLICITY
    if private and status == "live":
        return "hidden", "private"
    if not private and status == "hidden" and party.get("statusReason") == "private":
        return "live", None
    return None


def _same(field: str, old, new) -> bool:
    if field == "areas":
        return sorted(old or []) == sorted(new or [])
    if field == "date":
        old_dt, new_dt = parse_local(old), parse_local(new)
        if old_dt and new_dt:
            return old_dt == new_dt
    if field == "ticketPrice" and isinstance(old, (int, float)) and isinstance(new, (int, float)):
        return abs(old - new) < 0.005
    if field == "soldOut":
        return bool(old) == bool(new)
    if field == "imageUrl" and old and new:
        # GoOut serves one flyer under several path prefixes; same file = same image.
        return str(old).rsplit("/", 1)[-1] == str(new).rsplit("/", 1)[-1]
    if field == "name":
        return _text(old) == _text(new)
    return old == new


def diff_fields(desired: dict, party: dict) -> dict:
    """{field: new_value} for what would really change, skipping locked fields."""
    locks = set(party.get("locks") or [])
    changes = {}
    for field, new in desired.items():
        if field in locks:
            continue
        if not _same(field, party.get(field), new):
            changes[field] = new
    return changes


def locks_for_admin_edit(update: dict, existing: dict) -> list[str]:
    """Fields an admin edit should lock: only ones whose value really changed.
    The admin form PUTs the whole party on every save, so "the field was sent"
    would lock everything on the first edit."""
    return [
        field for field in LOCKABLE_FIELDS
        if field in update and not _same(field, existing.get(field), update[field])
    ]


# --- duplicates ------------------------------------------------------------

def norm_title(title: str) -> str:
    """Title with emoji, punctuation and embedded dates removed, so
    "WineNot? In The City 17.10🌅" == "WineNot In The City 17.10🌅🍷"."""
    text = _EMOJI_RE.sub(" ", title or "")
    text = _DATE_RANGE_RE.sub(" ", text)
    text = _DATE_TOKEN_RE.sub(" ", text)
    text = _NON_WORD_RE.sub(" ", text.replace("_", " "))
    return re.sub(r"\s+", " ", text).strip().lower()


def title_similarity(a: str, b: str) -> float:
    ta = {w for w in norm_title(a).split() if len(w) > 1}
    tb = {w for w in norm_title(b).split() if len(w) > 1}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def venue_key(source: dict | None) -> str | None:
    if not has_real_geo(source):
        return None
    return f"{float(source['geo']['lat']):.3f},{float(source['geo']['lng']):.3f}"


def _promoter(source: dict) -> str:
    return str(source.get("organizerId") or source.get("creatorId") or source.get("producersName") or "?")


def series_rule_key(a: dict, b: dict) -> str | None:
    """Key a remembered duplicate answer is stored under: this venue + this
    pair of promoters. Titles are deliberately not part of it — weekly lines
    change their title (date, lineup) every week, the venue and promoters don't."""
    sa, sb = a.get("source") or {}, b.get("source") or {}
    venue = venue_key(sa) or venue_key(sb)
    if not venue:
        return None
    first, second = sorted([_promoter(sa), _promoter(sb)])
    return f"{venue}|{first}|{second}"


def _event_ref(party: dict) -> str:
    return str(party.get("goOutEventId") or (party.get("source") or {}).get("eventSerial") or party.get("_id"))


def pair_fingerprint(a: dict, b: dict) -> str:
    first, second = sorted([_event_ref(a), _event_ref(b)])
    return f"dup:{first}:{second}"


def pair_level(a: dict, b: dict) -> dict | None:
    """How sure we are that two listings are the same real-world party.
    Compares GoOut's live data (`source`), never our stored name — the stored
    name is exactly what used to go stale and blind the old dedupe."""
    if is_clone(a) or is_clone(b):
        return None
    sa, sb = a.get("source") or {}, b.get("source") or {}

    id_a, id_b = a.get("goOutEventId"), b.get("goOutEventId")
    if id_a and id_b and str(id_a) == str(id_b):
        return {"level": "identical", "hours": 0.0, "distance": 0.0, "similarity": 1.0, "sameImage": True}

    start_a, start_b = parse_local(sa.get("startsAt")), parse_local(sb.get("startsAt"))
    if not start_a or not start_b:
        return None
    hours = abs((start_a - start_b).total_seconds()) / 3600
    if hours > 3:
        return None

    distance = distance_m(sa, sb)
    near = distance is None or distance <= 2000
    title_a, title_b = norm_title(sa.get("title", "")), norm_title(sb.get("title", ""))
    similarity = title_similarity(sa.get("title", ""), sb.get("title", ""))
    same_image = bool(sa.get("blurhash")) and sa.get("blurhash") == sb.get("blurhash")
    info = {
        "hours": round(hours, 2),
        "distance": None if distance is None else round(distance),
        "similarity": round(similarity, 2),
        "sameImage": same_image,
    }

    same_title = bool(title_a) and len(title_a) >= 4 and title_a == title_b
    if hours <= 0.5 and near and (same_title or same_image):
        return {"level": "certain", **info}
    # Same venue at the same hour: possibly one party sold under two names by
    # two promoters (or by one promoter under two brands). Only a human knows.
    if hours <= 1.0 and distance is not None and distance <= 200:
        return {"level": "possible", **info}
    if near and similarity >= 0.6:
        return {"level": "possible", **info}
    return None


def find_duplicate_pairs(parties: list[dict]) -> list[dict]:
    """All suspicious pairs among the given (live, upcoming) parties."""
    pairs = []
    for i, a in enumerate(parties):
        for b in parties[i + 1:]:
            match = pair_level(a, b)
            if match:
                pairs.append({"a": a, "b": b, **match})
    return pairs


def _pick_keeper(a: dict, b: dict, account1_referral: str | None, revenue: dict) -> tuple[dict, dict, str]:
    """(keeper, loser, why). account1 always wins (priority account); then the
    listing that actually earned us commission; then the older document."""
    a1 = bool(account1_referral) and a.get("referralCode") == account1_referral
    b1 = bool(account1_referral) and b.get("referralCode") == account1_referral
    if a1 != b1:
        return (a, b, "account1") if a1 else (b, a, "account1")
    rev_a, rev_b = revenue.get(str(a.get("_id")), 0.0), revenue.get(str(b.get("_id")), 0.0)
    if rev_a != rev_b:
        return (a, b, "revenue") if rev_a > rev_b else (b, a, "revenue")
    return (a, b, "older") if str(a.get("_id")) <= str(b.get("_id")) else (b, a, "older")


def plan_duplicates(pairs: list[dict], account1_referral: str | None = None, revenue: dict | None = None,
                    series_rules: dict | None = None, decided: set | None = None) -> dict:
    """Turn suspicious pairs into {"merges": [...], "issues": [...]}.

    Owner rule (2026-10-09): when one side is account1, account1 is kept
    automatically; account2-only groups are asked. Low-confidence ("possible")
    pairs are always asked the first time — that is the "same party, different
    promoter, different name" case — and a "remember for this series" answer
    (series_rules) decides every later week without asking again.
    """
    revenue = revenue or {}
    series_rules = series_rules or {}
    decided = decided or set()
    merges, issues, gone = [], [], set()

    order = {"identical": 0, "certain": 1, "possible": 2}
    for pair in sorted(pairs, key=lambda p: order[p["level"]]):
        a, b = pair["a"], pair["b"]
        if str(a.get("_id")) in gone or str(b.get("_id")) in gone:
            continue
        fingerprint = pair_fingerprint(a, b)
        keeper, loser, why = _pick_keeper(a, b, account1_referral, revenue)
        evidence = {k: pair[k] for k in ("level", "hours", "distance", "similarity", "sameImage")}

        def merge(reason: str, keeper=keeper, loser=loser):
            merges.append({"keeperId": str(keeper["_id"]), "loserId": str(loser["_id"]),
                           "reason": reason, "fingerprint": fingerprint, "evidence": evidence})
            gone.add(str(loser["_id"]))

        if pair["level"] == "identical":
            merge(f"identical:{why}")
            continue
        if pair["level"] == "certain" and why == "account1":
            merge("certain:account1")
            continue
        if fingerprint in decided:
            continue
        if pair["level"] == "possible":
            rule = series_rules.get(series_rule_key(a, b) or "")
            if rule == "different":
                continue
            if rule == "same":
                merge(f"series_rule:{why}")
                continue
        issues.append({
            "type": "duplicate",
            "fingerprint": fingerprint,
            "partyIds": [str(a["_id"]), str(b["_id"])],
            "summary": f"{(a.get('source') or {}).get('title') or a.get('name')} ↔ "
                       f"{(b.get('source') or {}).get('title') or b.get('name')}",
            "evidence": {**evidence, "seriesKey": series_rule_key(a, b)},
            "suggestion": {"keeperId": str(keeper["_id"]), "why": why},
        })
    return {"merges": merges, "issues": issues}


# --- per-party detectors ---------------------------------------------------

def title_dates(title: str) -> list[tuple[int, int]]:
    """(day, month) pairs written in a title: "17.10", "6/12", "13-14.8"."""
    found = []
    for match in _DATE_RANGE_RE.finditer(title or ""):
        month = int(match.group(3))
        found += [(int(match.group(1)), month), (int(match.group(2)), month)]
    for match in _DATE_TOKEN_RE.finditer(title or ""):
        found.append((int(match.group(1)), int(match.group(2))))
    return [(d, m) for d, m in found if 1 <= d <= 31 and 1 <= m <= 12]


def title_date_mismatch(source: dict) -> bool:
    """The title advertises a date the event doesn't start on. A party that
    starts after midnight is normally advertised under the night before, and a
    multi-day event under any of its days."""
    start = parse_local(source.get("startsAt"))
    dates = title_dates(source.get("title", ""))
    if not start or not dates:
        return False
    end = parse_local(source.get("endsAt")) or start
    allowed = set()
    day = (start - timedelta(days=1)) if start.hour < 7 else start
    while day.date() <= max(end, start).date() and len(allowed) < 10:
        allowed.add((day.day, day.month))
        day += timedelta(days=1)
    allowed.add((start.day, start.month))
    return not any(d in allowed for d in dates)


_VAGUE_LOCATIONS = {"", "ישראל", "israel", "israël", "unknown location"}


def location_is_vague(party: dict) -> bool:
    if "location" in (party.get("locks") or []):
        return False
    source = party.get("source") or {}
    if has_real_geo(source):
        return False
    return _text(party.get("location")).lower() in _VAGUE_LOCATIONS


def detect_party_issues(party: dict, now: datetime | None = None) -> list[dict]:
    """Issues for one live, upcoming party that need a person. Each fingerprint
    is stable, so a question answered once is never asked again."""
    pid = str(party.get("_id"))
    source = party.get("source") or {}
    price_info = party.get("priceInfo") or {}
    name = source.get("title") or party.get("name") or pid
    ref = _event_ref(party)
    issues = []

    def add(kind: str, fingerprint: str, summary: str, evidence: dict | None = None):
        issues.append({"type": kind, "fingerprint": fingerprint, "partyIds": [pid],
                       "summary": summary, "evidence": evidence or {}})

    if not source:
        add("stale_sync", f"stale:{ref}", f"{name}: never synced with GoOut")
        return issues

    if (source.get("failCount") or 0) >= 2:
        add("source_gone", f"gone:{ref}", f"{name}: GoOut page is gone",
            {"failCount": source.get("failCount"), "pageStatus": source.get("pageStatus")})
    elif now is not None:
        fetched = parse_local(source.get("fetchedAt"))
        if fetched and now - fetched > timedelta(hours=24):
            add("stale_sync", f"stale:{ref}", f"{name}: not synced for over 24h",
                {"fetchedAt": source.get("fetchedAt")})

    if price_info.get("verified") is False and (source.get("tiersFailCount") or 0) >= 3:
        add("price_unverified", f"price:{ref}", f"{name}: ticket tiers unreachable, price not verified")

    for tier_name in price_info.get("unknownZeroTiers") or []:
        key = zero_tier_key(tier_name)
        if key:
            # Global on purpose: the answer is about the tier's wording, not this party.
            add("zero_tier", f"zero:{key}", f"₪0 ticket \"{tier_name}\" — is this free entry?",
                {"tierName": tier_name, "tierKey": key})

    if title_date_mismatch(source):
        add("title_date", f"titledate:{ref}:{(source.get('startsAt') or '')[:10]}",
            f"{name}: title date doesn't match the start date",
            {"title": source.get("title"), "startsAt": source.get("startsAt")})

    if location_is_vague(party):
        add("location_vague", f"loc:{ref}", f"{name}: no usable location",
            {"location": party.get("location"), "englishAddress": source.get("englishAddress")})

    return issues


def detect_site_issues(party: dict, check: dict) -> list[dict]:
    """Compare what the public site actually rendered for a live party with
    what the database says (stale ISR/Runtime cache, broken redirect, 404)."""
    pid, ref = str(party.get("_id")), _event_ref(party)
    name = party.get("name") or pid
    problems = {}
    status = check.get("status")
    if status != 200:
        problems["status"] = status
    else:
        rendered_price = _num(check.get("price"))
        expected = party.get("ticketPrice")
        if isinstance(expected, (int, float)) and rendered_price is not None and abs(rendered_price - expected) > 0.5:
            problems["price"] = {"site": rendered_price, "db": expected}
        rendered_name = _text(check.get("name"))
        if rendered_name and norm_title(rendered_name) != norm_title(party.get("name") or ""):
            problems["name"] = {"site": rendered_name, "db": party.get("name")}
        final_url = str(check.get("finalUrl") or "")
        if final_url and "/archive/" in final_url:
            problems["redirectedTo"] = final_url
    if not problems:
        return []
    return [{"type": "site_render", "fingerprint": f"site:{ref}:{','.join(sorted(problems))}",
             "partyIds": [pid], "summary": f"{name}: the live page doesn't match the database",
             "evidence": problems}]


def is_upcoming(party: dict, now: datetime) -> bool:
    start = parse_local((party.get("source") or {}).get("startsAt") or party.get("date") or party.get("startsAt"))
    return bool(start) and start >= now - timedelta(days=1)
