"""
WhatsApp promo drafting.

Pure functions (no Flask, no Mongo) that rank upcoming parties by the commission
they are expected to earn and draft ready-to-paste Hebrew WhatsApp messages for
them. The `/api/admin/promo/whatsapp` route in app.py gathers the data and calls
in here; keeping this module import-free makes it trivially unit-testable.

Revenue model (see workspace root CLAUDE.md):
- account1 parties: flat ₪25 per ticket  -> always the top tier
- account2 parties: 6% of the ticket price
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")

ACCOUNT1_FLAT_FEE = 25.0
ACCOUNT2_PCT = 0.06
# Used only to estimate account2 commission when a party has no scraped price yet.
DEFAULT_TICKET_PRICE = 100.0

SITE_BASE_URL = "https://www.parties247.co.il"

# Python weekday(): Monday=0 ... Sunday=6
HEBREW_WEEKDAYS = {
    0: "שני",
    1: "שלישי",
    2: "רביעי",
    3: "חמישי",
    4: "שישי",
    5: "שבת",
    6: "ראשון",
}


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

def party_datetime(value) -> datetime | None:
    """
    Parse a party date into an aware Israel-time datetime.

    Backend party dates are naive Israel wall-clock strings (confirmed
    2026-09-07: 119/119 upcoming events carry no offset). A naive value is
    therefore interpreted as Israel time, never as UTC. Aware values are
    converted.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            try:
                dt = datetime.fromisoformat(text[:19])
            except ValueError:
                return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=ISRAEL_TZ)
    return dt.astimezone(ISRAEL_TZ)


def hebrew_date_label(dt: datetime) -> str:
    """'יום חמישי 10.09 · 23:00' (time omitted when it's exactly midnight)."""
    local = dt.astimezone(ISRAEL_TZ)
    label = f"יום {HEBREW_WEEKDAYS[local.weekday()]} {local.day:02d}.{local.month:02d}"
    if local.hour or local.minute:
        label += f" · {local.hour:02d}:{local.minute:02d}"
    return label


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def account_tier(account_ids, referral_code: str | None = None,
                 account1_referral: str | None = None) -> str:
    """
    'account1' when any sales-tracker doc for the event belongs to account1, or
    when the party's referral code is account1's. Otherwise 'account2'.
    """
    for account_id in account_ids or ():
        if "account1" in str(account_id).lower():
            return "account1"
    if account1_referral and referral_code and referral_code == account1_referral:
        return "account1"
    return "account2"


def expected_commission_per_ticket(tier: str, ticket_price: float | None) -> float:
    if tier == "account1":
        return ACCOUNT1_FLAT_FEE
    price = ticket_price if ticket_price and ticket_price > 0 else DEFAULT_TICKET_PRICE
    return round(price * ACCOUNT2_PCT, 2)


def urgency_multiplier(days_until: float) -> float:
    """Parties happening sooner are worth pushing now; far-out ones can wait."""
    if days_until <= 2:
        return 2.0
    if days_until <= 4:
        return 1.5
    return 1.0


def score_candidate(per_ticket: float, tickets_recent: int, days_until: float) -> float:
    """
    Expected value-ish: commission per ticket, scaled by demonstrated demand
    (recent confirmed tickets) and urgency. `1 + tickets` so parties with no
    sales yet still rank by commission tier rather than dropping to zero.
    """
    return round(per_ticket * (1 + max(0, tickets_recent)) * urgency_multiplier(days_until), 2)


def _first_url(party: dict) -> str | None:
    for key in ("goOutUrl", "originalUrl", "canonicalUrl"):
        url = party.get(key)
        if url:
            return url
    return None


def _location_text(party: dict) -> str:
    loc = party.get("location")
    if isinstance(loc, dict):
        return loc.get("name") or loc.get("address") or ""
    return str(loc or "")


def _price(party: dict) -> float | None:
    raw = party.get("ticketPrice")
    try:
        price = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None
    return price if price and price > 0 else None


def rank_promo_candidates(parties, *, sales_by_event_id: dict, accounts_by_event_id: dict,
                          now: datetime, days: int = 7, limit: int = 12,
                          account1_referral: str | None = None) -> list[dict]:
    """
    Pick the upcoming parties (within `days`) most worth promoting, best first.

    parties            - party docs (already carrying referral-tagged URLs)
    sales_by_event_id  - {goOutEventId: {"totalTicketsSold": n, "totalRevenue": x}} for a recent window
    accounts_by_event_id - {goOutEventId: {"account1", ...}} from goout_sales
    """
    now_il = now.astimezone(ISRAEL_TZ) if now.tzinfo else now.replace(tzinfo=timezone.utc).astimezone(ISRAEL_TZ)
    horizon = now_il + timedelta(days=days)
    today = now_il.date()

    candidates: list[dict] = []
    for party in parties:
        if party.get("hidden") or party.get("isPromotion"):
            continue
        dt = party_datetime(party.get("date") or party.get("startsAt"))
        if dt is None or dt.date() < today or dt > horizon:
            continue
        url = _first_url(party)
        if not url:
            continue

        event_id = str(party.get("goOutEventId") or "")
        sales = sales_by_event_id.get(event_id, {}) if event_id else {}
        tickets_recent = int(sales.get("totalTicketsSold") or 0)
        tier = account_tier(
            accounts_by_event_id.get(event_id, ()) if event_id else (),
            party.get("referralCode"),
            account1_referral,
        )
        price = _price(party)
        per_ticket = expected_commission_per_ticket(tier, price)
        days_until = (dt - now_il).total_seconds() / 86400
        score = score_candidate(per_ticket, tickets_recent, days_until)

        slug = party.get("slug") or ""
        candidate = {
            "partyId": str(party.get("_id") or party.get("id") or ""),
            "slug": slug,
            "name": party.get("name") or "",
            "date": dt.isoformat(),
            "dateLabel": hebrew_date_label(dt),
            "location": _location_text(party),
            "musicType": party.get("musicType") or "",
            "age": party.get("age") or "",
            "ticketPrice": price,
            "tier": tier,
            "expectedPerTicket": per_ticket,
            "ticketsLast30d": tickets_recent,
            "revenueLast30d": round(float(sales.get("totalRevenue") or 0.0), 2),
            "daysUntil": round(days_until, 1),
            "score": score,
            "url": url,
            "siteUrl": f"{SITE_BASE_URL}/event/{slug}" if slug else None,
        }
        candidate["message"] = format_party_message(candidate)
        candidates.append(candidate)

    candidates.sort(key=lambda c: (-c["score"], c["tier"] != "account1", c["date"]))
    return candidates[:limit]


# ---------------------------------------------------------------------------
# Message templates (WhatsApp markdown: *bold*, _italic_)
# ---------------------------------------------------------------------------

def format_party_message(c: dict) -> str:
    lines = [f"🎉 *{c['name']}*", f"📅 {c['dateLabel']}"]
    if c.get("location"):
        lines.append(f"📍 {c['location']}")
    details = " · ".join(x for x in (c.get("musicType"), c.get("age")) if x and x != "אחר")
    if details:
        lines.append(f"🎶 {details}")
    if c.get("ticketPrice"):
        lines.append(f"💸 החל מ-₪{int(c['ticketPrice'])}")
    lines.append("🎟️ כרטיסים בקישור 👇")
    lines.append(c["url"])
    return "\n".join(lines)


_DIGIT_EMOJI = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]


def format_digest_message(candidates: list[dict], *, days: int = 7, max_items: int = 8) -> str:
    """One roundup message for a promo group: the top N parties with links."""
    if not candidates:
        return ""
    heading = "לסוף השבוע" if days <= 4 else "לשבוע הקרוב" if days <= 8 else "לתקופה הקרובה"
    lines = [f"🔥 *המסיבות הכי חמות {heading}* 🔥", ""]
    for idx, c in enumerate(candidates[:max_items]):
        bullet = _DIGIT_EMOJI[idx] if idx < len(_DIGIT_EMOJI) else "•"
        where = f", {c['location']}" if c.get("location") else ""
        lines.append(f"{bullet} *{c['name']}* — {c['dateLabel']}{where}")
        lines.append(f"🎟️ {c['url']}")
        lines.append("")
    lines.append(f"כל המסיבות: {SITE_BASE_URL}")
    return "\n".join(lines).rstrip()
