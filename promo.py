"""
WhatsApp promo drafting.

Pure functions (no Flask, no Mongo) that rank upcoming parties by the commission
they are expected to earn and draft ready-to-paste Hebrew WhatsApp messages for
them. The `/api/admin/promo/whatsapp` route in app.py gathers the data and calls
in here; keeping this module import-free makes it trivially unit-testable.

Revenue model (see workspace root CLAUDE.md):
- account1 parties: flat ₪25 per ticket  -> always the top tier
- account2 parties: 6% of the ticket price
- a party's own `commission` ({"type": "fixed"|"percent", "value"}), set in the
  admin, overrides both for that party.

Also the site ranking (`rank_parties`): expected ₪ per page view = our ₪ per
ticket x how likely a viewer is to buy, which orders parties on the website.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")

# Set ACCOUNT1_FLAT_FEE on Render when the account1 deal changes (the owner
# said on 2026-10-10 it is about to drop to ₪15). This is the *expected* fee
# used for ranking and for choosing which duplicate listing to keep; recorded
# sales keep whatever the sales tracker logged at the time.
ACCOUNT1_FLAT_FEE = float(os.environ.get("ACCOUNT1_FLAT_FEE") or 25.0)
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


COMMISSION_TYPES = ("fixed", "percent")


def clean_commission(raw) -> dict | None:
    """A party's own deal, set by the owner in the admin: {"type": "fixed",
    "value": ₪ per ticket} or {"type": "percent", "value": % of the ticket
    price}. Returns the cleaned dict, or None when `raw` is not a valid one
    (callers then fall back to the account default)."""
    if not isinstance(raw, dict) or raw.get("type") not in COMMISSION_TYPES:
        return None
    try:
        value = float(raw.get("value"))
    except (TypeError, ValueError):
        return None
    if value != value or value < 0 or (raw["type"] == "percent" and value > 100) or value > 10000:
        return None
    return {"type": raw["type"], "value": round(value, 2)}


def commission_needs_price(tier: str, override: dict | None = None) -> bool:
    """True when the per-ticket figure is a share of the ticket price (so an
    unknown price makes it a guess)."""
    override = clean_commission(override)
    if override:
        return override["type"] == "percent"
    return tier != "account1"


def expected_commission_per_ticket(tier: str, ticket_price: float | None,
                                   override: dict | None = None) -> float:
    """Our ₪ per ticket. A per-party override (the party's `commission`)
    beats the account default (account1 flat fee, account2 6%)."""
    override = clean_commission(override)
    if override and override["type"] == "fixed":
        return override["value"]
    if not override and tier == "account1":
        return ACCOUNT1_FLAT_FEE
    price = ticket_price if ticket_price and ticket_price > 0 else DEFAULT_TICKET_PRICE
    pct = override["value"] / 100 if override else ACCOUNT2_PCT
    return round(price * pct, 2)


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


def commission_by_party(parties, *, sales_by_event_id: dict, accounts_by_event_id: dict,
                        now: datetime, account1_referral: str | None = None) -> dict[str, dict]:
    """
    Per upcoming party (keyed by party id): our commission per ticket and what
    the party has earned us so far. Same tier rule as the promo ranking.

    sales_by_event_id    - lifetime {goOutEventId: {"totalTicketsSold", "totalRevenue"}} from goout_sales_log
    accounts_by_event_id - {goOutEventId: {"account1", ...}} from goout_sales
    """
    now_il = now.astimezone(ISRAEL_TZ) if now.tzinfo else now.replace(tzinfo=timezone.utc).astimezone(ISRAEL_TZ)
    today = now_il.date()
    out: dict[str, dict] = {}
    for party in parties:
        party_id = str(party.get("_id") or party.get("id") or "")
        dt = party_datetime(party.get("date") or party.get("startsAt"))
        if not party_id or dt is None or dt.date() < today:
            continue
        event_id = str(party.get("goOutEventId") or "")
        sales = sales_by_event_id.get(event_id, {}) if event_id else {}
        tier = account_tier(
            accounts_by_event_id.get(event_id, ()) if event_id else (),
            party.get("referralCode"),
            account1_referral,
        )
        price = _price(party)
        override = clean_commission(party.get("commission"))
        out[party_id] = {
            "tier": tier,
            "perTicket": expected_commission_per_ticket(tier, price, override),
            # A %-of-price deal without a scraped price uses DEFAULT_TICKET_PRICE: say so.
            "perTicketEstimated": commission_needs_price(tier, override) and price is None,
            "commission": override,
            "ticketPrice": price,
            "ticketsSold": int(sales.get("totalTicketsSold") or 0),
            "earned": round(float(sales.get("totalRevenue") or 0.0), 2),
        }
    return out


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
        per_ticket = expected_commission_per_ticket(tier, price, party.get("commission"))
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
        candidate["campaignTemplate"] = format_party_template(candidate)
        candidates.append(candidate)

    candidates.sort(key=lambda c: (-c["score"], c["tier"] != "account1", c["date"]))
    return candidates[:limit]


# ---------------------------------------------------------------------------
# Message templates (WhatsApp markdown: *bold*, _italic_)
# ---------------------------------------------------------------------------

def _promo_link(c: dict) -> str:
    """Prefer our own site page over the raw GoOut link: keeps the visitor
    inside our funnel (site analytics, archive/upcoming redirects) instead of
    sending them straight to GoOut. The site's buy button already carries the
    party's referral-tagged GoOut URL, so nothing is lost."""
    return c.get("siteUrl") or c["url"]


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
    lines.append(_promo_link(c))
    return "\n".join(lines)


def format_party_template(c: dict) -> str:
    """Same message, but for the real WhatsApp-campaign engine: ends in the
    literal {link} placeholder (per-target referral link, filled in by the
    engine at send time) instead of a resolved URL. Feeds the WhatsApp tab's
    'New Send' form so a suggestion becomes a queued campaign in one click,
    rather than a manual copy-paste."""
    lines = [f"🎉 *{c['name']}*", f"📅 {c['dateLabel']}"]
    if c.get("location"):
        lines.append(f"📍 {c['location']}")
    details = " · ".join(x for x in (c.get("musicType"), c.get("age")) if x and x != "אחר")
    if details:
        lines.append(f"🎶 {details}")
    if c.get("ticketPrice"):
        lines.append(f"💸 החל מ-₪{int(c['ticketPrice'])}")
    lines.append("🎟️ כרטיסים בקישור 👇")
    lines.append("{link}")
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
        lines.append(f"🎟️ {_promo_link(c)}")
        lines.append("")
    lines.append(f"כל המסיבות: {SITE_BASE_URL}")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Site ranking: expected ₪ per view
# ---------------------------------------------------------------------------

# Prior strength, in views / clicks: a party with fewer than this much of its
# own traffic is mostly judged by the site-wide average, so one lucky click on
# a party nobody saw doesn't put it on top.
VIEW_PRIOR = 50
CLICK_PRIOR = 20
# Used only before the site has any traffic/sales to average over.
FALLBACK_CLICK_RATE = 0.05
FALLBACK_BUY_PER_CLICK = 0.1


def _site_rates(stats: list[dict]) -> tuple[float, float]:
    """Site-wide click rate and tickets per click, themselves nudged toward the
    fallbacks so a site with no sales yet doesn't score every party 0."""
    views = sum(s["views"] for s in stats)
    clicks = sum(s["redirects"] for s in stats)
    tickets = sum(s["tickets"] for s in stats)
    click_rate = (clicks + VIEW_PRIOR * FALLBACK_CLICK_RATE) / (views + VIEW_PRIOR)
    buy_per_click = (tickets + CLICK_PRIOR * FALLBACK_BUY_PER_CLICK) / (clicks + CLICK_PRIOR)
    return min(click_rate, 1.0), buy_per_click


def tickets_per_100_views(views: int, redirects: int, tickets: int,
                          site_click_rate: float, site_buy_per_click: float) -> float:
    """How likely a visitor of this party's page is to buy, as expected tickets
    per 100 views: (click rate) x (tickets per click), each shrunk toward the
    site average until the party has traffic of its own. Tickets include sales
    from any of our links (WhatsApp, Instagram), so a party that sells well off
    the site still counts as one people buy."""
    click_rate = (redirects + VIEW_PRIOR * site_click_rate) / (views + VIEW_PRIOR)
    buy_per_click = (tickets + CLICK_PRIOR * site_buy_per_click) / (redirects + CLICK_PRIOR)
    return round(100 * click_rate * buy_per_click, 3)


def rank_parties(parties, *, sales_by_event_id: dict, accounts_by_event_id: dict,
                 stats_by_party_id: dict, account1_referral: str | None = None) -> dict[str, dict]:
    """
    Per party id: our ₪ per ticket, the buy likelihood, and `score` = expected
    ₪ per 100 page views (the two multiplied) — what the site orders by. A sold
    out party scores 0. `rankScore` is the score scaled to 0-100 against the
    best party, which is all the public site gets (no deal terms).

    sales_by_event_id  - lifetime {goOutEventId: {"totalTicketsSold", ...}}
    stats_by_party_id  - {partyId: {"views", "redirects"}} from the site's own counters
    """
    rows = []
    for party in parties:
        party_id = str(party.get("_id") or party.get("id") or "")
        if not party_id:
            continue
        event_id = str(party.get("goOutEventId") or "")
        sales = sales_by_event_id.get(event_id, {}) if event_id else {}
        stats = stats_by_party_id.get(party_id) or {}
        tier = account_tier(accounts_by_event_id.get(event_id, ()) if event_id else (),
                            party.get("referralCode"), account1_referral)
        price = _price(party)
        override = clean_commission(party.get("commission"))
        rows.append({
            "partyId": party_id,
            "tier": tier,
            "commission": override,
            "perTicket": expected_commission_per_ticket(tier, price, override),
            "perTicketEstimated": commission_needs_price(tier, override) and price is None,
            "ticketPrice": price,
            "soldOut": bool(party.get("soldOut")),
            "views": max(0, int(stats.get("views") or 0)),
            "redirects": max(0, int(stats.get("redirects") or 0)),
            "tickets": max(0, int(sales.get("totalTicketsSold") or 0)),
        })
    click_rate, buy_per_click = _site_rates(rows)
    for row in rows:
        row["ticketsPer100Views"] = tickets_per_100_views(
            row["views"], row["redirects"], row["tickets"], click_rate, buy_per_click)
        row["score"] = 0.0 if row["soldOut"] else round(row["perTicket"] * row["ticketsPer100Views"], 3)
    best = max((r["score"] for r in rows), default=0.0)
    ordered = sorted(rows, key=lambda r: (-r["score"], r["partyId"]))
    for position, row in enumerate(ordered, start=1):
        row["rankScore"] = round(100 * row["score"] / best, 1) if best > 0 else 0.0
        row["position"] = position
    return {row["partyId"]: row for row in rows}
