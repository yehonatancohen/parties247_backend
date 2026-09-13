"""
WhatsApp send facts: freeze what we knew at send time, then fold in outcomes.

Pure functions (no Flask, no Mongo) so this is trivially unit-testable, same
split as promo.py. Two jobs live here:

1. `build_campaign_features` / `build_group_snapshot` — called once, at
   `POST /api/admin/wa/campaigns` time, to freeze party/message/group context
   onto the campaign doc. Never re-derived later: parties get edited or
   deleted, and waGroups.memberCount changes daily, so re-joining at read
   time would silently rewrite history.
2. `build_send_facts` — called by a rebuild job, joins the frozen features
   with clicks/analytics/sales snapshots into one row per (campaignId,
   chatId): the dataset a future recommender trains on.

Revenue model mirrors promo.py / the fetcher's sales_tracker.py exactly:
account1 = flat ₪25/ticket, account2 = 6% of ticket price. Sales here are
correlational (GoOut exposes no per-link attribution) and every fact row
says so explicitly rather than presenting a time-window count as verified.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import promo

ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")

ACCOUNT1_FLAT_FEE = promo.ACCOUNT1_FLAT_FEE
ACCOUNT2_PCT = promo.ACCOUNT2_PCT

# Outcome windows, in hours after sentAt, that build_send_facts buckets into.
OUTCOME_WINDOWS_HOURS = (1, 6, 24, 72)

_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]"
)


# ---------------------------------------------------------------------------
# 1. Feature snapshot, frozen at campaign-creation time
# ---------------------------------------------------------------------------

def _text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def build_message_features(text: str) -> dict:
    lines = [l for l in (text or "").split("\n") if l.strip()]
    return {
        "length": len(text or ""),
        "lineCount": len(lines),
        "emojiCount": len(_EMOJI_RE.findall(text or "")),
        "hasPrice": "₪" in (text or ""),
        "textHash": _text_hash(text or ""),
    }


def build_party_features(party: dict, *, account_ids, account1_referral: str | None,
                          ticket_totals: dict | None, now: datetime) -> dict:
    """
    Snapshot of everything about the party worth remembering at send time.
    `ticket_totals` is the matching `goout_sales` doc (confirmed_count etc.),
    already looked up by the caller via goOutEventId.
    """
    dt = promo.party_datetime(party.get("date") or party.get("startsAt"))
    tier = promo.account_tier(account_ids, party.get("referralCode"), account1_referral)
    price = party.get("ticketPrice")
    try:
        price = float(price) if price is not None else None
    except (TypeError, ValueError):
        price = None
    hours_until = None
    if dt is not None:
        now_il = now.astimezone(ISRAEL_TZ) if now.tzinfo else now.replace(tzinfo=timezone.utc).astimezone(ISRAEL_TZ)
        hours_until = round((dt - now_il).total_seconds() / 3600, 1)
    ticket_totals = ticket_totals or {}
    return {
        "goOutEventId": str(party.get("goOutEventId") or "") or None,
        "tier": tier,
        "expectedPerTicket": promo.expected_commission_per_ticket(tier, price),
        "ticketPrice": price,
        "musicType": party.get("musicType") or None,
        "eventType": party.get("eventType") or None,
        "age": party.get("age") or None,
        "areas": party.get("areas") or [],
        "region": party.get("region") or None,
        "partyStartsAt": dt.isoformat() if dt else None,
        "hoursUntilParty": hours_until,
        "ticketsSoldBefore": int(ticket_totals.get("confirmed_count") or 0),
    }


def build_group_snapshot(group: dict | None, *, chat_id: str, sends_last_7d: int) -> dict:
    """Per-target context, frozen per campaign target (waGroups mutates daily)."""
    group = group or {}
    return {
        "chatId": chat_id,
        "memberCount": group.get("memberCount"),
        "kind": group.get("kind") or "group",
        "sendsToGroupLast7d": sends_last_7d,
    }


def count_recent_sends_per_group(campaigns, *, chat_id: str, before: datetime, window_days: int = 7) -> int:
    """How many campaigns already targeted this group in the trailing window —
    the fatigue signal. `campaigns` is any iterable of campaign docs with
    `createdAt` and `targets[].chatId`."""
    cutoff = before - timedelta(days=window_days)
    count = 0
    for c in campaigns:
        created = c.get("createdAt")
        if not isinstance(created, datetime) or created < cutoff or created >= before:
            continue
        if any(t.get("chatId") == chat_id for t in (c.get("targets") or [])):
            count += 1
    return count


def count_unique_members_reached(members, *, chat_ids: set[str]) -> int:
    """Distinct memberHash across the target groups. `members` is an iterable
    of {"memberHash": ..., "groups": [...]} docs, projected down to those two
    fields by the caller (mirrors wa_members_overlap's own projection)."""
    seen = set()
    for m in members:
        groups = set(m.get("groups") or [])
        if groups & chat_ids:
            h = m.get("memberHash")
            if h:
                seen.add(h)
    return len(seen)


# ---------------------------------------------------------------------------
# 6. The fact table: one row per (campaignId, chatId)
# ---------------------------------------------------------------------------

def _local_hour_dow(dt: datetime) -> tuple[int | None, int | None]:
    if dt is None:
        return None, None
    local = dt.astimezone(ISRAEL_TZ) if dt.tzinfo else dt.replace(tzinfo=timezone.utc).astimezone(ISRAEL_TZ)
    return local.hour, local.weekday()


def _commission_for_tickets(tier: str, tickets: int, ticket_price: float | None) -> float:
    if tickets <= 0:
        return 0.0
    if tier == "account1":
        return round(ACCOUNT1_FLAT_FEE * tickets, 2)
    price = ticket_price if ticket_price and ticket_price > 0 else promo.DEFAULT_TICKET_PRICE
    return round(price * ACCOUNT2_PCT * tickets, 2)


def _window_bucket(sent_at: datetime, at: datetime) -> int | None:
    """Which OUTCOME_WINDOWS_HOURS bucket `at` falls into relative to
    `sent_at`, or None if beyond the largest window. Both must be aware."""
    if at < sent_at:
        return None
    delta_hours = (at - sent_at).total_seconds() / 3600
    for hours in OUTCOME_WINDOWS_HOURS:
        if delta_hours <= hours:
            return hours
    return None


def _empty_outcome_counts() -> dict:
    return {f"clicks_{h}h": 0 for h in OUTCOME_WINDOWS_HOURS} | \
           {f"buyClicks_{h}h": 0 for h in OUTCOME_WINDOWS_HOURS} | \
           {f"reactions_{h}h": 0 for h in OUTCOME_WINDOWS_HOURS} | \
           {f"replies_{h}h": 0 for h in OUTCOME_WINDOWS_HOURS}


def build_send_facts(campaigns, *, clicks, analytics_rows, snapshots, msg_events,
                      now: datetime) -> list[dict]:
    """
    One doc per (campaignId, chatId) target that actually has a sentAt.

    campaigns       - waCampaigns docs (with frozen `features`/`targets[].groupSnapshot`)
    clicks          - waClicks docs {code, campaignId, chatId, at, isBot, ipHash}
    analytics_rows  - `analytics` docs {category:"party", action, waCode, createdAt, partyId}
    snapshots       - goout_sales_snapshots docs {go_out_id, at, accepted} sorted or not
    msg_events      - waMessageEvents docs {waMsgId, chatId, type, at}
    """
    # Index clicks by (campaignId, chatId) -> list of (at, isBot, ipHash)
    clicks_by_target: dict[tuple, list] = {}
    for c in clicks or []:
        key = (c.get("campaignId"), c.get("chatId"))
        clicks_by_target.setdefault(key, []).append(c)

    # Index buy-click analytics rows by waCode -> list of docs
    analytics_by_code: dict[str, list] = {}
    for row in analytics_rows or []:
        code = row.get("waCode")
        if code:
            analytics_by_code.setdefault(code, []).append(row)

    # Index message events by (waMsgId, chatId) -> list
    msg_events_by_target: dict[tuple, list] = {}
    for e in msg_events or []:
        key = (e.get("waMsgId"), e.get("chatId"))
        msg_events_by_target.setdefault(key, []).append(e)

    # Sales snapshots by go_out_id, sorted by time, for baseline + delta lookups.
    snaps_by_event: dict[str, list] = {}
    for s in snapshots or []:
        snaps_by_event.setdefault(str(s.get("go_out_id")), []).append(s)
    for rows in snaps_by_event.values():
        rows.sort(key=lambda r: r.get("at") or now)

    facts = []
    for campaign in campaigns or []:
        features = campaign.get("features") or {}
        message_features = campaign.get("messageFeatures") or {}
        campaign_id = str(campaign.get("_id") or campaign.get("id") or "")
        tier = features.get("tier", "account2")
        ticket_price = features.get("ticketPrice")
        go_out_event_id = features.get("goOutEventId")

        for target in campaign.get("targets") or []:
            sent_at = target.get("sentAt")
            if not isinstance(sent_at, datetime):
                continue  # not actually sent yet, nothing to learn from
            chat_id = target.get("chatId")
            code = target.get("code")
            local_hour, local_dow = _local_hour_dow(sent_at)

            outcomes = _empty_outcome_counts()
            unique_clickers: set[str] = set()
            cross_party_buy_clicks = 0

            for click in clicks_by_target.get((campaign_id, chat_id), []):
                if click.get("isBot"):
                    continue
                at = click.get("at")
                if not isinstance(at, datetime):
                    continue
                bucket = _window_bucket(sent_at, at)
                if bucket is None:
                    continue
                for hours in OUTCOME_WINDOWS_HOURS:
                    if bucket <= hours:
                        outcomes[f"clicks_{hours}h"] += 1
                ip_hash = click.get("ipHash")
                if ip_hash:
                    unique_clickers.add(ip_hash)

            for row in analytics_by_code.get(code, []):
                at = row.get("createdAt")
                if not isinstance(at, datetime):
                    continue
                bucket = _window_bucket(sent_at, at)
                if bucket is None:
                    continue
                is_buy = row.get("action") == "redirect"
                if is_buy and row.get("partyId") and row["partyId"] != campaign.get("partyId"):
                    cross_party_buy_clicks += 1
                    continue
                if is_buy:
                    for hours in OUTCOME_WINDOWS_HOURS:
                        if bucket <= hours:
                            outcomes[f"buyClicks_{hours}h"] += 1

            for event in msg_events_by_target.get((target.get("waMsgId"), chat_id), []):
                at = event.get("at")
                if not isinstance(at, datetime):
                    continue
                bucket = _window_bucket(sent_at, at)
                if bucket is None:
                    continue
                kind = "reactions" if event.get("type") == "reaction" else \
                       "replies" if event.get("type") == "reply" else None
                if kind is None:
                    continue
                for hours in OUTCOME_WINDOWS_HOURS:
                    if bucket <= hours:
                        outcomes[f"{kind}_{hours}h"] += 1

            sales = {"attribution": "time-window", "note": (
                "Party+day level only — GoOut exposes no per-link attribution. "
                "This is a time correlation across all groups the party was sent to, "
                "not a verified per-group attribution."
            )}
            if go_out_event_id and go_out_event_id in snaps_by_event:
                rows = snaps_by_event[go_out_event_id]
                baseline = _accepted_delta(rows, sent_at - timedelta(hours=24), sent_at)
                delta_6h = _accepted_delta(rows, sent_at, sent_at + timedelta(hours=6))
                delta_24h = _accepted_delta(rows, sent_at, sent_at + timedelta(hours=24))
                sales["baselineAcceptedPer24h"] = baseline
                sales["acceptedDelta6h"] = delta_6h
                sales["acceptedDelta24h"] = delta_24h
                sales["estCommission6h"] = (
                    _commission_for_tickets(tier, max(0, delta_6h), ticket_price) if delta_6h is not None else None
                )
                sales["estCommission24h"] = (
                    _commission_for_tickets(tier, max(0, delta_24h), ticket_price) if delta_24h is not None else None
                )
            else:
                sales["baselineAcceptedPer24h"] = None
                sales["acceptedDelta6h"] = None
                sales["acceptedDelta24h"] = None
                sales["estCommission6h"] = None
                sales["estCommission24h"] = None

            group_snapshot = next(
                (g for g in campaign.get("groupSnapshots") or [] if g.get("chatId") == chat_id),
                {},
            )

            facts.append({
                "campaignId": campaign_id,
                "chatId": chat_id,
                "code": code,
                "partyId": campaign.get("partyId"),
                "partySlug": campaign.get("partySlug"),
                "sentAt": sent_at,
                "localHour": local_hour,
                "localDow": local_dow,
                "party": features,
                "message": message_features,
                "group": group_snapshot,
                "outcomes": outcomes,
                "uniqueClickers": len(unique_clickers),
                "crossPartyBuyClicks": cross_party_buy_clicks,
                "sales": sales,
                "updatedAt": now,
            })

    return facts


def _accepted_delta(snapshot_rows: list[dict], start: datetime, end: datetime) -> int | None:
    """accepted-at-end minus accepted-at-or-before-start, using the closest
    snapshot on each side. None if we have no snapshot before `start` (no
    baseline to diff against)."""
    before = None
    after = None
    for row in snapshot_rows:
        at = row.get("at")
        if not isinstance(at, datetime):
            continue
        if at <= start:
            before = row
        elif at <= end and after is None:
            after = row
        elif at <= end:
            after = row
    if before is None:
        return None
    end_value = after.get("accepted") if after else before.get("accepted")
    try:
        return int(end_value) - int(before.get("accepted") or 0)
    except (TypeError, ValueError):
        return None
