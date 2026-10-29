"""
Admin data export (GET /api/admin/export): one window's views, clicks to GoOut
and sales, joined per party, ready for the admin to write into an Excel file.

Pure: app.py loads the rows from Mongo (views already grouped per party per
Israel day, clicks and sales one row each) and this module joins them. The
counting rules match the analytics dashboard (build_party_funnel): every party
event counts, bots included, so the two never disagree; bots get their own
column instead. Hidden/merged parties are not filtered out; a merged listing's
views, clicks and sales are folded into the party it was merged into.

Times come back as Israel local time, "YYYY-MM-DDTHH:MM:SS", no offset.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")
SITE_URL = "https://www.parties247.co.il/event/"
CLICK_TO_SALE_WINDOW = timedelta(hours=24)


def _utc(value):
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def local_str(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(ISRAEL_TZ).strftime("%Y-%m-%dT%H:%M:%S")


def local_day(value: datetime) -> str:
    return local_str(value)[:10]


def parse_party_date(value) -> datetime | None:
    """Party dates are stored as naive Israel-local strings or datetimes."""
    if not value:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=ISRAEL_TZ) if value.tzinfo is None else value


def resolve_party(party_id: str | None, parties: dict[str, dict]) -> str | None:
    """Follow mergedInto to the listing that is actually live."""
    seen = set()
    while party_id and party_id not in seen:
        seen.add(party_id)
        target = (parties.get(party_id) or {}).get("mergedInto")
        if not target or str(target) not in parties:
            return party_id
        party_id = str(target)
    return party_id


def _r(value: float) -> float:
    return round(float(value or 0), 2)


def _pct(part: float, whole: float) -> float | None:
    return round(100.0 * part / whole, 2) if whole else None


def build_export(*, start: datetime | None, end: datetime, parties: dict[str, dict],
                 view_buckets: list[dict], unique_sessions: dict[str, dict],
                 clicks: list[dict], sales: list[dict], snapshots: list[dict],
                 lifetime_by_event: dict[str, dict], now: datetime) -> dict:
    """
    parties          party id -> party doc (name, slug, date, goOutEventId, mergedInto, ...)
    view_buckets     {partyId, day, action, channel, bot, count}: party events grouped in Mongo
    unique_sessions  party id -> {"view": n, "redirect": n} distinct sessionIds in the window
    clicks           raw redirect events in the window plus the 24h before it (for sales
                     right after the window opens); only the ones inside it are listed
    sales            goout_sales_log rows recorded in the window
    snapshots        goout_sales docs (GoOut lifetime counters per event and account)
    lifetime_by_event go_out_id -> {"totalRevenue", "totalTicketsSold"} over all time
    """
    # pymongo hands back naive UTC datetimes; make every time comparable.
    clicks = [{**c, "createdAt": _utc(c.get("createdAt"))} for c in clicks]
    sales = [{**s, "recorded_at": _utc(s.get("recorded_at"))} for s in sales]
    start, end, now = _utc(start), _utc(end), _utc(now)

    by_event = {}
    for pid, party in parties.items():
        event_id = party.get("goOutEventId")
        if event_id:
            by_event.setdefault(str(event_id), pid)

    def keeper_of(pid):
        return resolve_party(str(pid), parties) if pid else None

    stats: dict[str, dict] = {}

    def row_for(pid: str) -> dict:
        if pid not in stats:
            stats[pid] = {"views": 0, "botViews": 0, "clicks": 0, "botClicks": 0,
                          "tickets": 0, "commission": 0.0, "gross": 0.0, "salesRows": 0,
                          "uniqueViewers": 0, "uniqueClickers": 0}
        return stats[pid]

    daily: dict[str, dict] = {}
    party_daily: dict[tuple, dict] = {}
    channels: dict[str, dict] = {}

    def day_row(day):
        return daily.setdefault(day, {"day": day, "views": 0, "clicks": 0, "tickets": 0,
                                      "commission": 0.0, "gross": 0.0})

    def party_day_row(day, pid):
        return party_daily.setdefault((day, pid), {"day": day, "partyId": pid, "views": 0,
                                                   "clicks": 0, "tickets": 0, "commission": 0.0})

    def channel_row(name):
        return channels.setdefault(name, {"channel": name, "views": 0, "clicks": 0,
                                          "salesAfterClick": 0, "ticketsAfterClick": 0,
                                          "commissionAfterClick": 0.0})

    for bucket in view_buckets:
        pid = keeper_of(bucket.get("partyId"))
        if not pid:
            continue
        count = int(bucket.get("count") or 0)
        is_click = bucket.get("action") == "redirect"
        row = row_for(pid)
        row["clicks" if is_click else "views"] += count
        if bucket.get("bot"):
            row["botClicks" if is_click else "botViews"] += count
        day = bucket.get("day")
        day_row(day)["clicks" if is_click else "views"] += count
        party_day_row(day, pid)["clicks" if is_click else "views"] += count
        channel_row(bucket.get("channel") or "untracked")["clicks" if is_click else "views"] += count

    for pid, counts in unique_sessions.items():
        kid = keeper_of(pid)
        if kid:
            row_for(kid)["uniqueViewers"] += int(counts.get("view") or 0)
            row_for(kid)["uniqueClickers"] += int(counts.get("redirect") or 0)

    # Clicks per party, sorted, for the click <-> sale joins below.
    clicks_by_party: dict[str, list[dict]] = {}
    for click in clicks:
        pid = keeper_of(click.get("partyId"))
        at = click.get("createdAt")
        if pid and isinstance(at, datetime):
            clicks_by_party.setdefault(pid, []).append(click)
    for lst in clicks_by_party.values():
        lst.sort(key=lambda c: c["createdAt"])
    click_times = {pid: [c["createdAt"] for c in lst] for pid, lst in clicks_by_party.items()}

    sale_rows = []
    sale_times_by_party: dict[str, list[tuple]] = {}
    for sale in sorted(sales, key=lambda s: s.get("recorded_at") or now):
        tickets = int(sale.get("delta_confirmed") or 0)
        earned = float(sale.get("revenue_earned") or 0)
        gross = float(sale.get("delta_event_revenue") or 0)
        if tickets <= 0 and earned <= 0 and gross <= 0:
            continue  # nothing happened (e.g. a double count that was corrected to zero)
        at = sale.get("recorded_at")
        event_id = str(sale.get("go_out_id") or "")
        pid = keeper_of(by_event.get(event_id))
        party = parties.get(pid) or {}
        party_date = parse_party_date(party.get("date") or party.get("startsAt"))

        before = []
        if pid and at:
            times = click_times.get(pid, [])
            lo = bisect_left(times, at - CLICK_TO_SALE_WINDOW)
            hi = bisect_right(times, at)
            before = clicks_by_party[pid][lo:hi] if times else []
        last = before[-1] if before else None
        day = local_day(at) if at else None

        if pid:
            row = row_for(pid)
            row["tickets"] += max(0, tickets)
            row["commission"] += earned
            row["gross"] += gross
            row["salesRows"] += 1
            sale_times_by_party.setdefault(pid, []).append((at, max(0, tickets)))
        if day:
            d = day_row(day)
            d["tickets"] += max(0, tickets)
            d["commission"] += earned
            d["gross"] += gross
            if pid:
                pd = party_day_row(day, pid)
                pd["tickets"] += max(0, tickets)
                pd["commission"] += earned
        if last:
            ch = channel_row(last.get("channel") or "untracked")
            ch["salesAfterClick"] += 1
            ch["ticketsAfterClick"] += max(0, tickets)
            ch["commissionAfterClick"] += earned

        sale_rows.append({
            "recordedAt": local_str(at),
            "source": "fast (20 min)" if sale.get("source") == "fast_sales_alerts" else "4h run",
            "account": sale.get("account_id"),
            "partyName": party.get("name") or sale.get("event_name"),
            "partyDate": local_str(party_date),
            "daysBeforeParty": round((party_date - at).total_seconds() / 86400, 1)
            if party_date and at else None,
            "goOutEventId": event_id,
            "tickets": tickets,
            "ticketPrice": sale.get("ticket_price"),
            "grossDelta": _r(gross),
            "commission": _r(earned),
            "clicks24hBefore": len(before),
            "lastClickAt": local_str(last["createdAt"]) if last else None,
            "minutesFromLastClick": round((at - last["createdAt"]).total_seconds() / 60)
            if last and at else None,
            "lastClickChannel": last.get("channel") if last else None,
            "lastClickSource": last.get("firstSource") if last else None,
            "lastClickCampaign": last.get("firstCampaign") if last else None,
            "lastClickDevice": last.get("deviceType") if last else None,
            "corrected": bool(sale.get("double_count_fix") or sale.get("fee_change")),
            "partySlug": party.get("slug"),
        })

    click_rows = []
    for click in sorted(clicks, key=lambda c: c.get("createdAt") or now):
        at = click.get("createdAt")
        if not isinstance(at, datetime) or (start and at < start) or at >= end:
            continue
        pid = keeper_of(click.get("partyId"))
        party = parties.get(pid) or {}
        followed = [t for (t, n) in sale_times_by_party.get(pid, []) if t and at <= t <= at + CLICK_TO_SALE_WINDOW]
        click_rows.append({
            "clickedAt": local_str(at),
            "partyName": party.get("name") or click.get("partyName"),
            "partyDate": local_str(parse_party_date(party.get("date") or party.get("startsAt"))),
            "channel": click.get("channel") or "untracked",
            "source": click.get("firstSource"),
            "medium": click.get("firstMedium"),
            "campaign": click.get("firstCampaign"),
            "referrerHost": click.get("firstReferrerHost") or click.get("refererHost"),
            "landingPath": click.get("landingPath"),
            "device": click.get("deviceType"),
            "waCode": click.get("waCode"),
            "session": (click.get("sessionId") or "")[:8] or None,
            "saleWithin24h": bool(followed),
            "partySlug": party.get("slug"),
        })

    # GoOut lifetime counters per party (all accounts that track its event).
    goout: dict[str, dict] = {}
    goout_daily = []
    for snap in snapshots:
        event_id = str(snap.get("go_out_id") or "")
        pid = keeper_of(by_event.get(event_id))
        if not pid:
            continue
        g = goout.setdefault(pid, {"accounts": set(), "goOutViews": 0, "accepted": 0, "pending": 0,
                                   "rejected": 0, "abandoned": 0, "lifetimeTickets": 0,
                                   "lifetimeCommission": 0.0, "goOutGross": 0.0})
        g["accounts"].add(snap.get("account_id"))
        g["goOutViews"] = max(g["goOutViews"], int(snap.get("views") or 0))
        ticket_stats = ((snap.get("endone_stats") or {}).get("ticket_stats")) or {}
        for key, src in (("accepted", "Accepted"), ("pending", "Pending"),
                         ("rejected", "Rejected"), ("abandoned", "Abandoned")):
            g[key] += int(ticket_stats.get(src) or 0)
        g["goOutGross"] += float(snap.get("real_own_revenue") or 0)
        life = lifetime_by_event.get(event_id) or {}
        g["lifetimeTickets"] += int(life.get("totalTicketsSold") or 0)
        g["lifetimeCommission"] += float(life.get("totalRevenue") or 0)
        dates = (((snap.get("endone_stats") or {}).get("sales_per_date")) or {}).get("dates") or {}
        party = parties.get(pid) or {}
        for ms, n in dates.items():
            try:
                day_utc = datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)
            except (TypeError, ValueError, OverflowError):
                continue
            if (start and day_utc < start - timedelta(days=1)) or day_utc >= end:
                continue
            goout_daily.append({"day": day_utc.strftime("%Y-%m-%d"), "partyName": party.get("name"),
                                "account": snap.get("account_id"), "tickets": int(n or 0),
                                "goOutEventId": event_id})
    goout_daily.sort(key=lambda r: (r["day"], r["partyName"] or ""))

    party_rows = []
    for pid, row in stats.items():
        party = parties.get(pid) or {}
        g = goout.get(pid) or {}
        party_rows.append({
            "partyName": party.get("name"),
            "partyDate": local_str(parse_party_date(party.get("date") or party.get("startsAt"))),
            "status": party.get("listingStatus") or "listed",
            "accounts": ", ".join(sorted(a for a in g.get("accounts", set()) if a)) or None,
            "region": party.get("region"),
            "location": party.get("location") if isinstance(party.get("location"), str) else None,
            "musicType": party.get("musicType"),
            "ticketPrice": party.get("ticketPrice"),
            "views": row["views"],
            "uniqueViewers": row["uniqueViewers"],
            "botViews": row["botViews"],
            "clicks": row["clicks"],
            "uniqueClickers": row["uniqueClickers"],
            "clickRate": _pct(row["clicks"], row["views"]),
            "tickets": row["tickets"],
            "ticketsPerClick": _pct(row["tickets"], row["clicks"]),
            "commission": _r(row["commission"]),
            "commissionPer100Views": _r(100 * row["commission"] / row["views"]) if row["views"] else None,
            "gross": _r(row["gross"]),
            "goOutViews": g.get("goOutViews"),
            "goOutAccepted": g.get("accepted"),
            "goOutPending": g.get("pending"),
            "goOutRejected": g.get("rejected"),
            "goOutAbandoned": g.get("abandoned"),
            "goOutGross": _r(g["goOutGross"]) if g else None,
            "lifetimeTickets": g.get("lifetimeTickets"),
            "lifetimeCommission": _r(g["lifetimeCommission"]) if g else None,
            "goOutEventId": party.get("goOutEventId"),
            "siteUrl": SITE_URL + party["slug"] if party.get("slug") else None,
        })
    party_rows.sort(key=lambda r: (-r["commission"], -r["clicks"], -r["views"]))

    for d in daily.values():
        d["clickRate"] = _pct(d["clicks"], d["views"])
        d["commission"] = _r(d["commission"])
        d["gross"] = _r(d["gross"])
    for pd in party_daily.values():
        party = parties.get(pd["partyId"]) or {}
        pd["partyName"] = party.get("name")
        pd["commission"] = _r(pd["commission"])
    channel_rows = sorted(channels.values(), key=lambda c: -c["clicks"])
    for ch in channel_rows:
        ch["clickRate"] = _pct(ch["clicks"], ch["views"])
        ch["commissionAfterClick"] = _r(ch["commissionAfterClick"])

    by_account: dict[str, dict] = {}
    for s in sale_rows:
        a = by_account.setdefault(s["account"] or "?", {"account": s["account"], "tickets": 0, "commission": 0.0, "gross": 0.0})
        a["tickets"] += max(0, s["tickets"])
        a["commission"] = _r(a["commission"] + s["commission"])
        a["gross"] = _r(a["gross"] + s["grossDelta"])

    total_views = sum(r["views"] for r in stats.values())
    total_clicks = sum(r["clicks"] for r in stats.values())
    total_tickets = sum(max(0, s["tickets"]) for s in sale_rows)
    total_commission = _r(sum(s["commission"] for s in sale_rows))
    return {
        "range": {"from": local_str(start)[:10] if start else None,
                  "to": local_str(end - timedelta(seconds=1))[:10],
                  "generatedAt": local_str(now)},
        "totals": {
            "views": total_views,
            "botViews": sum(r["botViews"] for r in stats.values()),
            "clicks": total_clicks,
            "clickRate": _pct(total_clicks, total_views),
            "tickets": total_tickets,
            "ticketsPerClick": _pct(total_tickets, total_clicks),
            "commission": total_commission,
            "gross": _r(sum(s["grossDelta"] for s in sale_rows)),
            "parties": len(party_rows),
            "salesRows": len(sale_rows),
            "byAccount": sorted(by_account.values(), key=lambda a: a["account"] or ""),
        },
        "parties": party_rows,
        "sales": sale_rows,
        "clicks": click_rows,
        "daily": sorted(daily.values(), key=lambda d: d["day"]),
        "partyDaily": sorted(party_daily.values(), key=lambda d: (d["day"], d["partyName"] or "")),
        "channels": channel_rows,
        "goOutDaily": goout_daily,
    }
