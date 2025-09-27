import os
import json
import logging
import hmac
import re
import copy
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from functools import wraps
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, quote_plus, urljoin
try:
    from pymongo import UpdateOne
except Exception:  # pragma: no cover - used only when pymongo isn't available in tests
    class UpdateOne:  # minimal stub for tests
        def __init__(self, *args, **kwargs):
            self._args = args
            self._kwargs = kwargs
import socket
import ipaddress

import bcrypt
import jwt
from dotenv import load_dotenv
from flask import Flask, request, jsonify, url_for
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pymongo import MongoClient, errors
from pymongo.collection import Collection
from bson.objectid import ObjectId
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, ValidationError

# --- App setup ---
load_dotenv()
app = Flask(__name__)
app.config["RATELIMIT_HEADERS_ENABLED"] = True
CORS(app)
limiter = Limiter(get_remote_address, app=app)
logging.basicConfig(level=logging.INFO)

CANONICAL_BASE_URL = (os.environ.get("CANONICAL_BASE_URL") or "https://parties247-website.vercel.app").rstrip("/")
INDEXNOW_KEY = os.environ.get("INDEXNOW_KEY")
INDEXNOW_ENDPOINT = os.environ.get("INDEXNOW_ENDPOINT", "https://api.indexnow.org/indexnow")
INDEXNOW_KEY_LOCATION = os.environ.get("INDEXNOW_KEY_LOCATION")
REVALIDATE_ENDPOINT = os.environ.get("REVALIDATE_ENDPOINT")
REVALIDATE_SECRET = os.environ.get("REVALIDATE_SECRET")

EVENT_CACHE_SECONDS = 1800
LIST_CACHE_SECONDS = 600
FEED_CACHE_SECONDS = 900
ICS_CACHE_SECONDS = 3600
SITEMAP_CACHE_SECONDS = 900
EVENT_SITEMAP_CHUNK = 500

# --- URL canonicalization ---
TRACKING_PREFIXES = ("utm_",)
TRACKING_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid"}

REFERRAL_KEY = "referral"

CANONICAL_PATTERNS = {
    "event": "/event/{slug}",
    "city": "/city/{slug}",
    "venue": "/venue/{slug}",
    "genre": "/genre/{slug}",
    "date": "/date/{slug}",
    "events_feed": "/feeds/events.{fmt}",
    "scoped_feed": "/feeds/{slug}.{fmt}",
    "events_ics": "/ics/event/{slug}.ics",
    "city_ics": "/ics/city/{slug}.ics",
    "sitemap_index": "/sitemap.xml",
    "sitemap_child": "/sitemaps/{name}",
}


def build_canonical(kind: str, **params) -> str:
    """Return an absolute canonical URL based on the configured host."""
    pattern = CANONICAL_PATTERNS.get(kind)
    if not pattern:
        raise KeyError(f"Unsupported canonical kind: {kind}")
    path = pattern.format(**params)
    if not path.startswith("/"):
        path = "/" + path
    return f"{CANONICAL_BASE_URL}{path}"


def absolute_url(path: str) -> str:
    """Build an absolute URL from an absolute or relative path."""
    if not path:
        return CANONICAL_BASE_URL
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if not path.startswith("/"):
        path = "/" + path
    return f"{CANONICAL_BASE_URL}{path}"


def slugify_value(value: str | None) -> str | None:
    if not value:
        return None
    slug = re.sub(r"[^0-9a-zA-Z]+", "-", value.lower()).strip("-")
    return slug or None


def parse_datetime(value) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            if text.endswith("Z"):
                return datetime.fromisoformat(text[:-1]).replace(tzinfo=timezone.utc)
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(text, fmt)
                if fmt == "%Y-%m-%d":
                    return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def isoformat_or_none(value) -> str | None:
    dt = parse_datetime(value)
    if not dt:
        return None
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def extract_bilingual(source: dict, base_key: str, fallback: str | None = None) -> dict:
    he_keys = [f"{base_key}He", f"{base_key}_he", f"{base_key}_hebrew"]
    en_keys = [f"{base_key}En", f"{base_key}_en", f"{base_key}_english"]
    he_val = None
    en_val = None
    for key in he_keys:
        if key in source and source[key]:
            he_val = source[key]
            break
    for key in en_keys:
        if key in source and source[key]:
            en_val = source[key]
            break
    base_val = source.get(base_key)
    if isinstance(base_val, dict):
        he_val = he_val or base_val.get("he")
        en_val = en_val or base_val.get("en")
    elif isinstance(base_val, str):
        he_val = he_val or base_val
        en_val = en_val or base_val
    he_val = he_val or fallback or ""
    en_val = en_val or fallback or ""
    return {"he": he_val, "en": en_val}


def extract_geo(doc: dict) -> dict:
    geo = doc.get("geo") or {}
    if isinstance(geo, dict):
        lat = geo.get("lat") or geo.get("latitude")
        lon = geo.get("lon") or geo.get("lng") or geo.get("longitude")
        address = geo.get("address") or geo.get("street")
        postal = geo.get("postalCode") or geo.get("zip")
    else:
        lat = lon = address = postal = None
    lat = lat if isinstance(lat, (int, float)) else doc.get("lat") or doc.get("latitude")
    lon = lon if isinstance(lon, (int, float)) else doc.get("lon") or doc.get("longitude")
    address = address or doc.get("address") or doc.get("location")
    postal = postal or doc.get("postalCode")
    result = {
        "lat": float(lat) if isinstance(lat, (int, float, str)) and str(lat).strip() else None,
        "lon": float(lon) if isinstance(lon, (int, float, str)) and str(lon).strip() else None,
        "address": address or "",
        "postalCode": postal or "",
    }
    return result


def derive_status(doc: dict) -> str:
    status = (doc.get("status") or "").lower()
    allowed = {"scheduled", "cancelled", "postponed", "past"}
    if status in allowed:
        return status
    start = parse_datetime(doc.get("startsAt") or doc.get("date"))
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    if start and start < now:
        return "past"
    return "scheduled"

def normalize_url(raw: str) -> str:
    p = urlparse((raw or "").strip())
    scheme = (p.scheme or "https").lower()
    netloc = p.netloc.lower()
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    if netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]
    path = p.path.rstrip("/") or "/"
    cleaned_qs = []
    for k, v in parse_qsl(p.query, keep_blank_values=True):
        kl = k.lower()
        if kl.startswith(TRACKING_PREFIXES) or kl in TRACKING_KEYS:
            continue
        cleaned_qs.append((k, v))
    query = urlencode(cleaned_qs, doseq=True)
    return urlunparse((scheme, netloc, path, "", query, ""))

def normalized_or_none_for_dedupe(raw: str) -> str | None:
    n = normalize_url(raw)
    return n if urlparse(n).netloc else None


def append_referral_param(url: str | None, referral: str | None) -> str | None:
    """Append the referral query parameter to the URL when missing."""
    if not url or not referral:
        return url
    try:
        parsed = urlparse(url)
    except Exception:
        return url
    if not parsed.netloc:
        return url
    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    if any(k.lower() == "ref" for k, _ in query_items):
        return urlunparse(parsed)
    query_items.append(("ref", referral))
    new_query = urlencode(query_items, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def default_referral_code() -> str | None:
    """Fetch the default referral code from the settings collection."""
    if settings_collection is None:
        return None
    try:
        doc = settings_collection.find_one({"key": REFERRAL_KEY}) or {}
    except Exception as exc:
        app.logger.warning(f"Failed to fetch referral code: {exc}")
        return None
    value = (doc.get("value") or "").strip()
    return value or None


def apply_default_referral(party: dict, referral: str | None) -> None:
    """Mutate the party dict to ensure URLs carry the referral code."""
    if not party or not referral:
        return
    for key in ("goOutUrl", "originalUrl"):
        if key in party:
            party[key] = append_referral_param(party.get(key), referral)
    if not party.get("referralCode"):
        party["referralCode"] = referral

ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_PORTS = {80, 443}

def is_url_allowed(raw: str) -> bool:
    try:
        parsed = urlparse(raw)
    except Exception:
        return False
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return False
    if parsed.port and parsed.port not in ALLOWED_PORTS:
        return False
    host = parsed.hostname
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, parsed.port or (80 if parsed.scheme == "http" else 443))
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local or ip.is_multicast:
            return False
    return True


def json_response(payload, status: int = 200, cache_seconds: int | None = None, robots: str | None = "noindex"):
    headers = {
        "Content-Type": "application/json",
    }
    if cache_seconds is not None:
        headers["Cache-Control"] = f"public, max-age={cache_seconds}"
    if robots:
        headers["X-Robots-Tag"] = robots
    return jsonify(payload), status, headers


def text_response(body: str, content_type: str, cache_seconds: int | None = None, robots: str | None = None, status: int = 200):
    headers = {"Content-Type": content_type}
    if cache_seconds is not None:
        headers["Cache-Control"] = f"public, max-age={cache_seconds}"
    if robots:
        headers["X-Robots-Tag"] = robots
    return body, status, headers

# --- Mongo helpers ---
def ensure_index(coll: Collection, keys, name: str, **kwargs):
    info = coll.index_information()
    if name in info:
        meta = info[name]
        same_keys = meta.get("key") == list(keys)
        same_unique = bool(meta.get("unique", False)) == bool(kwargs.get("unique", False))
        same_pfe = meta.get("partialFilterExpression") == kwargs.get("partialFilterExpression")
        if same_keys and same_unique and same_pfe:
            return
        try:
            coll.drop_index(name)
        except Exception as e:
            app.logger.warning(f"Drop index {name} failed: {e}")
    coll.create_index(list(keys), name=name, **kwargs)

# --- Mongo connection + index hygiene ---
try:
    client = MongoClient(os.environ.get("MONGODB_URI"))
    db = client["party247"]
    parties_collection: Collection = db.parties
    carousels_collection: Collection = db.carousels
    tags_collection: Collection = db.tags
    settings_collection: Collection = db.settings

    try:
        parties_collection.update_many(
            {"$or": [{"goOutUrl": None}, {"goOutUrl": ""}]},
            {"$unset": {"goOutUrl": ""}}
        )
    except Exception as e:
        app.logger.warning(f"goOutUrl cleanup failed: {e}")

    try:
        for name, meta in parties_collection.index_information().items():
            if meta.get("unique") and meta.get("key") == [("originalUrl", 1)]:
                parties_collection.drop_index(name)
                app.logger.info(f"Dropped legacy unique index on originalUrl: {name}")
    except Exception as e:
        app.logger.warning(f"Drop originalUrl index failed: {e}")

    try:
        for name, meta in parties_collection.index_information().items():
            if meta.get("key") == [("goOutUrl", 1)]:
                parties_collection.drop_index(name)
                app.logger.info(f"Dropped legacy index on goOutUrl: {name}")
    except Exception as e:
        app.logger.warning(f"Drop goOutUrl index failed: {e}")

    ensure_index(
        parties_collection,
        [("canonicalUrl", 1)],
        name="unique_canonicalUrl",
        unique=True,
        partialFilterExpression={"canonicalUrl": {"$exists": True, "$type": "string"}}
    )
    ensure_index(
        parties_collection,
        [("goOutUrl", 1)],
        name="unique_goOutUrl",
        unique=True,
        partialFilterExpression={"goOutUrl": {"$exists": True, "$type": "string"}}
    )
    ensure_index(parties_collection, [("date", 1)], name="date_asc")

    ensure_index(carousels_collection, [("title", 1)], name="title_asc")
    ensure_index(carousels_collection, [("order", 1)], name="carousel_order_asc")

    ensure_index(tags_collection, [("slug", 1)], name="unique_slug", unique=True)
    ensure_index(tags_collection, [("order", 1)], name="tag_order_asc")

    ensure_index(settings_collection, [("key", 1)], name="unique_key", unique=True)

    app.logger.info("Connected to MongoDB and ensured indexes.")
except Exception as e:
    app.logger.error(f"Error connecting to MongoDB Atlas: {e}")
    parties_collection = None
    carousels_collection = None
    tags_collection = None
    settings_collection = None


def record_setting_hit(key: str, extra: dict | None = None):
    if settings_collection is None:
        return
    payload = {"key": key, "lastHit": datetime.utcnow()}
    if extra:
        payload.update(extra)
    try:
        settings_collection.update_one({"key": key}, {"$set": payload}, upsert=True)
    except Exception as exc:  # pragma: no cover - best effort only
        app.logger.warning(f"Failed to persist analytics hit {key}: {exc}")


def all_events() -> list[dict]:
    if parties_collection is None:
        return []
    try:
        docs = list(parties_collection.find())
    except TypeError:  # pragma: no cover - compatibility with simple stubs
        docs = list(parties_collection.find({}))
    except Exception as exc:
        app.logger.error(f"Failed to fetch events: {exc}")
        return []
    for doc in docs:
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
    return docs


def normalize_event(doc: dict) -> dict:
    slug = doc.get("slug") or doc.get("slug_en") or doc.get("slug_he")
    if not slug:
        slug = slugify_value(doc.get("canonicalUrl")) or slugify_value(doc.get("name"))
    title = extract_bilingual(doc, "name", fallback=doc.get("title"))
    description = extract_bilingual(doc, "description")
    summary = extract_bilingual(doc, "summary", fallback=description.get("en") or description.get("he"))
    starts_at = isoformat_or_none(doc.get("startsAt") or doc.get("date"))
    ends_at = isoformat_or_none(doc.get("endsAt") or doc.get("endDate"))
    lastmod_candidates = [
        doc.get("updatedAt"),
        doc.get("modifiedAt"),
        doc.get("lastModified"),
        doc.get("lastmod"),
        doc.get("createdAt"),
        doc.get("date"),
    ]
    lastmod = None
    for candidate in lastmod_candidates:
        lastmod = isoformat_or_none(candidate)
        if lastmod:
            break
    geo = extract_geo(doc)

    city_raw = doc.get("city") or {}
    if isinstance(city_raw, str):
        city_name = city_raw
        city_slug = slugify_value(city_raw)
        city_geo = geo
    elif isinstance(city_raw, dict):
        city_name = city_raw.get("name") or city_raw.get("title") or ""
        city_slug = city_raw.get("slug") or slugify_value(city_name)
        city_geo = extract_geo(city_raw)
    else:
        city_name = ""
        city_slug = None
        city_geo = {"lat": None, "lon": None, "address": "", "postalCode": ""}
    city = {
        "slug": city_slug or "",
        "name": extract_bilingual(city_raw if isinstance(city_raw, dict) else {}, "name", fallback=city_name),
        "geo": city_geo,
        "canonicalUrl": build_canonical("city", slug=city_slug or ""),
    }

    venue_raw = doc.get("venue") or {}
    if isinstance(venue_raw, str):
        venue_name = venue_raw
        venue_slug = slugify_value(venue_raw)
        venue_geo = geo
    elif isinstance(venue_raw, dict):
        venue_name = venue_raw.get("name") or venue_raw.get("title") or ""
        venue_slug = venue_raw.get("slug") or slugify_value(venue_name)
        venue_geo = extract_geo(venue_raw)
    else:
        venue_name = ""
        venue_slug = None
        venue_geo = {"lat": None, "lon": None, "address": "", "postalCode": ""}
    venue = {
        "slug": venue_slug or "",
        "name": extract_bilingual(venue_raw if isinstance(venue_raw, dict) else {}, "name", fallback=venue_name),
        "geo": venue_geo,
        "canonicalUrl": build_canonical("venue", slug=venue_slug or ""),
    }

    genres = []
    raw_genres = doc.get("genres") or []
    if isinstance(raw_genres, str):
        raw_genres = [raw_genres]
    for item in raw_genres:
        if isinstance(item, dict):
            name = item.get("name") or item.get("title") or ""
            slug_val = item.get("slug") or slugify_value(name)
            genres.append({
                "slug": slug_val or "",
                "name": extract_bilingual(item, "name", fallback=name),
                "canonicalUrl": build_canonical("genre", slug=slug_val or ""),
            })
        elif isinstance(item, str):
            slug_val = slugify_value(item)
            genres.append({
                "slug": slug_val or "",
                "name": {"he": item, "en": item},
                "canonicalUrl": build_canonical("genre", slug=slug_val or ""),
            })

    event_status = derive_status(doc)
    canonical_url = build_canonical("event", slug=slug or "")
    images = doc.get("images") or []
    if isinstance(images, str):
        images = [images]

    return {
        "slug": slug or "",
        "status": event_status,
        "title": title,
        "summary": summary,
        "description": description,
        "canonicalUrl": canonical_url,
        "lastmod": lastmod,
        "startsAt": starts_at,
        "endsAt": ends_at,
        "geo": geo,
        "city": city,
        "venue": venue,
        "genres": genres,
        "images": images,
        "language": doc.get("language") or {"primary": "he", "available": ["he", "en"]},
    }


def upcoming_events(events: list[dict]) -> list[dict]:
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    upcoming = []
    for event in events:
        status = derive_status(event)
        start = parse_datetime(event.get("startsAt") or event.get("date"))
        if status == "past":
            continue
        if start and start < now:
            continue
        upcoming.append(event)
    return upcoming


def aggregate_dimension(events: list[dict], key: str) -> list[dict]:
    aggregates: dict[str, dict] = {}
    for event in events:
        resource = event.get(key) or {}
        if not isinstance(resource, dict):
            continue
        slug = resource.get("slug") or ""
        if not slug:
            continue
        entry = aggregates.setdefault(slug, {
            "slug": slug,
            "name": resource.get("name", {"he": "", "en": ""}),
            "geo": resource.get("geo", {"lat": None, "lon": None, "address": "", "postalCode": ""}),
            "lastmod": event.get("lastmod"),
            "eventCount": 0,
        })
        entry["eventCount"] += 1
        if event.get("lastmod"):
            if not entry.get("lastmod") or event["lastmod"] > entry["lastmod"]:
                entry["lastmod"] = event["lastmod"]
    result = []
    for slug, payload in aggregates.items():
        payload["canonicalUrl"] = build_canonical(key[:-1] if key.endswith("s") else key, slug=slug)
        result.append(payload)
    result.sort(key=lambda item: item["slug"])
    return result


def serialize_events(include_past: bool = True) -> list[dict]:
    docs = all_events()
    serialized = [normalize_event(doc) for doc in docs]
    if not include_past:
        serialized = [event for event in serialized if event["status"] != "past"]
    serialized.sort(key=lambda event: (event.get("startsAt") or "", event.get("slug")))
    return serialized


def unique_dates(events: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for event in events:
        start = event.get("startsAt")
        if not start:
            continue
        day = start.split("T", 1)[0]
        entry = seen.setdefault(day, {"slug": day, "lastmod": event.get("lastmod"), "eventCount": 0})
        entry["eventCount"] += 1
        if event.get("lastmod") and event["lastmod"] > (entry.get("lastmod") or ""):
            entry["lastmod"] = event["lastmod"]
    results = []
    for slug, payload in seen.items():
        payload["canonicalUrl"] = build_canonical("date", slug=slug)
        results.append(payload)
    results.sort(key=lambda item: item["slug"])
    return results


def aggregate_genres(events: list[dict]) -> list[dict]:
    aggregates: dict[str, dict] = {}
    for event in events:
        for genre in event.get("genres", []):
            slug = genre.get("slug") or ""
            if not slug:
                continue
            entry = aggregates.setdefault(slug, {
                "slug": slug,
                "name": genre.get("name", {"he": "", "en": ""}),
                "canonicalUrl": genre.get("canonicalUrl", build_canonical("genre", slug=slug)),
                "lastmod": event.get("lastmod"),
                "eventCount": 0,
            })
            entry["eventCount"] += 1
            if event.get("lastmod") and event["lastmod"] > (entry.get("lastmod") or ""):
                entry["lastmod"] = event["lastmod"]
    items = list(aggregates.values())
    items.sort(key=lambda item: item["slug"])
    return items


def shard_events(events: list[dict], chunk_size: int = EVENT_SITEMAP_CHUNK) -> list[list[dict]]:
    if chunk_size <= 0:
        return [events]
    return [events[i : i + chunk_size] for i in range(0, len(events), chunk_size)] or [[]]


def format_rfc822(date_str: str | None) -> str | None:
    if not date_str:
        return None
    dt = parse_datetime(date_str)
    if not dt:
        return None
    return format_datetime(dt)


def format_ics(date_str: str | None) -> str | None:
    if not date_str:
        return None
    dt = parse_datetime(date_str)
    if not dt:
        return None
    return dt.strftime("%Y%m%dT%H%M%SZ")


def build_feed(events: list[dict], title: str, url: str, fmt: str) -> str:
    updated = None
    for event in events:
        if event.get("lastmod"):
            if not updated or event["lastmod"] > updated:
                updated = event["lastmod"]
    updated_rfc = format_rfc822(updated) or format_rfc822(datetime.utcnow().replace(tzinfo=timezone.utc).isoformat())

    if fmt == "rss":
        items = []
        for event in events:
            pub = format_rfc822(event.get("startsAt") or event.get("lastmod")) or updated_rfc
            description = event.get("summary", {}).get("en") or event.get("summary", {}).get("he")
            items.append(
                """<item><title>{title}</title><link>{link}</link><guid isPermaLink="true">{link}</guid><pubDate>{pub}</pubDate><description>{desc}</description></item>""".format(
                    title=(event.get("title", {}).get("en") or event.get("title", {}).get("he") or ""),
                    link=event.get("canonicalUrl"),
                    pub=pub,
                    desc=(description or ""),
                )
            )
        return (
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            f"<rss version=\"2.0\"><channel><title>{title}</title><link>{url}</link><lastBuildDate>{updated_rfc}</lastBuildDate>"
            + "".join(items)
            + "</channel></rss>"
        )
    if fmt == "atom":
        entries = []
        for event in events:
            updated_entry = event.get("lastmod") or updated or ""
            entries.append(
                """<entry><title>{title}</title><id>{link}</id><link href="{link}"/><updated>{updated}</updated><summary>{summary}</summary></entry>""".format(
                    title=(event.get("title", {}).get("en") or event.get("title", {}).get("he") or ""),
                    link=event.get("canonicalUrl"),
                    updated=event.get("lastmod") or updated or "",
                    summary=(event.get("summary", {}).get("en") or event.get("summary", {}).get("he") or ""),
                )
            )
        return (
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            f"<feed xmlns=\"http://www.w3.org/2005/Atom\"><title>{title}</title><id>{url}</id><updated>{updated or ''}</updated>"
            + "".join(entries)
            + "</feed>"
        )
    # default to json feed
    items = []
    for event in events:
        items.append(
            {
                "id": event.get("canonicalUrl"),
                "url": event.get("canonicalUrl"),
                "title": event.get("title", {}).get("en") or event.get("title", {}).get("he"),
                "summary": event.get("summary", {}).get("en") or event.get("summary", {}).get("he"),
                "date_published": event.get("startsAt"),
                "date_modified": event.get("lastmod"),
            }
        )
    feed = {
        "version": "https://jsonfeed.org/version/1.1",
        "title": title,
        "home_page_url": url,
        "feed_url": url,
        "items": items,
    }
    return json.dumps(feed)


def build_ics(events: list[dict], title: str) -> str:
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//parties247//Events//EN",
        f"X-WR-CALNAME:{title}",
    ]
    generated = format_ics(datetime.utcnow().replace(tzinfo=timezone.utc).isoformat())
    for event in events:
        uid = f"{event.get('slug')}@parties247"
        dtstart = format_ics(event.get("startsAt"))
        dtend = format_ics(event.get("endsAt"))
        dtstamp = format_ics(event.get("lastmod")) or generated
        location = event.get("venue", {}).get("name", {}).get("en") or event.get("venue", {}).get("name", {}).get("he")
        description = event.get("summary", {}).get("en") or event.get("summary", {}).get("he")
        lines.extend(
            [
                "BEGIN:VEVENT",
                f"UID:{uid}",
                f"DTSTAMP:{dtstamp}",
                f"DTSTART:{dtstart}" if dtstart else "",
                f"DTEND:{dtend}" if dtend else "",
                f"SUMMARY:{event.get('title', {}).get('en') or event.get('title', {}).get('he')}",
                f"DESCRIPTION:{description or ''}",
                f"LOCATION:{location or ''}",
                f"URL:{event.get('canonicalUrl')}",
                "END:VEVENT",
            ]
        )
    lines.append("END:VCALENDAR")
    return "\r\n".join(filter(None, lines)) + "\r\n"


def notify_indexers(urls: list[str]):
    unique = sorted({absolute_url(u) for u in urls if u})
    if not unique:
        return
    payload = {
        "urls": unique,
        "timestamp": datetime.utcnow().isoformat(),
    }
    record_setting_hit("indexing:lastPing", {"urls": unique, "timestamp": payload["timestamp"]})
    if INDEXNOW_KEY:
        indexnow_payload = {
            "host": urlparse(CANONICAL_BASE_URL).netloc,
            "key": INDEXNOW_KEY,
            "keyLocation": INDEXNOW_KEY_LOCATION or absolute_url("/indexnow.txt"),
            "urlList": unique,
        }
        try:
            requests.post(INDEXNOW_ENDPOINT, json=indexnow_payload, timeout=5)
        except Exception as exc:  # pragma: no cover - network best effort
            app.logger.warning(f"IndexNow ping failed: {exc}")
    sitemap_url = build_canonical("sitemap_index")
    for ping_url in (
        f"https://www.google.com/ping?sitemap={quote_plus(sitemap_url)}",
        f"https://www.bing.com/ping?sitemap={quote_plus(sitemap_url)}",
    ):
        try:
            requests.get(ping_url, timeout=5)
        except Exception as exc:  # pragma: no cover
            app.logger.warning(f"Search engine ping failed: {exc}")


def trigger_revalidation(paths: list[str]):
    if not REVALIDATE_ENDPOINT or not paths:
        return
    payload = {"paths": paths}
    if REVALIDATE_SECRET:
        payload["secret"] = REVALIDATE_SECRET
    try:
        requests.post(REVALIDATE_ENDPOINT, json=payload, timeout=5)
        record_setting_hit("indexing:lastRevalidate", {"paths": paths})
    except Exception as exc:  # pragma: no cover
        app.logger.warning(f"Revalidation webhook failed: {exc}")

# --- Security ---
JWT_SECRET = os.environ.get("JWT_SECRET_KEY", "")
ADMIN_HASH = os.environ.get("ADMIN_PASSWORD_HASH", "").encode()

def protect(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not JWT_SECRET:
            app.logger.error("JWT secret not configured; rejecting protected request.")
            return jsonify({"message": "Server misconfigured."}), 500
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1].strip()
            try:
                jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
                return f(*args, **kwargs)
            except jwt.ExpiredSignatureError:
                return jsonify({"message": "Token expired."}), 401
            except jwt.InvalidTokenError:
                return jsonify({"message": "Invalid token."}), 401
        return jsonify({"message": "Unauthorized."}), 401
    return decorated_function

def apply_security_headers(response):
    """Apply a minimal set of security headers on every response."""
    headers = getattr(response, "headers", None)
    if headers is None:
        return response
    headers.setdefault("X-Content-Type-Options", "nosniff")
    headers.setdefault("X-Frame-Options", "DENY")
    headers.setdefault("Referrer-Policy", "same-origin")
    headers.setdefault("X-XSS-Protection", "1; mode=block")
    return response


if hasattr(app, "after_request"):
    app.after_request(apply_security_headers)


@app.route("/api/health", methods=["GET"])
def health():
    """Lightweight health-check endpoint used by monitoring probes."""
    return jsonify({"status": "ok"}), 200


OPENAPI_TEMPLATE = {
    "openapi": "3.1.0",
    "info": {
        "title": "Parties247 API",
        "version": "1.0.0",
        "description": "REST API for reading party data and managing the Parties247 catalog.",
        "contact": {
            "name": "Parties247",
            "url": "https://parties247-website.vercel.app",
        },
    },
    "paths": {
        "/api/health": {
            "get": {
                "summary": "Health check",
                "description": "Returns a simple status payload indicating the API is responsive.",
                "responses": {
                    "200": {
                        "description": "Service is healthy.",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string", "example": "ok"}
                                    },
                                }
                            }
                        },
                    }
                },
            }
        },
        "/api/parties": {
            "get": {
                "summary": "List parties",
                "description": "Fetch the public parties currently tracked by Parties247.",
                "responses": {
                    "200": {
                        "description": "Array of party objects.",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/Party"},
                                }
                            }
                        },
                    },
                    "500": {
                        "description": "Unexpected error fetching parties.",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"},
                                        "error": {"type": "string"},
                                    },
                                }
                            }
                        },
                    },
                },
            }
        },
        "/api/events": {
            "get": {
                "summary": "Events feed",
                "description": "Return a denormalized events list used by downstream integrations.",
                "responses": {
                    "200": {
                        "description": "Events response payload.",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "generatedAt": {"type": "string", "format": "date-time"},
                                        "items": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/Event"},
                                        },
                                    },
                                }
                            }
                        },
                    }
                },
            }
        },
        "/api/cities": {
            "get": {
                "summary": "List cities",
                "description": "Aggregate the number of parties per city.",
                "responses": {
                    "200": {
                        "description": "List of cities with event counts.",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "generatedAt": {"type": "string", "format": "date-time"},
                                        "items": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/Dimension"},
                                        },
                                    },
                                }
                            }
                        },
                    }
                },
            }
        },
        "/api/venues": {
            "get": {
                "summary": "List venues",
                "description": "Aggregate the number of parties per venue.",
                "responses": {
                    "200": {
                        "description": "List of venues with event counts.",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "generatedAt": {"type": "string", "format": "date-time"},
                                        "items": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/Dimension"},
                                        },
                                    },
                                }
                            }
                        },
                    }
                },
            }
        },
        "/api/genres": {
            "get": {
                "summary": "List genres",
                "description": "Aggregate the number of parties per genre.",
                "responses": {
                    "200": {
                        "description": "List of genres with event counts.",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "generatedAt": {"type": "string", "format": "date-time"},
                                        "items": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/Dimension"},
                                        },
                                    },
                                }
                            }
                        },
                    }
                },
            }
        },
        "/api/tags": {
            "get": {
                "summary": "List tags",
                "description": "Return the curated tags available for parties.",
                "responses": {
                    "200": {
                        "description": "List of tags.",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/Tag"},
                                }
                            }
                        },
                    }
                },
            }
        },
        "/api/referral": {
            "get": {
                "summary": "Get referral configuration",
                "description": "Expose the configured default referral code used in external links.",
                "responses": {
                    "200": {
                        "description": "Referral configuration payload.",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "referral": {"type": "string", "nullable": True}
                                    },
                                }
                            }
                        },
                    }
                },
            }
        },
        "/api/admin/login": {
            "post": {
                "summary": "Obtain admin token",
                "description": "Authenticate with the admin password to receive a JWT for protected endpoints.",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {"password": {"type": "string"}},
                                "required": ["password"],
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "JWT issued.",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"token": {"type": "string"}},
                                }
                            }
                        },
                    },
                    "401": {
                        "description": "Invalid credentials.",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"message": {"type": "string"}},
                                }
                            }
                        },
                    },
                },
            }
        },
        "/api/admin/carousels": {
            "get": {
                "summary": "List carousels",
                "description": "Return the ordered list of carousels with their assigned parties. Requires admin token.",
                "security": [{"bearerAuth": []}],
                "responses": {
                    "200": {
                        "description": "Array of carousels.",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/Carousel"}
                                }
                            }
                        }
                    },
                    "500": {"description": "Error fetching carousels."}
                }
            },
            "post": {
                "summary": "Create a carousel",
                "description": "Create a new carousel at the end of the display order. Requires admin token.",
                "security": [{"bearerAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["title"],
                                "properties": {
                                    "title": {"type": "string"},
                                    "partyIds": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Party identifiers to include in the carousel."
                                    }
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "201": {
                        "description": "Carousel created.",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Carousel"}
                            }
                        }
                    },
                    "400": {"description": "Invalid payload."},
                    "500": {"description": "Error creating carousel."}
                }
            }
        },
        "/api/admin/carousels/{carouselId}": {
            "put": {
                "summary": "Update carousel info",
                "description": "Update carousel metadata such as title or display order. Requires admin token.",
                "security": [{"bearerAuth": []}],
                "parameters": [
                    {
                        "name": "carouselId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"}
                    }
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "order": {"type": "integer"}
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Carousel updated.",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"},
                                        "carousel": {"$ref": "#/components/schemas/Carousel"}
                                    }
                                }
                            }
                        }
                    },
                    "400": {"description": "Invalid payload."},
                    "404": {"description": "Carousel not found."},
                    "500": {"description": "Error updating carousel."}
                }
            },
            "delete": {
                "summary": "Delete a carousel",
                "description": "Remove a carousel by identifier. Requires admin token.",
                "security": [{"bearerAuth": []}],
                "parameters": [
                    {
                        "name": "carouselId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"}
                    }
                ],
                "responses": {
                    "200": {"description": "Carousel deleted."},
                    "404": {"description": "Carousel not found."},
                    "500": {"description": "Error deleting carousel."}
                }
            }
        },
        "/api/admin/carousels/{carouselId}/parties": {
            "get": {
                "summary": "List carousel parties",
                "description": "Return the parties currently assigned to the carousel in order. Requires admin token.",
                "security": [{"bearerAuth": []}],
                "parameters": [
                    {
                        "name": "carouselId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"}
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Array of party documents.",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/Party"}
                                }
                            }
                        }
                    },
                    "400": {"description": "Invalid carousel id."},
                    "404": {"description": "Carousel not found."},
                    "500": {"description": "Error fetching carousel parties."}
                }
            },
            "put": {
                "summary": "Update carousel parties",
                "description": "Replace the ordered party list for a carousel. Requires admin token.",
                "security": [{"bearerAuth": []}],
                "parameters": [
                    {
                        "name": "carouselId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"}
                    }
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["partyIds"],
                                "properties": {
                                    "partyIds": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Carousel parties updated.",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"},
                                        "carousel": {"$ref": "#/components/schemas/Carousel"}
                                    }
                                }
                            }
                        }
                    },
                    "400": {"description": "Invalid payload or missing parties."},
                    "404": {"description": "Carousel not found."},
                    "500": {"description": "Error updating carousel parties."}
                }
            }
        },
        "/api/admin/sections": {
            "post": {
                "summary": "Import parties into a carousel",
                "description": "Scrape a curated page for party links, ensure each party exists, and create a carousel using them. Requires admin token.",
                "security": [{"bearerAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["url"],
                                "properties": {
                                    "url": {"type": "string", "format": "uri"},
                                    "carouselName": {"type": "string"},
                                    "title": {"type": "string"}
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "201": {
                        "description": "Carousel created from section.",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"},
                                        "carousel": {"$ref": "#/components/schemas/Carousel"},
                                        "partyCount": {"type": "integer"},
                                        "warnings": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "url": {"type": "string"},
                                                    "error": {"type": "string"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "400": {"description": "Invalid payload."},
                    "404": {"description": "No parties found at the provided URL."},
                    "502": {"description": "Error fetching the source URL."},
                    "500": {"description": "Server error while importing the section."}
                }
            }
        },
        "/api/admin/add-party": {
            "post": {
                "summary": "Add a party",
                "description": "Scrape and persist a new party. Requires admin token.",
                "security": [{"bearerAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {"url": {"type": "string", "format": "uri"}},
                                "required": ["url"],
                            }
                        }
                    },
                },
                "responses": {
                    "201": {
                        "description": "Party created.",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"},
                                        "party": {"$ref": "#/components/schemas/Party"},
                                    },
                                }
                            }
                        },
                    },
                    "400": {"description": "Invalid payload."},
                    "409": {"description": "Party already exists."},
                    "500": {"description": "Server error."},
                },
            }
        },
        "/api/admin/delete-party/{partyId}": {
            "delete": {
                "summary": "Delete a party",
                "description": "Remove an existing party by its identifier. Requires admin token.",
                "security": [{"bearerAuth": []}],
                "parameters": [
                    {
                        "name": "partyId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "MongoDB ObjectId of the party to delete.",
                    }
                ],
                "responses": {
                    "200": {"description": "Party deleted."},
                    "404": {"description": "Party not found."},
                    "500": {"description": "Error deleting party."},
                },
            }
        },
        "/api/admin/update-party/{partyId}": {
            "put": {
                "summary": "Update a party",
                "description": "Update select fields on an existing party. Requires admin token.",
                "security": [{"bearerAuth": []}],
                "parameters": [
                    {
                        "name": "partyId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "description": "Any subset of party fields to update.",
                                "additionalProperties": True,
                            }
                        }
                    },
                },
                "responses": {
                    "200": {"description": "Party updated."},
                    "400": {"description": "Invalid payload."},
                    "404": {"description": "Party not found."},
                    "500": {"description": "Server error."},
                },
            }
        },
    },
    "components": {
        "securitySchemes": {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
            }
        },
        "schemas": {
            "Party": {
                "type": "object",
                "properties": {
                    "_id": {"type": "string"},
                    "name": {"type": "string"},
                    "date": {"type": "string"},
                    "goOutUrl": {"type": "string", "format": "uri"},
                    "originalUrl": {"type": "string", "format": "uri"},
                    "startsAt": {"type": "string", "format": "date-time"},
                    "endsAt": {"type": "string", "format": "date-time"},
                    "city": {"type": "string"},
                    "venue": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": True,
            },
            "Event": {
                "type": "object",
                "description": "Normalized event payload consumed by listings.",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "startsAt": {"type": "string", "format": "date-time"},
                    "endsAt": {"type": "string", "format": "date-time"},
                    "canonicalUrl": {"type": "string", "format": "uri"},
                },
                "additionalProperties": True,
            },
            "Dimension": {
                "type": "object",
                "properties": {
                    "slug": {"type": "string"},
                    "name": {"type": "string"},
                    "eventCount": {"type": "integer", "minimum": 0},
                },
                "additionalProperties": True,
            },
            "Tag": {
                "type": "object",
                "properties": {
                    "slug": {"type": "string"},
                    "name": {"type": "string"},
                    "order": {"type": "integer"},
                },
                "required": ["slug", "name"],
            },
            "Carousel": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "order": {"type": "integer"},
                    "partyIds": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "additionalProperties": True,
            },
        },
    },
}


def build_openapi_document():
    """Generate the OpenAPI document, hydrating runtime server information."""
    document = copy.deepcopy(OPENAPI_TEMPLATE)
    server_url = (request.host_url or "").rstrip("/")
    if not server_url:
        server_url = "http://localhost"
    document["servers"] = [{"url": server_url}]
    return document


@app.route("/openapi.json", methods=["GET"])
def openapi_json():
    """Expose the OpenAPI definition for API consumers."""
    return jsonify(build_openapi_document()), 200


@app.route("/docs", methods=["GET"])
def docs_page():
    """Render Swagger UI backed by the generated OpenAPI definition."""
    spec_url = url_for("openapi_json", _external=True)
    html = f"""<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\">
    <title>Parties247 API Documentation</title>
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.10.5/swagger-ui.min.css\" crossorigin=\"anonymous\" />
    <style>
      body {{ margin: 0; background-color: #f5f5f5; }}
      #swagger-ui {{ box-shadow: none; }}
    </style>
  </head>
  <body>
    <div id=\"swagger-ui\"></div>
    <noscript>
      <p>JavaScript is required to view the interactive documentation. Download the <a href=\"{spec_url}\">OpenAPI JSON</a>.</p>
    </noscript>
    <script src=\"https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.10.5/swagger-ui-bundle.min.js\" crossorigin=\"anonymous\"></script>
    <script src=\"https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.10.5/swagger-ui-standalone-preset.min.js\" crossorigin=\"anonymous\"></script>
    <script>
      window.onload = () => {{
        SwaggerUIBundle({{
          url: "{spec_url}",
          dom_id: '#swagger-ui',
          presets: [SwaggerUIBundle.presets.apis, SwaggerUIStandalonePreset],
          layout: 'BaseLayout'
        }});
      }};
    </script>
  </body>
</html>"""
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    data = request.get_json(silent=True) or {}
    password = (data.get("password") or "").encode()
    if not ADMIN_HASH or not JWT_SECRET:
        app.logger.error("Admin credentials or JWT secret missing; refusing login attempt.")
        return jsonify({"message": "Server misconfigured."}), 500
    try:
        hashed_attempt = bcrypt.hashpw(password, ADMIN_HASH)
        if hmac.compare_digest(hashed_attempt, ADMIN_HASH):
            exp = datetime.utcnow() + timedelta(minutes=15)
            token = jwt.encode({"exp": exp}, JWT_SECRET, algorithm="HS256")
            if isinstance(token, bytes):
                token = token.decode()
            return jsonify({"token": token}), 200
    except ValueError:
        pass
    return jsonify({"message": "Invalid credentials."}), 401

# --- Tag helpers + schemas ---
def slugify_tag(name: str) -> str:
    return "-".join((name or "").strip().lower().split())

def ensure_tags_seeded():
    """Backfill tags collection from parties if empty."""
    if tags_collection.count_documents({}) == 0:
        distinct_tags = [t for t in (parties_collection.distinct("tags") or []) if isinstance(t, str) and t.strip()]
        ops = []
        for idx, t in enumerate(sorted(set(distinct_tags), key=lambda x: x.lower())):
            ops.append(tags_collection.update_one(
                {"slug": slugify_tag(t)},
                {"$setOnInsert": {"slug": slugify_tag(t), "name": t, "order": idx}},
                upsert=True
            ))
        if ops:
            tags_collection.bulk_write(ops)

class TagOrderRequest(BaseModel):
    slug: str
    newIndex: int
    class Config:
        extra = "forbid"

class TagRenameRequest(BaseModel):
    oldName: str
    newName: str
    class Config:
        extra = "forbid"

# --- Referral schemas ---
class ReferralUpdateSchema(BaseModel):
    code: str
    class Config:
        extra = "forbid"

# --- Parties schemas ---
class AddPartyRequest(BaseModel):
    url: str
    class Config:
        extra = "forbid"

class PartyUpdateSchema(BaseModel):
    name: str | None = None
    imageUrl: str | None = None
    date: str | None = None
    location: str | None = None
    description: str | None = None
    region: str | None = None
    musicType: str | None = None
    eventType: str | None = None
    age: str | None = None
    tags: list[str] | None = None
    originalUrl: str | None = None
    canonicalUrl: str | None = None
    goOutUrl: str | None = None
    referralCode: str | None = None
    class Config:
        extra = "forbid"

# --- Carousel schemas ---
class CarouselCreateSchema(BaseModel):
    title: str
    partyIds: list[str] = []
    class Config:
        extra = "forbid"

class CarouselUpdateSchema(BaseModel):
    title: str | None = None
    order: int | None = None
    class Config:
        extra = "forbid"


class CarouselPartiesUpdateSchema(BaseModel):
    partyIds: list[str]
    class Config:
        extra = "forbid"

class CarouselReorderSchema(BaseModel):
    orderedIds: list[str]
    class Config:
        extra = "forbid"


class CarouselImportRequest(BaseModel):
    url: str
    carouselName: str | None = None
    title: str | None = None

    class Config:
        extra = "forbid"

# --- Classification helpers ---
def get_region(location: str) -> str:
    south_keywords = ["באר שבע", "אילת", "אשדוד", "אשקלון", "דרום"]
    north_keywords = ["חיפה", "טבריה", "צפון", "קריות", "כרמיאל", "עכו"]
    center_keywords = ["תל אביב", "ירושלים", "ראשון לציון", "הרצליה", "נתניה", "מרכז", "י-ם", "tlv"]
    loc = (location or "").lower()
    if any(k in loc for k in south_keywords):
        return "דרום"
    if any(k in loc for k in north_keywords):
        return "צפון"
    if any(k in loc for k in center_keywords):
        return "מרכז"
    return "לא ידוע"

def get_music_type(text: str) -> str:
    techno_keywords = ["טכנו", "techno", "after", "אפטר", "house", "האוס", "electronic", "אלקטרונית"]
    trance_keywords = ["טראנס", "trance", "פסיי", "psy-trance", "psytrance"]
    mainstream_keywords = ["מיינסטרים", "mainstream", "היפ הופ", "hip hop", "רגאטון", "reggaeton", "pop", "פופ"]
    txt = (text or "").lower()
    if any(k in txt for k in techno_keywords):
        return "טכנו"
    if any(k in txt for k in trance_keywords):
        return "טראנס"
    if any(k in txt for k in mainstream_keywords):
        return "מיינסטרים"
    return "אחר"

def get_event_type(text: str) -> str:
    festival_keywords = ["פסטיבל", "festival"]
    nature_keywords = ["טבע", "nature", "יער", "forest", "חוף", "beach", "open air", "בחוץ"]
    club_keywords = ["מועדון", "club", "גגרין", "בלוק", "האומן 17", "gagarin", "block", "haoman 17", "rooftop", "גג"]
    txt = (text or "").lower()
    if any(k in txt for k in festival_keywords):
        return "פסטיבל"
    if any(k in txt for k in nature_keywords):
        return "מסיבת טבע"
    if any(k in txt for k in club_keywords):
        return "מסיבת מועדון"
    return "אחר"

def get_age(text: str, minimum_age: int) -> str:
    if minimum_age >= 21:
        return "21+"
    if minimum_age >= 18:
        return "18+"
    if "נוער" in (text or "").lower():
        return "נוער"
    if minimum_age > 0:
        return "18+"
    return "כל הגילאים"

def get_tags(text: str, location: str) -> list:
    tags = []
    tag_map = {
        "אלכוהול חופשי": ["אלכוהול חופשי", "free alcohol", "בר חופשי", "free bar"],
        "בחוץ": ["open air", "בחוץ", "טבע", "חוף", "יער", "rooftop", "גג"],
        "אילת": ["אילת", "eilat"],
        "תל אביב": ["תל אביב", "tel aviv", "tlv"],
    }
    combined_text = f"{text or ''} {location or ''}".lower()
    for tag, keywords in tag_map.items():
        if any(keyword in combined_text for keyword in keywords):
            tags.append(tag)
    return list(set(tags))

# --- Scraper ---
def scrape_party_details(url: str):
    if not is_url_allowed(url):
        raise ValueError("URL is not allowed.")
    app.logger.info(f"[SCRAPER] start {url}")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=15)
        app.logger.info(f"[SCRAPER] status {response.status_code}")
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        script_tag = soup.find("script", {"id": "__NEXT_DATA__"})
        if not script_tag:
            raise ValueError("Could not find party data script (__NEXT_DATA__).")

        json_data = json.loads(script_tag.string)
        event_data = json_data.get("props", {}).get("pageProps", {}).get("event")
        if not event_data:
            raise ValueError("Event data not in expected format inside JSON.")

        image_path = ""
        if event_data.get("CoverImage") and event_data["CoverImage"].get("Url"):
            image_path = event_data["CoverImage"]["Url"]
        elif event_data.get("WhatsappImage") and event_data["WhatsappImage"].get("Url"):
            image_path = event_data["WhatsappImage"]["Url"]

        if image_path:
            cover_image_path = image_path.replace("_whatsappImage.jpg", "_coverImage.jpg")
            image_url = f"https://d15q6k8l9pfut7.cloudfront.net/{cover_image_path}"
        else:
            og_image_tag = soup.find("meta", {"property": "og:image"})
            og_image_url = og_image_tag["content"] if og_image_tag else ""
            if og_image_url:
                image_url = og_image_url.replace("_whatsappImage.jpg", "_coverImage.jpg")
            else:
                raise ValueError("Could not find party image URL.")

        description = event_data.get("Description", "")
        cleaned_desc = " ".join(list(filter(None, description.split("\n")))[:3]).strip()
        if len(cleaned_desc) > 250:
            cleaned_desc = cleaned_desc[:247] + "..."

        full_text = f"{event_data.get('Title', '')} {description}"
        location = event_data.get("Adress", "")
        canonical = normalize_url(url)
        go_out = normalized_or_none_for_dedupe(url)

        party_details = {
            "name": event_data.get("Title"),
            "imageUrl": image_url,
            "date": event_data.get("StartingDate"),
            "location": location,
            "description": cleaned_desc or "No description available.",
            "region": get_region(location),
            "musicType": get_music_type(full_text),
            "eventType": get_event_type(full_text),
            "age": get_age(full_text, event_data.get("MinimumAge", 0)),
            "tags": get_tags(full_text, location),
            "originalUrl": url,
            "canonicalUrl": canonical,
        }
        if go_out:
            party_details["goOutUrl"] = go_out

        if not all([party_details["name"], party_details["date"], party_details["location"]]):
            raise ValueError("Scraped data is missing critical fields.")
        return party_details

    except requests.exceptions.RequestException as e:
        app.logger.error(f"[SCRAPER] HTTP error: {e}")
        raise
    except Exception as e:
        app.logger.error(f"[SCRAPER] error: {e}")
        raise

# --- Routes ---

# Static generation data APIs


def event_related_paths(event: dict) -> list[str]:
    paths = {f"/event/{event.get('slug') or ''}"}
    city_slug = event.get("city", {}).get("slug")
    if city_slug:
        paths.add(f"/city/{city_slug}")
    venue_slug = event.get("venue", {}).get("slug")
    if venue_slug:
        paths.add(f"/venue/{venue_slug}")
    for genre in event.get("genres", []):
        if genre.get("slug"):
            paths.add(f"/genre/{genre['slug']}")
    if event.get("startsAt"):
        day = event["startsAt"].split("T", 1)[0]
        paths.add(f"/date/{day}")
    return sorted(paths)


@app.route("/api/events", methods=["GET"])
@limiter.limit("100 per minute")
def list_events_api():
    events = serialize_events(include_past=True)
    payload = {
        "generatedAt": isoformat_or_none(datetime.utcnow().replace(tzinfo=timezone.utc)),
        "items": events,
    }
    return json_response(payload, cache_seconds=EVENT_CACHE_SECONDS)


@app.route("/api/cities", methods=["GET"])
@limiter.limit("100 per minute")
def list_cities_api():
    events = serialize_events(include_past=False)
    items = [item for item in aggregate_dimension(events, "city") if item.get("eventCount")]
    payload = {
        "generatedAt": isoformat_or_none(datetime.utcnow().replace(tzinfo=timezone.utc)),
        "items": items,
    }
    return json_response(payload, cache_seconds=LIST_CACHE_SECONDS)


@app.route("/api/venues", methods=["GET"])
@limiter.limit("100 per minute")
def list_venues_api():
    events = serialize_events(include_past=False)
    items = [item for item in aggregate_dimension(events, "venue") if item.get("eventCount")]
    payload = {
        "generatedAt": isoformat_or_none(datetime.utcnow().replace(tzinfo=timezone.utc)),
        "items": items,
    }
    return json_response(payload, cache_seconds=LIST_CACHE_SECONDS)


@app.route("/api/genres", methods=["GET"])
@limiter.limit("100 per minute")
def list_genres_api():
    events = serialize_events(include_past=False)
    items = [item for item in aggregate_genres(events) if item.get("eventCount")]
    payload = {
        "generatedAt": isoformat_or_none(datetime.utcnow().replace(tzinfo=timezone.utc)),
        "items": items,
    }
    return json_response(payload, cache_seconds=LIST_CACHE_SECONDS)


# Parties
@app.route("/api/admin/add-party", methods=["POST"])
@limiter.limit("100 per minute")
@protect
def add_party():
    payload = request.get_json(silent=True) or {}
    try:
        req = AddPartyRequest(**payload)
    except ValidationError as ve:
        app.logger.warning(f"[VALIDATION] {ve}")
        return jsonify({"message": "Invalid request", "errors": ve.errors()}), 400
    url = req.url
    if not is_url_allowed(url):
        return jsonify({"message": "URL is not allowed."}), 400
    try:
        party_data = scrape_party_details(url)
        referral = default_referral_code()
        apply_default_referral(party_data, referral)
        party_data.setdefault("slug", slugify_value(party_data.get("name")))
        canonical = party_data["canonicalUrl"]
        res = parties_collection.update_one(
            {"$or": [{"canonicalUrl": canonical}, {"goOutUrl": canonical}]},
            {"$setOnInsert": party_data},
            upsert=True,
        )
        if res.matched_count == 1 and res.upserted_id is None:
            doc = parties_collection.find_one(
                {"$or": [{"canonicalUrl": canonical}, {"goOutUrl": canonical}]},
                {"_id": 1}
            )
            return jsonify({"message": "This party has already been added.", "id": str(doc["_id"])}), 409
        party_data["_id"] = str(res.upserted_id)
        event_view = normalize_event(party_data)
        notify_indexers([event_view.get("canonicalUrl")])
        trigger_revalidation(event_related_paths(event_view))
        return jsonify({"message": "Party added successfully!", "party": party_data}), 201
    except errors.DuplicateKeyError as e:
        app.logger.warning(f"[DB] DuplicateKeyError: {getattr(e, 'details', None)}")
        return jsonify({"message": "This party has already been added."}), 409
    except Exception as e:
        app.logger.error(f"[DB] {str(e)}")
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500

@app.route("/api/admin/delete-party/<party_id>", methods=["DELETE"])
@limiter.limit("100 per minute")
@protect
def delete_party(party_id):
    try:
        obj_id = ObjectId(party_id)
        existing = parties_collection.find_one({"_id": obj_id}) or {}
        result = parties_collection.delete_one({"_id": obj_id})
        if result.deleted_count == 0:
            return jsonify({"message": "Party not found."}), 404
        event_view = normalize_event(existing)
        notify_indexers([event_view.get("canonicalUrl")])
        trigger_revalidation(event_related_paths(event_view))
        return jsonify({"message": "Party deleted successfully!"}), 200
    except Exception as e:
        return jsonify({"message": "Error deleting party", "error": str(e)}), 500

@app.route("/api/admin/update-party/<party_id>", methods=["PUT"])
@limiter.limit("100 per minute")
@protect
def update_party(party_id):
    try:
        payload = request.get_json(silent=True) or {}
        obj_id = ObjectId(party_id)
        update = PartyUpdateSchema(**payload)
        data = update.dict(exclude_unset=True)
        if not data:
            return jsonify({"message": "No valid fields provided."}), 400
        result = parties_collection.update_one({"_id": obj_id}, {"$set": data})
        if result.matched_count == 0:
            return jsonify({"message": "Party not found."}), 404
        updated_doc = parties_collection.find_one({"_id": obj_id}) or {}
        event_view = normalize_event(updated_doc)
        notify_indexers([event_view.get("canonicalUrl")])
        trigger_revalidation(event_related_paths(event_view))
        return jsonify({"message": "Party updated successfully!"}), 200
    except ValidationError as ve:
        app.logger.warning(f"[VALIDATION] {ve}")
        return jsonify({"message": "Invalid party data", "errors": ve.errors()}), 400
    except Exception as e:
        app.logger.error(f"Error updating party {party_id}: {e}")
        return jsonify({"message": "An error occurred", "error": str(e)}), 500

@app.route("/api/parties", methods=["GET"])
@limiter.limit("100 per minute")
def get_parties():
    try:
        items = []
        referral = default_referral_code()
        for party in parties_collection.find().sort("date", 1):
            party["_id"] = str(party["_id"])
            apply_default_referral(party, referral)
            items.append(party)
        return jsonify(items), 200
    except Exception as e:
        return jsonify({"message": "Error fetching parties", "error": str(e)}), 500

# --- Carousels ---


def serialize_carousel(doc: dict) -> dict:
    data = dict(doc or {})
    raw_id = data.pop("_id", None)
    if raw_id is not None:
        data["id"] = str(raw_id)
    elif "id" in data and data["id"] is not None:
        data["id"] = str(data["id"])
    party_ids = []
    for pid in data.get("partyIds", []):
        if isinstance(pid, ObjectId):
            party_ids.append(str(pid))
        elif isinstance(pid, str):
            party_ids.append(pid)
    data["partyIds"] = party_ids
    return data


def extract_event_urls_from_page(source_url: str, html: str) -> list[str]:
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return []
    discovered: list[str] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a"):
        href = getattr(anchor, "get", lambda *args, **kwargs: None)("href")
        if not href:
            continue
        absolute = urljoin(source_url, href)
        normalized = normalize_url(absolute)
        if "/event/" not in urlparse(normalized).path:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        discovered.append(normalized)
    if discovered:
        return discovered
    script_tag = soup.find("script", {"id": "__NEXT_DATA__"})
    if not script_tag or not getattr(script_tag, "string", None):
        return []
    try:
        data = json.loads(script_tag.string)
    except Exception:
        return []

    ordered: list[str] = []
    seen = set()

    def _walk(node):
        if isinstance(node, dict):
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)
        elif isinstance(node, str) and "/event/" in node:
            absolute_url = urljoin(source_url, node)
            normalized_url = normalize_url(absolute_url)
            if "/event/" not in urlparse(normalized_url).path:
                return
            if normalized_url in seen:
                return
            seen.add(normalized_url)
            ordered.append(normalized_url)

    _walk(data)
    return ordered


def ensure_party_for_url(event_url: str, referral: str | None) -> tuple[dict | None, bool]:
    if parties_collection is None:
        return None, False
    canonical = normalize_url(event_url)
    query = {"$or": [{"canonicalUrl": canonical}, {"goOutUrl": canonical}]}
    existing = None
    try:
        existing = parties_collection.find_one(query)
    except Exception as exc:
        app.logger.error(f"Failed to query party for {event_url}: {exc}")
    if existing:
        if referral:
            mutated = dict(existing)
            apply_default_referral(mutated, referral)
            updates = {}
            for key in ("goOutUrl", "originalUrl", "referralCode"):
                if mutated.get(key) != existing.get(key):
                    updates[key] = mutated.get(key)
            if updates:
                try:
                    parties_collection.update_one({"_id": existing.get("_id")}, {"$set": updates})
                    existing.update(updates)
                except Exception as exc:
                    app.logger.warning(
                        f"Failed to update referral for party {existing.get('_id')}: {exc}"
                    )
        return existing, False

    party_data = scrape_party_details(event_url)
    party_data.setdefault("canonicalUrl", canonical)
    if not party_data.get("originalUrl"):
        party_data["originalUrl"] = event_url
    if not party_data.get("goOutUrl"):
        party_data["goOutUrl"] = canonical
    apply_default_referral(party_data, referral)
    party_data.setdefault("slug", slugify_value(party_data.get("name")))
    try:
        result = parties_collection.update_one(query, {"$setOnInsert": party_data}, upsert=True)
    except Exception as exc:
        app.logger.error(f"Failed to upsert party from {event_url}: {exc}")
        raise
    upserted_id = getattr(result, "upserted_id", None)
    if upserted_id is None:
        try:
            existing = parties_collection.find_one(query)
        except Exception as exc:
            app.logger.error(f"Failed to load existing party for {event_url}: {exc}")
            existing = None
        return existing, False

    party_data["_id"] = upserted_id
    event_view = normalize_event(party_data)
    notify_indexers([event_view.get("canonicalUrl")])
    trigger_revalidation(event_related_paths(event_view))
    return party_data, True


@app.route("/api/admin/carousels", methods=["POST"])
@limiter.limit("10 per minute")
@protect
def add_carousel():
    payload = request.get_json(silent=True) or {}
    try:
        carousel = CarouselCreateSchema(**payload)
    except ValidationError as ve:
        app.logger.warning(f"[VALIDATION] {ve}")
        return jsonify({"message": "Invalid carousel data", "errors": ve.errors()}), 400
    try:
        doc = carousel.dict()
        # assign at end
        last = carousels_collection.find().sort("order", -1).limit(1)
        last_order = next(last, {}).get("order", -1)
        doc["order"] = last_order + 1
        result = carousels_collection.insert_one(doc)
        doc["_id"] = result.inserted_id
        return jsonify(serialize_carousel(doc)), 201
    except Exception as e:
        return jsonify({"message": "Error adding carousel", "error": str(e)}), 500

@app.route("/api/admin/carousels", methods=["GET"])
@limiter.limit("30 per minute")
@protect
def list_admin_carousels():
    try:
        items = [serialize_carousel(doc) for doc in carousels_collection.find().sort("order", 1)]
        return jsonify(items), 200
    except Exception as e:
        return jsonify({"message": "Error fetching carousels", "error": str(e)}), 500


@app.route("/api/admin/carousels/<carousel_id>", methods=["PUT"])
@protect
def update_carousel(carousel_id):
    payload = request.get_json(silent=True) or {}
    try:
        update = CarouselUpdateSchema(**payload)
    except ValidationError as ve:
        app.logger.warning(f"[VALIDATION] {ve}")
        return jsonify({"message": "Invalid carousel data", "errors": ve.errors()}), 400
    update_data = update.dict(exclude_unset=True)
    if not update_data:
        return jsonify({"message": "No valid fields provided."}), 400
    try:
        obj_id = ObjectId(carousel_id)
        result = carousels_collection.update_one({"_id": obj_id}, {"$set": update_data})
        if result.matched_count == 0:
            return jsonify({"message": "Carousel not found"}), 404
        updated_doc = carousels_collection.find_one({"_id": obj_id}) or {}
        return (
            jsonify({"message": "Carousel updated successfully!", "carousel": serialize_carousel(updated_doc)}),
            200,
        )
    except Exception as e:
        return jsonify({"message": "Error updating carousel", "error": str(e)}), 500


@app.route("/api/admin/carousels/<carousel_id>/parties", methods=["GET"])
@limiter.limit("60 per minute")
@protect
def get_carousel_parties(carousel_id):
    try:
        obj_id = ObjectId(carousel_id)
    except Exception:
        return jsonify({"message": "Invalid carousel id."}), 400
    try:
        carousel = carousels_collection.find_one({"_id": obj_id})
        if not carousel:
            return jsonify({"message": "Carousel not found"}), 404
        party_ids = carousel.get("partyIds") or []
        resolved_ids: list[tuple[str, ObjectId]] = []
        for pid in party_ids:
            try:
                resolved_ids.append((str(pid), ObjectId(pid)))
            except Exception:
                return jsonify({"message": "Carousel contains invalid party identifiers."}), 400
        if not resolved_ids:
            return jsonify([]), 200
        lookup_ids = [oid for _, oid in resolved_ids]
        parties = list(parties_collection.find({"_id": {"$in": lookup_ids}}))
        parties_by_id = {str(doc.get("_id")): doc for doc in parties}
        items = []
        for original_id, oid in resolved_ids:
            doc = parties_by_id.get(str(oid))
            if not doc:
                continue
            payload = dict(doc)
            payload["_id"] = str(payload.get("_id"))
            items.append(payload)
        return jsonify(items), 200
    except Exception as e:
        return jsonify({"message": "Error fetching carousel parties", "error": str(e)}), 500


@app.route("/api/admin/carousels/<carousel_id>/parties", methods=["PUT"])
@protect
def update_carousel_parties(carousel_id):
    payload = request.get_json(silent=True) or {}
    try:
        update = CarouselPartiesUpdateSchema(**payload)
    except ValidationError as ve:
        app.logger.warning(f"[VALIDATION] {ve}")
        return jsonify({"message": "Invalid carousel data", "errors": ve.errors()}), 400
    normalized_ids: list[str] = []
    lookup_ids: list[ObjectId] = []
    for pid in update.partyIds:
        try:
            oid = ObjectId(pid)
        except Exception:
            return jsonify({"message": "partyIds must contain valid identifiers."}), 400
        normalized_ids.append(str(oid))
        lookup_ids.append(oid)
    try:
        obj_id = ObjectId(carousel_id)
    except Exception:
        return jsonify({"message": "Invalid carousel id."}), 400
    try:
        if lookup_ids:
            existing_ids = {str(doc.get("_id")) for doc in parties_collection.find({"_id": {"$in": lookup_ids}}, {"_id": 1})}
            missing = [pid for pid in normalized_ids if pid not in existing_ids]
            if missing:
                return jsonify({"message": "Some parties do not exist.", "missing": missing}), 400
        result = carousels_collection.update_one({"_id": obj_id}, {"$set": {"partyIds": normalized_ids}})
        if result.matched_count == 0:
            return jsonify({"message": "Carousel not found"}), 404
        updated_doc = carousels_collection.find_one({"_id": obj_id}) or {}
        return (
            jsonify({"message": "Carousel parties updated successfully!", "carousel": serialize_carousel(updated_doc)}),
            200,
        )
    except Exception as e:
        return jsonify({"message": "Error updating carousel parties", "error": str(e)}), 500


@app.route("/api/admin/carousels/<carousel_id>", methods=["DELETE"])
@protect
def delete_carousel(carousel_id):
    try:
        obj_id = ObjectId(carousel_id)
        result = carousels_collection.delete_one({"_id": obj_id})
        if result.deleted_count == 0:
            return jsonify({"message": "Carousel not found."}), 404
        return jsonify({"message": "Carousel deleted successfully!"}), 200
    except Exception as e:
        return jsonify({"message": "Error deleting carousel", "error": str(e)}), 500

@app.route("/api/carousels", methods=["GET"])
@limiter.limit("100 per minute")
def get_carousels():
    try:
        items = [serialize_carousel(carousel) for carousel in carousels_collection.find().sort("order", 1)]
        return jsonify(items), 200
    except Exception as e:
        return jsonify({"message": "Error fetching carousels", "error": str(e)}), 500

@app.route("/api/admin/carousels/reorder", methods=["POST"])
@protect
def reorder_carousels():
    payload = request.get_json(silent=True) or {}
    try:
        req = CarouselReorderSchema(**payload)
    except ValidationError as ve:
        return jsonify({"message": "Invalid payload", "errors": ve.errors()}), 400
    ids = []
    try:
        ids = [ObjectId(i) for i in req.orderedIds]
    except Exception:
        return jsonify({"message": "orderedIds must be valid ObjectIds"}), 400
    existing = list(carousels_collection.find({"_id": {"$in": ids}}, {"_id": 1}))
    if len(existing) != len(ids):
        return jsonify({"message": "Some carousel IDs do not exist."}), 400
    ids = [ObjectId(i) for i in req.orderedIds]
    ops = [UpdateOne({"_id": oid}, {"$set": {"order": idx}}) for idx, oid in enumerate(ids)]
    if not ops:
        return jsonify({"message": "No items to reorder"}), 400
    try:
        carousels_collection.bulk_write(ops, ordered=True)
        return jsonify({"message": "Reordered"}), 200
    except Exception as e:
        return jsonify({"message": "Error reordering carousels", "error": str(e)}), 500
# --- Tags APIs ---
@app.route("/api/tags", methods=["GET"])
@limiter.limit("100 per minute")
def list_tags():
    try:
        ensure_tags_seeded()
        docs = list(tags_collection.find().sort([("order", 1), ("name", 1)]))
        items = [{"slug": d["slug"], "name": d.get("name", d["slug"]), "order": d.get("order", 0)} for d in docs]
        return jsonify(items), 200
    except Exception as e:
        return jsonify({"message": "Error fetching tags", "error": str(e)}), 500

@app.route("/api/admin/tags/order", methods=["POST"])
@limiter.limit("100 per minute")
@protect
def reorder_tag():
    payload = request.get_json(silent=True) or {}
    try:
        req = TagOrderRequest(**payload)
    except ValidationError as ve:
        return jsonify({"message": "Invalid payload", "errors": ve.errors()}), 400
    ensure_tags_seeded()
    docs = list(tags_collection.find().sort([("order", 1), ("name", 1)]))
    slugs = [d["slug"] for d in docs]
    if req.slug not in slugs:
        return jsonify({"message": "Tag not found"}), 404
    old_index = slugs.index(req.slug)
    new_index = max(0, min(req.newIndex, len(slugs) - 1))
    if new_index == old_index:
        return jsonify({"message": "No change"}), 200
    # reorder list
    item = docs.pop(old_index)
    docs.insert(new_index, item)
    # write back sequential order
    ops = []
    for idx, d in enumerate(docs):
        ops.append(tags_collection.update_one({"_id": d["_id"]}, {"$set": {"order": idx}}))
    try:
        tags_collection.bulk_write(ops)
        return jsonify({"message": "Tag order updated"}), 200
    except Exception as e:
        return jsonify({"message": "Error updating tag order", "error": str(e)}), 500

@app.route("/api/admin/tags/rename", methods=["POST"])
@limiter.limit("100 per minute")
@protect
def rename_tag():
    payload = request.get_json(silent=True) or {}
    tag_id_raw = (payload.get("tagId") or "").strip()
    new_name = (payload.get("newName") or "").strip()
    if not tag_id_raw or not new_name:
        return jsonify({"message": "tagId and newName are required"}), 400

    try:
        tag_id = ObjectId(tag_id_raw)
    except Exception as exc:
        return jsonify({"message": "Invalid tagId", "error": str(exc)}), 400

    tag_doc = tags_collection.find_one({"_id": tag_id})
    if not tag_doc:
        return jsonify({"message": "Tag not found"}), 404

    new_slug = slugify_tag(new_name)
    existing = tags_collection.find_one({"slug": new_slug})
    if existing and existing.get("_id") != tag_doc.get("_id"):
        return jsonify({"message": "A tag with that name already exists"}), 409

    old_name = tag_doc.get("name", "")
    try:
        tags_collection.update_one(
            {"_id": tag_doc["_id"]},
            {"$set": {"name": new_name, "slug": new_slug}},
        )
        if old_name:
            parties_collection.update_many(
                {"tags": old_name},
                {"$set": {"tags.$[elem]": new_name}},
                array_filters=[{"elem": old_name}],
            )
        return jsonify({"message": "Tag renamed"}), 200
    except Exception as exc:
        return jsonify({"message": "Error renaming tag", "error": str(exc)}), 500


@app.route("/api/admin/carousel/rename", methods=["POST"])
@limiter.limit("100 per minute")
@protect
def rename_carousel():
    payload = request.get_json(silent=True) or {}
    carousel_id = (payload.get("carouselId") or "").strip()
    new_title = (payload.get("newName") or "").strip()
    if not carousel_id or not new_title:
        return jsonify({"message": "carouselId and newName are required"}), 400

    try:
        obj_id = ObjectId(carousel_id)
    except Exception as e:
        return jsonify({"message": "Invalid carouselId", "error": str(e)}), 400

    carousel = carousels_collection.find_one({"_id": obj_id})
    if not carousel:
        return jsonify({"message": "Carousel not found"}), 404

    old_title = carousel.get("title", "")
    if old_title == new_title:
        return jsonify({"message": "No change"}), 200

    existing = carousels_collection.find_one({"title": new_title})
    if existing and existing["_id"] != carousel["_id"]:
        return jsonify({"message": "A carousel with the new title already exists"}), 409

    try:
        carousels_collection.update_one(
            {"_id": carousel["_id"]},
            {"$set": {"title": new_title}}
        )
        return jsonify({"message": "Carousel renamed"}), 200
    except Exception as e:
        return jsonify({"message": "Error renaming carousel", "error": str(e)}), 500


@app.route("/api/admin/sections", methods=["POST"])
@limiter.limit("5 per minute")
@protect
def add_section():
    if carousels_collection is None or parties_collection is None:
        return jsonify({"message": "Storage unavailable."}), 503
    payload = request.get_json(silent=True) or {}
    try:
        req = CarouselImportRequest(**payload)
    except ValidationError as ve:
        app.logger.warning(f"[VALIDATION] {ve}")
        return jsonify({"message": "Invalid payload", "errors": ve.errors()}), 400

    title = (req.carouselName or req.title or "").strip()
    if not title:
        return jsonify({"message": "carouselName or title is required"}), 400
    source_url = (req.url or "").strip()
    if not source_url or not is_url_allowed(source_url):
        return jsonify({"message": "URL is not allowed."}), 400

    try:
        response = requests.get(source_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        return jsonify({"message": "Unable to fetch source URL.", "error": str(exc)}), 502

    event_urls = extract_event_urls_from_page(source_url, response.text)
    if not event_urls:
        return jsonify({"message": "No parties were found at the provided URL."}), 404

    referral = default_referral_code()
    collected: list[dict] = []
    warnings: list[dict] = []
    for event_url in event_urls:
        try:
            party_doc, _created = ensure_party_for_url(event_url, referral)
        except Exception as exc:
            warnings.append({"url": event_url, "error": str(exc)})
            continue
        if not party_doc or not party_doc.get("_id"):
            warnings.append({"url": event_url, "error": "Missing party identifier"})
            continue
        collected.append(party_doc)

    ordered_party_ids = []
    seen_ids: set[str] = set()
    for doc in collected:
        raw_id = doc.get("_id")
        if raw_id is None:
            continue
        str_id = str(raw_id)
        if str_id in seen_ids:
            continue
        seen_ids.add(str_id)
        ordered_party_ids.append(raw_id)

    if not ordered_party_ids:
        return jsonify({"message": "No parties could be imported from the provided URL."}), 404

    try:
        last = carousels_collection.find().sort("order", -1).limit(1)
        last_order = next(last, {}).get("order", -1)
    except Exception:
        last_order = -1

    doc = {
        "title": title,
        "partyIds": ordered_party_ids,
        "order": int(last_order) + 1,
    }

    try:
        result = carousels_collection.insert_one(doc)
        doc["_id"] = result.inserted_id
    except Exception as exc:
        return jsonify({"message": "Error creating carousel", "error": str(exc)}), 500

    payload = {
        "message": "Carousel created from section.",
        "carousel": serialize_carousel(doc),
        "partyCount": len(ordered_party_ids),
    }
    if warnings:
        payload["warnings"] = warnings
    return jsonify(payload), 201


# --- Discovery surfaces ---


def sitemap_xml(entries: list[dict], root_tag: str = "urlset") -> str:
    item_tag = "sitemap" if root_tag == "sitemapindex" else "url"
    body = [f"<{root_tag} xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">"]
    for entry in entries:
        loc = entry.get("loc")
        lastmod = entry.get("lastmod")
        if not loc:
            continue
        body.append(f"<{item_tag}><loc>{loc}</loc>{'<lastmod>'+lastmod+'</lastmod>' if lastmod else ''}</{item_tag}>")
    body.append(f"</{root_tag}>")
    return "".join(body)


@app.route("/sitemap.xml")
def sitemap_index():
    upcoming = serialize_events(include_past=False)
    shards = shard_events(upcoming)
    entries = []
    for idx, shard in enumerate(shards):
        lastmod = None
        for event in shard:
            if event.get("lastmod") and (not lastmod or event["lastmod"] > lastmod):
                lastmod = event["lastmod"]
        name = f"events-{idx}.xml"
        entries.append({
            "loc": build_canonical("sitemap_child", name=name),
            "lastmod": lastmod,
        })
    for name, getter in (
        ("cities.xml", lambda: aggregate_dimension(upcoming, "city")),
        ("venues.xml", lambda: aggregate_dimension(upcoming, "venue")),
        ("genres.xml", lambda: aggregate_genres(upcoming)),
        ("dates.xml", lambda: unique_dates(upcoming)),
    ):
        items = getter()
        lastmod = None
        for item in items:
            if item.get("lastmod") and (not lastmod or item["lastmod"] > lastmod):
                lastmod = item["lastmod"]
        entries.append({
            "loc": build_canonical("sitemap_child", name=name),
            "lastmod": lastmod,
        })
    record_setting_hit("analytics:sitemap:index")
    xml = sitemap_xml(entries, root_tag="sitemapindex")
    return text_response(xml, "application/xml", cache_seconds=SITEMAP_CACHE_SECONDS, robots=None)


def sitemap_child_response(items: list[dict]):
    entries = []
    for item in items:
        loc = item.get("canonicalUrl") or item.get("loc")
        if not loc:
            continue
        entries.append({"loc": loc, "lastmod": item.get("lastmod")})
    xml = sitemap_xml(entries)
    return text_response(xml, "application/xml", cache_seconds=SITEMAP_CACHE_SECONDS, robots=None)


@app.route("/sitemaps/events-<int:shard>.xml")
def sitemap_events_shard(shard: int):
    upcoming = serialize_events(include_past=False)
    shards = shard_events(upcoming)
    if shard < 0 or shard >= len(shards):
        return text_response("", "application/xml", cache_seconds=SITEMAP_CACHE_SECONDS, robots=None, status=404)
    events = shards[shard]
    items = [{"canonicalUrl": event.get("canonicalUrl"), "lastmod": event.get("lastmod") } for event in events]
    record_setting_hit("analytics:sitemap:events", {"shard": shard})
    return sitemap_child_response(items)


@app.route("/sitemaps/cities.xml")
def sitemap_cities():
    upcoming = serialize_events(include_past=False)
    items = aggregate_dimension(upcoming, "city")
    record_setting_hit("analytics:sitemap:cities")
    return sitemap_child_response(items)


@app.route("/sitemaps/venues.xml")
def sitemap_venues():
    upcoming = serialize_events(include_past=False)
    items = aggregate_dimension(upcoming, "venue")
    record_setting_hit("analytics:sitemap:venues")
    return sitemap_child_response(items)


@app.route("/sitemaps/genres.xml")
def sitemap_genres():
    upcoming = serialize_events(include_past=False)
    items = aggregate_genres(upcoming)
    record_setting_hit("analytics:sitemap:genres")
    return sitemap_child_response(items)


@app.route("/sitemaps/dates.xml")
def sitemap_dates():
    upcoming = serialize_events(include_past=False)
    items = unique_dates(upcoming)
    record_setting_hit("analytics:sitemap:dates")
    return sitemap_child_response(items)


def filtered_events_for_scope(scope_slug: str | None) -> tuple[list[dict], str, str]:
    events = serialize_events(include_past=False)
    scope_title = "Parties247 Events"
    scope_url = build_canonical("events_feed", fmt="rss").rsplit(".", 1)[0]
    if scope_slug:
        cities = {item["slug"]: item for item in aggregate_dimension(events, "city")}
        genres = {item["slug"]: item for item in aggregate_genres(events)}
        matched = []
        if scope_slug in cities:
            matched = [event for event in events if event.get("city", {}).get("slug") == scope_slug]
            scope_title = f"Events in {cities[scope_slug]['name']['en'] or cities[scope_slug]['name']['he']}"
            scope_url = build_canonical("scoped_feed", slug=scope_slug, fmt="rss").rsplit(".", 1)[0]
        elif scope_slug in genres:
            matched = [event for event in events if any(g.get("slug") == scope_slug for g in event.get("genres", []))]
            scope_title = f"{genres[scope_slug]['name']['en'] or genres[scope_slug]['name']['he']} Events"
            scope_url = build_canonical("scoped_feed", slug=scope_slug, fmt="rss").rsplit(".", 1)[0]
        events = matched
    return events, scope_title, scope_url


def feed_response(scope_slug: str | None, fmt: str):
    fmt = fmt.lower()
    events, title, url = filtered_events_for_scope(scope_slug)
    if scope_slug and not events:
        return text_response("", "text/plain", cache_seconds=FEED_CACHE_SECONDS, robots="noindex", status=404)
    body = build_feed(events, title, url, fmt)
    content_type = {
        "rss": "application/rss+xml",
        "atom": "application/atom+xml",
        "json": "application/feed+json",
    }.get(fmt, "application/octet-stream")
    record_setting_hit("analytics:feed", {"scope": scope_slug or "events", "format": fmt})
    return text_response(body, content_type, cache_seconds=FEED_CACHE_SECONDS, robots="noindex")


@app.route("/feeds/events.<fmt>")
def feeds_events(fmt: str):
    return feed_response(None, fmt)


@app.route("/feeds/<slug>.<fmt>")
def feeds_scope(slug: str, fmt: str):
    return feed_response(slug, fmt)


@app.route("/ics/event/<slug>.ics")
def ics_event(slug: str):
    events = [event for event in serialize_events(include_past=True) if event.get("slug") == slug]
    if not events:
        return text_response("", "text/calendar", cache_seconds=ICS_CACHE_SECONDS, robots="noindex", status=404)
    body = build_ics(events, title=events[0].get("title", {}).get("en") or events[0].get("title", {}).get("he") or "Event")
    record_setting_hit("analytics:ics:event", {"slug": slug})
    return text_response(body, "text/calendar", cache_seconds=ICS_CACHE_SECONDS, robots="noindex")


@app.route("/ics/city/<slug>.ics")
def ics_city(slug: str):
    events = [event for event in serialize_events(include_past=False) if event.get("city", {}).get("slug") == slug]
    if not events:
        return text_response("", "text/calendar", cache_seconds=ICS_CACHE_SECONDS, robots="noindex", status=404)
    title = f"Events in {events[0].get('city', {}).get('name', {}).get('en') or events[0].get('city', {}).get('name', {}).get('he') or slug}"
    body = build_ics(events, title=title)
    record_setting_hit("analytics:ics:city", {"slug": slug})
    return text_response(body, "text/calendar", cache_seconds=ICS_CACHE_SECONDS, robots="noindex")


@app.route("/robots.txt")
def robots_txt():
    lines = [
        "User-agent: *",
        "Disallow: /api/",
        "Disallow: /feeds/",
        "Disallow: /search",
        "Disallow: /admin",
        f"Sitemap: {build_canonical('sitemap_index')}",
    ]
    return text_response("\n".join(lines) + "\n", "text/plain", cache_seconds=LIST_CACHE_SECONDS, robots=None)

@app.route("/api/referral", methods=["GET"])
def get_referral():
    try:
        code = default_referral_code() or ""
        return jsonify({"referral": code}), 200
    except Exception as e:
        return jsonify({"message": "Error fetching referral", "error": str(e)}), 500

@app.route("/api/admin/referral", methods=["POST"])
@protect
def set_referral():
    payload = request.get_json(silent=True) or {}
    try:
        req = ReferralUpdateSchema(**payload)
    except ValidationError as ve:
        return jsonify({"message": "Invalid payload", "errors": ve.errors()}), 400
    val = (req.code or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", val):
        return jsonify({"message": "Invalid referral format. Use 1-64 chars [A-Za-z0-9_-]."}), 400
    try:
        settings_collection.update_one(
            {"key": REFERRAL_KEY},
            {"$set": {"key": REFERRAL_KEY, "value": val}},
            upsert=True
        )
        return jsonify({"message": "Referral updated"}), 200
    except Exception as e:
        return jsonify({"message": "Error updating referral", "error": str(e)}), 500

# --- Token verify ---
@app.route('/api/admin/verify-token', methods=['POST'])
@limiter.limit("10 per minute")
@protect
def verify_token():
    return jsonify({"message": "Token is valid."}), 200

if __name__ == "__main__":
    flask_env = os.environ.get("FLASK_ENV", "production").lower()
    debug = flask_env == "development" or (
        flask_env != "production"
        and os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")
    )
    app.run(debug=debug, port=int(os.environ.get("PORT", 3001)))