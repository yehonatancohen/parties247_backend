# parties247_backend

Flask + MongoDB API on Render (`https://parties247-backend.onrender.com`). Business model,
accounts and data flow are in the workspace root `../CLAUDE.md` — read that first.
`README.md` here documents the public/admin endpoints and analytics semantics in depth.

## Shape

Everything is in one file, `app.py` (~6.6k lines), except `promo.py` (pure ranking +
Hebrew WhatsApp message templates behind `GET /api/admin/promo/whatsapp`). Rough map of
`app.py` by line region:

| Region | What |
|---|---|
| ~100–400 | URL/slug/date helpers, `normalize_url` (dedup key), `append_referral_param`/`apply_default_referral` (adds `?ref=`/`?aff=` to every outbound link), `is_url_allowed` |
| ~470–520 | Mongo wiring + `ensure_index` (compares TTL too — the `visitor_analytics` 35-day TTL matters for 30d analytics) |
| ~580–1180 | Analytics builders: `build_analytics_summary`, `build_sales_by_party` (lifetime), `build_party_funnel` (windowed, uses `_sales_totals_by_event_id(cutoff)`), `build_time_series_analytics` |
| ~1180–1570 | Event normalisation, sitemaps, RSS/Atom feeds, ICS |
| ~1600–1660 | `protect` (admin JWT decorator), `_check_service_token`, security headers |
| ~1660–2440 | Public analytics beacons + admin analytics routes (`/api/admin/analytics/{visitors,detailed,sales,funnel}`) |
| ~2440–3800 | Hand-written OpenAPI document (`build_openapi_document`) — update it when adding routes |
| ~3800–4100 | Login, pydantic schemas, taxonomy classifiers (`get_region`, `get_music_type`, `get_event_type`, `get_age`, `get_tags`) — Hebrew/English keyword tables |
| ~4100–4400 | GoOut page scraping (`scrape_party_details`, `scrape_ticket_info`), `scheduled_price_scan` (every 30 min, upcoming parties only) |
| ~4400–5000 | Party CRUD, `delete_party` (+ `?redirectTo=` → `party_redirects`), `get_parties` |
| ~5000–5830 | Carousels, sections, tags, URL imports |
| ~5830–6000 | Sitemap/feed/robots routes, referral get/set |
| ~6000–6400 | GoOut pending approve/reject (admin + internal/service-token variants), `internal_scrape_party` |

Scheduler: `Flask-APScheduler` inside the web process — `price_scan` cron is the only job.
Render runs gunicorn, so be careful adding jobs (multiple workers = duplicate runs).

## Auth

- `@protect` → `Authorization: Bearer <JWT>` from `POST /api/admin/login` (`ADMIN_PASSWORD_HASH`
  bcrypt, `JWT_SECRET_KEY`). Tokens expire after **30 days** (`issue_admin_token`); the
  README's "15 minute" claim is outdated. `POST /api/admin/refresh-token` exchanges a valid
  or ≤14-days-expired token for a fresh one without the password (used by the
  `/seo-update` routine via `parties247-website/scripts/refresh-admin-token.sh`).
- `X-Service-Token: SERVICE_TOKEN` for `/api/internal/*` (fetcher only).
- CORS allowlist is hardcoded in `app.py` (localhost:3000, the Vercel preview, prod domain).

## Working here

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && playwright install chromium
cp .env.example .env   # needs a real MONGODB_URI; db name is party247
FLASK_ENV=development python app.py     # :3001 — /docs (Swagger), /analytics (mini dashboard)
pytest                                  # tests/ — fast, no network, no Mongo
```

`tests/conftest.py` stubs `flask`, `flask_cors`, `flask_limiter`, `flask_apscheduler`,
`jwt`, `bcrypt`, `dotenv`, `pymongo` and imports `app.py` as a plain module. So tests
exercise **helper functions and pure logic**, not HTTP. Follow that style: import the
function, call it with dicts, assert. Collections are module globals set to `None` under
test — monkeypatch `app.<name>_collection` with an in-memory fake whose `find(*args,
**kwargs)` accepts a query and projection (see `test_analytics_projections.py`). When a
new import or a changed call signature in `app.py` breaks collection, fix the stub in
`conftest.py` — the suite silently sat un-runnable for weeks in 2026-08 because of that.
`pytest.ini` restricts collection to `tests/`; the `test_*.py` files at repo root are live
scrape scripts.

Performance rule: never `fetch_all_documents(coll)` without a `projection` on `parties`
or `goout_sales` — the Render↔Atlas link moves ~140 KB/s, so full-document scans are
where the admin dashboard's multi-second waits came from (see root doc).

Every new endpoint or classifier change should come with a test in `tests/` and an
OpenAPI entry. `test_*.py` at repo root are live scrape scripts, not unit tests.

Deploy: push to `master` → Render rebuilds (`build.sh` installs Playwright). There is no
version marker exposed by the API; confirm deploys via `/api/health` timing or a log line.

## Rules

- Never change the referral logic so account2's code wins over account1's on a shared
  party (see root doc). `settings` collection key `referral` is the site-wide default.
- Analytics `revenue` must stay **our commission** (from `goout_sales_log`), not GoOut gross;
  `real*` fields carry GoOut's own numbers.
- Time windows: filter `goout_sales_log` by `recorded_at` for any "last N days" figure —
  `build_sales_by_party` is lifetime and was misused for that once.
- Deleting parties: always take the `redirectTo` path so slugs keep their SEO signal.
- `party247` is the database name. `parties247` is a different, empty DB that caused a
  split-brain incident once.
