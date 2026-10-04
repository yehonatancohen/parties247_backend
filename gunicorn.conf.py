# Loaded automatically by `gunicorn app:app` (Render's start command).
#
# Gunicorn's default is ONE sync worker = one request at a time. The admin
# analytics page fires ~8 requests at once, so they queued behind each other
# (and behind public site traffic): /api/health took 16-26s while a slow
# /api/analytics/summary was in flight, 0.5s when idle.
#
# Threads, not extra workers: APScheduler runs inside the web process, so a
# second worker would run price_scan / content_refresh twice. Request handlers
# are Mongo/HTTP I/O bound (GIL released), and the shared caches are lock-guarded.
worker_class = "gthread"
workers = 1
threads = 8
timeout = 120
