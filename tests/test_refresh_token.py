import sys
import types
from datetime import datetime, timedelta, timezone

import app


class _FakeJwt(types.SimpleNamespace):
    """Records what was encoded and lets tests choose the decoded exp."""

    class InvalidTokenError(Exception):
        pass

    class ExpiredSignatureError(Exception):
        pass

    def __init__(self, exp_by_token):
        super().__init__()
        self.exp_by_token = exp_by_token
        self.encoded = []

    def encode(self, payload, secret, algorithm="HS256"):
        self.encoded.append(payload)
        return "fresh-token"

    def decode(self, token, secret, algorithms=None, options=None):
        if token not in self.exp_by_token:
            raise self.InvalidTokenError("bad token")
        return {"exp": self.exp_by_token[token]}


def _call(monkeypatch, header, exp_by_token):
    flask_mod = sys.modules["flask"]
    fake = _FakeJwt(exp_by_token)
    monkeypatch.setattr(app, "jwt", fake)
    monkeypatch.setattr(app, "JWT_SECRET", "secret")
    flask_mod.request.headers = {"Authorization": header} if header else {}
    res, status = app.refresh_admin_token()
    return res, status, fake


def _ts(delta: timedelta) -> float:
    return (datetime.now(timezone.utc) + delta).timestamp()


def test_refresh_valid_token_issues_new_30_day_token(monkeypatch):
    res, status, fake = _call(monkeypatch, "Bearer good", {"good": _ts(timedelta(days=10))})
    assert status == 200
    assert res["token"] == "fresh-token"
    assert res["expiresAt"].endswith("+00:00")
    lifetime = fake.encoded[0]["exp"] - datetime.utcnow()
    assert timedelta(days=29, hours=23) < lifetime <= timedelta(days=30)


def test_refresh_recently_expired_token_still_works(monkeypatch):
    res, status, _ = _call(monkeypatch, "Bearer stale", {"stale": _ts(-timedelta(days=5))})
    assert status == 200
    assert res["token"] == "fresh-token"


def test_refresh_rejects_token_past_grace_window(monkeypatch):
    res, status, fake = _call(monkeypatch, "Bearer ancient", {"ancient": _ts(-timedelta(days=15))})
    assert status == 401
    assert "Log in again" in res["message"]
    assert fake.encoded == []


def test_refresh_rejects_missing_or_invalid_token(monkeypatch):
    _, status, _ = _call(monkeypatch, None, {})
    assert status == 401
    _, status, _ = _call(monkeypatch, "Bearer nope", {"good": _ts(timedelta(days=1))})
    assert status == 401


def test_refresh_requires_configured_secret(monkeypatch):
    flask_mod = sys.modules["flask"]
    monkeypatch.setattr(app, "JWT_SECRET", "")
    flask_mod.request.headers = {"Authorization": "Bearer whatever"}
    _, status = app.refresh_admin_token()
    assert status == 500


def test_openapi_documents_new_routes():
    spec = app.build_openapi_document()
    assert "/api/admin/refresh-token" in spec["paths"]
    assert "/api/admin/promo/whatsapp" in spec["paths"]
