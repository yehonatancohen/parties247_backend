import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app


def authenticated_headers():
    flask_mod = sys.modules["flask"]
    app.JWT_SECRET = "secret"
    flask_mod.request.headers = {"Authorization": "Bearer token"}


def test_import_carousel_from_urls_uses_payload(monkeypatch):
    authenticated_headers()
    flask_mod = sys.modules["flask"]
    flask_mod.request.get_json = lambda silent=True: {
        "carouselName": " Spotlight Picks ",
        "urls": [
            "https://example.com/event/alpha",
            "https://example.com/event/beta",
        ],
    }

    monkeypatch.setattr(app, "is_url_allowed", lambda url: True)
    monkeypatch.setattr(app, "default_referral_code", lambda: "refcode")

    captured = {}

    def fake_import(name, urls, referral):
        captured.update({
            "name": name,
            "urls": urls,
            "referral": referral,
        })
        return ({"_id": "c42", "partyIds": ["p1", "p2"], "title": name.strip()}, 2, [])

    monkeypatch.setattr(app, "_import_carousel_from_urls", fake_import)
    monkeypatch.setattr(
        app,
        "serialize_carousel",
        lambda doc: {"id": str(doc.get("_id")), "partyIds": doc.get("partyIds", [])},
    )

    response, status = app.import_carousel_from_urls()

    assert status == 200
    assert response["processedUrlCount"] == 2
    assert response["addedCount"] == 2
    assert response["carousel"]["id"] == "c42"
    assert captured["name"] == "Spotlight Picks"
    assert captured["urls"] == [
        "https://example.com/event/alpha",
        "https://example.com/event/beta",
    ]
    assert captured["referral"] == "refcode"


def test_import_carousel_from_urls_rejects_invalid_urls(monkeypatch):
    authenticated_headers()
    flask_mod = sys.modules["flask"]
    flask_mod.request.get_json = lambda silent=True: {
        "carouselName": "Spotlight",
        "urls": ["https://example.com/event/alpha"],
    }

    monkeypatch.setattr(app, "is_url_allowed", lambda url: False)

    response, status = app.import_carousel_from_urls()

    assert status == 400
    assert "valid" in response["message"].lower()
