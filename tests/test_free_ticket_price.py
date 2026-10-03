import app


def test_free_ticket_is_a_valid_available_price():
    assert app._extract_price_from_tickets({"Tickets": [
        {"Price": 0, "Amount": 100, "sold": 0, "Active": True},
        {"Price": 200, "Commision": 5, "Amount": 100, "sold": 0, "Active": True},
    ]}) == 0


def test_price_scan_preserves_free_schema_price(monkeypatch):
    import json
    from types import SimpleNamespace

    event = {
        "schemaOrg": [{"@type": "FAQPage", "mainEntity": [
            {"acceptedAnswer": {"text": "מתחילים ב-0.00₪"}}
        ]}],
        "Tickets": [{"Price": 200, "Commision": 5, "Amount": 150, "sold": 0, "Active": True}],
    }
    html = '<script id="__NEXT_DATA__" type="application/json">' + json.dumps(
        {"props": {"pageProps": {"event": event}}}
    ) + '</script>'
    monkeypatch.setattr(app.requests, "get", lambda *args, **kwargs: SimpleNamespace(text=html, encoding="utf-8"))
    assert app.scrape_ticket_info("https://go-out.co/event/1786345860168") == {"price": 0, "soldOut": False}
