"""
Quick manual test for ticket price extraction.
Usage:
    python test_price.py <go-out-event-url>
    python test_price.py https://www.go-out.co/event/some-event
"""
import sys
import json
import requests
from bs4 import BeautifulSoup
from app import _extract_price_from_tickets, _extract_price_from_schema_org, _is_sold_out

def test_url(url: str):
    print(f"\nFetching: {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "he-IL,he;q=0.9,en-US;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    script = soup.find("script", {"id": "__NEXT_DATA__"})
    if not script or not script.string:
        print("ERROR: __NEXT_DATA__ not found in page")
        return

    data = json.loads(script.string)
    event_data = data.get("props", {}).get("pageProps", {}).get("event", {})

    import sys as _sys
    out = _sys.stdout.buffer if hasattr(_sys.stdout, 'buffer') else _sys.stdout

    def p(s): out.write((str(s) + "\n").encode("utf-8", errors="replace"))

    p("\n--- event_data keys ---")
    for k, v in event_data.items():
        if isinstance(v, list):
            preview = f"[list:{len(v)}]"
            if v and isinstance(v[0], dict):
                preview += f" keys={list(v[0].keys())}"
        elif isinstance(v, dict):
            preview = f"{{dict keys={list(v.keys())[:8]}}}"
        else:
            preview = repr(v)[:100]
        p(f"  {k}: {preview}")

    p(f"\n  cheapestTicket: {event_data.get('cheapestTicket', '(not present)')}")

    tickets = event_data.get("Tickets", [])
    p(f"\n--- Tickets ({len(tickets) if isinstance(tickets, list) else 'not found'}) ---")
    if isinstance(tickets, list):
        for t in tickets:
            p(f"  {t}")

    price_from_tickets = _extract_price_from_tickets(event_data)
    price_from_schema = _extract_price_from_schema_org(event_data.get("schemaOrg"))
    sold_out = _is_sold_out(event_data)

    p(f"\n--- Results ---")
    p(f"  soldOut           : {sold_out}")
    p(f"  Tickets price     : {price_from_tickets} ILS")
    p(f"  schemaOrg price   : {price_from_schema} ILS")
    final = None if sold_out else (price_from_schema or price_from_tickets)
    p(f"  Final (used)      : {final} ILS")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_price.py <go-out-event-url>")
        sys.exit(1)
    test_url(sys.argv[1])
