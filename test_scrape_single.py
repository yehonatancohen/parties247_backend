import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import scrape_party_details

if len(sys.argv) < 2:
    print("Usage: python test_scrape_single.py <go-out-event-url>")
    sys.exit(1)

url = sys.argv[1]
try:
    print(f"Scraping {url}...")
    details = scrape_party_details(url)
    print("Success! Details:")
    for key, val in details.items():
        print(f"  {key}: {val}")
except Exception as e:
    print("Error:", e)
