import sys
import os
import logging
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.DEBUG)

from app import app, GOOUT_ACCOUNTS_CONFIG, _get_goout_db, telegram_mgr
from goout_scraper import GoOutAccount, run_daily_scrape

def run_test():
    with app.app_context():
        accounts = [GoOutAccount(**cfg) for cfg in GOOUT_ACCOUNTS_CONFIG]
        if not accounts:
            print("No accounts configured.")
            return

        print(f"Testing Go-Out Panel Scraper with {len(accounts)} account(s)...")
        
        # Try to clear any existing polling sessions to avoid Conflict errors
        try:
            import requests
            token = os.environ.get("TELEGRAM_BOT_TOKEN")
            if token:
                requests.get(f"https://api.telegram.org/bot{token}/deleteWebhook", timeout=10)
                # Short sleep to give the API time to process the termination
                print("Clearing bot session...")
                import time
                time.sleep(10)
        except Exception: pass

        print("Waiting for Telegram bot to initialize...")
        import time
        time.sleep(2)
        
        # Test the orchestrator which logs in and discovers events
        # We pass force_send=True to send ALL discovered parties to Telegram
        run_daily_scrape(accounts, _get_goout_db(), telegram_mgr, app, force_send=True)
        
        print("Scrape test completed. Check the output logs.")

if __name__ == "__main__":
    run_test()
