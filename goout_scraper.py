"""
Go-Out Scraper — Playwright-based scraper for the Go-Out organizer panel.

Handles:
- Login with email/password + 2FA via Telegram relay
- Persistent session storage in MongoDB
- Event discovery from the organizer panel ("פאנל מארגנים") at /businesspage

Panel structure (observed):
  URL: https://go-out.co/businesspage
  - Event rows containing: event name, ID (#XXXXX), date, role, ticket stats
  - Each event row is clickable → leads to /businesspage/<event_id>
  - Event dashboard has "צפייה באירוע החי" link → /event/<slug>?ref=<code>
  - "..." menu per row with "שיתוף אירוע" (share) and "הסרת אירוע" (remove)
  - Filter dropdown: "אירועים פעילים" (active events)
"""

import os
import json
import asyncio
import logging
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any

logger = logging.getLogger(__name__)

GO_OUT_BASE = "https://www.go-out.co"
GO_OUT_LOGIN_URL = f"{GO_OUT_BASE}/loginpage"
GO_OUT_PANEL_URL = f"{GO_OUT_BASE}/businesspage"
GO_OUT_EVENT_BASE = f"{GO_OUT_BASE}/event/"

# Selectors observed from the live login page
LOGIN_EMAIL_SEL = "#register_email"
LOGIN_PASSWORD_SEL = "#register_password"
LOGIN_BUTTON_SEL = "#register_button"

# Organizer panel nav text
ORGANIZER_PANEL_TEXT = "פאנל מארגנים"


class GoOutAccount:
    """Configuration for a single Go-Out account."""
    def __init__(self, account_id: str, email: str, password: str, referral: str):
        self.account_id = account_id
        self.email = email
        self.password = password
        self.referral = referral


class GoOutScraper:
    """
    Playwright-based scraper that logs into Go-Out, navigates the
    organizer panel (/businesspage), and discovers event URLs.
    """

    def __init__(self, account: GoOutAccount, db=None, telegram_mgr=None):
        self.account = account
        self._db = db
        self._telegram = telegram_mgr
        self._browser = None
        self._context = None
        self._page = None
        self._playwright = None

    # ------------------------------------------------------------------
    # Session persistence (MongoDB)
    # ------------------------------------------------------------------

    def _sessions_coll(self):
        if self._db is not None:
            return self._db.goout_sessions
        return None

    def _load_storage_state(self) -> dict | None:
        coll = self._sessions_coll()
        if coll is None:
            return None
        try:
            doc = coll.find_one({"account_id": self.account.account_id})
            if doc and doc.get("session_valid") and doc.get("storage_state"):
                return doc["storage_state"]
        except Exception as exc:
            logger.warning(f"Failed to load session for {self.account.account_id}: {exc}")
        return None

    def _save_storage_state(self, state: dict):
        coll = self._sessions_coll()
        if coll is None:
            return
        try:
            coll.update_one(
                {"account_id": self.account.account_id},
                {"$set": {
                    "account_id": self.account.account_id,
                    "email": self.account.email,
                    "storage_state": state,
                    "session_valid": True,
                    "last_login": datetime.now(timezone.utc).isoformat(),
                    "last_checked": datetime.now(timezone.utc).isoformat(),
                }},
                upsert=True,
            )
        except Exception as exc:
            logger.error(f"Failed to save session for {self.account.account_id}: {exc}")

    def _mark_session_invalid(self):
        coll = self._sessions_coll()
        if coll is None:
            return
        try:
            coll.update_one(
                {"account_id": self.account.account_id},
                {"$set": {
                    "session_valid": False,
                    "last_checked": datetime.now(timezone.utc).isoformat(),
                }},
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Browser lifecycle
    # ------------------------------------------------------------------

    async def _launch(self):
        from playwright.async_api import async_playwright
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )

        saved = self._load_storage_state()
        ctx_kwargs = dict(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        if saved:
            logger.info(f"[{self.account.account_id}] Restoring saved session...")
            ctx_kwargs["storage_state"] = saved

        self._context = await self._browser.new_context(**ctx_kwargs)
        self._page = await self._context.new_page()

    async def close(self):
        try:
            if self._context:
                state = await self._context.storage_state()
                self._save_storage_state(state)
        except Exception:
            pass
        try:
            if self._browser:
                await self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Login flow
    # ------------------------------------------------------------------

    async def _is_logged_in(self) -> bool:
        """Navigate to panel page; if we get redirected to login → not logged in."""
        try:
            resp = await self._page.goto(
                GO_OUT_PANEL_URL, wait_until="domcontentloaded", timeout=30000
            )
            await self._page.wait_for_timeout(3000)
            current = self._page.url

            # If redirected away from businesspage → not logged in
            if "/loginpage" in current or "/login" in current:
                logger.info(f"[{self.account.account_id}] Redirected to login → not logged in")
                return False

            # If we see "ניהול אירועים" (Event Management) heading → logged in
            heading = await self._page.query_selector('text="ניהול אירועים"')
            if heading:
                logger.info(f"[{self.account.account_id}] Panel loaded — session valid")
                return True

            # Also check for the user's name in the navbar (logged-in indicator)
            login_btn = await self._page.query_selector('text="התחברות/הרשמה"')
            if login_btn and await login_btn.is_visible():
                logger.info(f"[{self.account.account_id}] Login button visible → not logged in")
                return False

            # If we're on businesspage and it loaded, assume logged in
            if "/businesspage" in current:
                logger.info(f"[{self.account.account_id}] On businesspage — assuming logged in")
                return True

            return False
        except Exception as exc:
            logger.warning(f"[{self.account.account_id}] Login check failed: {exc}")
            return False

    async def _perform_login(self) -> bool:
        """Navigate to login page, fill credentials, handle 2FA if needed."""
        logger.info(f"[{self.account.account_id}] Navigating to login page...")
        try:
            await self._page.goto(GO_OUT_LOGIN_URL, wait_until="domcontentloaded", timeout=30000)
            await self._page.wait_for_timeout(2000)
        except Exception as exc:
            logger.error(f"[{self.account.account_id}] Failed to load login page: {exc}")
            return False

        # Fill email
        try:
            await self._page.wait_for_selector(LOGIN_EMAIL_SEL, timeout=10000)
            await self._page.fill(LOGIN_EMAIL_SEL, self.account.email)
            await self._page.wait_for_timeout(500)
        except Exception as exc:
            logger.error(f"[{self.account.account_id}] Email field not found: {exc}")
            return False

        # Fill password
        try:
            await self._page.fill(LOGIN_PASSWORD_SEL, self.account.password)
            await self._page.wait_for_timeout(500)
        except Exception as exc:
            logger.error(f"[{self.account.account_id}] Password field not found: {exc}")
            return False

        # ── Ask the manager if they're ready for 2FA BEFORE clicking login.
        # This gives them time to be ready at their phone/email before the
        # code is actually sent by Go-Out.
        if self._telegram:
            logger.info(f"[{self.account.account_id}] Asking manager 2FA availability...")
            available = self._telegram.ask_2fa_availability_sync(
                self.account.account_id, timeout=600
            )
            if not available:
                logger.warning(f"[{self.account.account_id}] Manager not available for 2FA; aborting login.")
                self._mark_session_invalid()
                return False
            logger.info(f"[{self.account.account_id}] Manager ready — clicking login now.")

        # Click login (this is when Go-Out sends the 2FA email/SMS)
        try:
            await self._page.click(LOGIN_BUTTON_SEL)
            await self._page.wait_for_timeout(5000)
        except Exception as exc:
            logger.error(f"[{self.account.account_id}] Login button click failed: {exc}")
            return False

        # Check if 2FA is required and collect the code
        needs_2fa = await self._check_for_2fa()
        if needs_2fa:
            success = await self._handle_2fa(already_confirmed_available=True)
            if not success:
                self._mark_session_invalid()
                return False

        # Verify login succeeded by trying to access the panel
        await self._page.wait_for_timeout(2000)
        logged_in = await self._is_logged_in()
        if logged_in:
            state = await self._context.storage_state()
            self._save_storage_state(state)
            logger.info(f"[{self.account.account_id}] Login successful, session saved.")
        else:
            self._mark_session_invalid()
            logger.error(f"[{self.account.account_id}] Login failed after attempt.")
            try:
                html = await self._page.content()
                import os
                os.makedirs('scratch', exist_ok=True)
                with open('scratch/login_failed.html', 'w', encoding='utf-8') as f:
                    f.write(html)
                await self._page.screenshot(path='scratch/login_failed.png')
            except Exception as e:
                logger.error(f"Could not dump html/screenshot: {e}")
        return logged_in

    async def _check_for_2fa(self) -> bool:
        """Check if a 2FA/OTP input appeared after login attempt."""
        try:
            html = await self._page.content()
            import os
            os.makedirs('scratch', exist_ok=True)
            with open('scratch/2fa_page.html', 'w', encoding='utf-8') as f:
                f.write(html)
            await self._page.screenshot(path='scratch/2fa_page.png')
        except Exception as e:
            logger.error(f"Could not dump 2FA html/screenshot: {e}")

        try:
            for selector in [
                'input[maxlength="6"]',
                'input[type="tel"]',
                'input[placeholder*="קוד"]',
                'input[placeholder*="code"]',
                'input[placeholder*="OTP"]',
                'input[name*="otp"]',
                'input[name*="code"]',
                'input[name*="verification"]',
                '[class*="otp"]',
                '[class*="verification"]',
            ]:
                elem = await self._page.query_selector(selector)
                if elem and await elem.is_visible():
                    logger.info(f"[{self.account.account_id}] 2FA input detected: {selector}")
                    return True

            content = await self._page.content()
            for indicator in ["קוד אימות", "verification code", "enter code", "אימות דו"]:
                if indicator.lower() in content.lower():
                    logger.info(f"[{self.account.account_id}] 2FA text detected: {indicator}")
                    return True
        except Exception:
            pass
        return False

    async def _handle_2fa(self, already_confirmed_available: bool = False) -> bool:
        """Coordinate 2FA code entry via Telegram."""
        if not self._telegram:
            logger.error(f"[{self.account.account_id}] 2FA required but no Telegram manager.")
            return False

        # Only ask availability if not already confirmed by the pre-login check
        if not already_confirmed_available:
            available = self._telegram.ask_2fa_availability_sync(
                self.account.account_id, timeout=600
            )
            if not available:
                logger.warning(f"[{self.account.account_id}] Manager not available for 2FA.")
                return False

        code = self._telegram.request_2fa_code_sync(
            self.account.account_id, timeout=600
        )
        if not code:
            logger.warning(f"[{self.account.account_id}] No 2FA code received.")
            return False

        # Find and fill the OTP input
        otp_filled = False
        for selector in [
            'input[maxlength="6"]',
            'input[type="tel"]',
            'input[placeholder*="קוד"]',
            'input[placeholder*="code"]',
            'input[name*="otp"]',
            'input[name*="code"]',
        ]:
            try:
                elem = await self._page.query_selector(selector)
                if elem and await elem.is_visible():
                    await elem.type(code, delay=100)
                    otp_filled = True
                    break
            except Exception:
                continue

        if not otp_filled:
            try:
                await self._page.keyboard.type(code)
                otp_filled = True
            except Exception:
                pass

        if not otp_filled:
            logger.error(f"[{self.account.account_id}] Could not fill 2FA code.")
            return False

        # Try submit buttons and Enter
        await self._page.wait_for_timeout(500)
        for btn_text in ["אימות", "verify", "submit", "אישור", "confirm"]:
            try:
                btn = await self._page.query_selector(f'button:has-text("{btn_text}")')
                if btn and await btn.is_visible():
                    await btn.click()
                    break
            except Exception:
                continue
        try:
            await self._page.keyboard.press("Enter")
        except Exception:
            pass

        await self._page.wait_for_timeout(3000)
        self._telegram.send_message_sync(f"✅ 2FA code entered for {self.account.account_id}.")
        return True

    # ------------------------------------------------------------------
    # Ensure session is active
    # ------------------------------------------------------------------

    async def ensure_session(self) -> bool:
        """Launch browser, check/restore session, login if needed."""
        await self._launch()
        if await self._is_logged_in():
            return True
        logger.info(f"[{self.account.account_id}] Session invalid, attempting login...")
        return await self._perform_login()

    # ------------------------------------------------------------------
    # Event discovery from organizer panel (/businesspage)
    # ------------------------------------------------------------------

    async def discover_events(self) -> list[dict]:
        """
        Navigate the organizer panel:
        1. Go to /businesspage (event list)
        2. Extract event IDs from the list
        3. For each event, click into /businesspage/<id> to get the live event URL
        Returns list of dicts with event URLs and metadata.
        """
        events = []
        try:
            events = await self._scrape_organizer_panel()
        except Exception as exc:
            logger.error(f"[{self.account.account_id}] Organizer panel scrape failed: {exc}")

        # Save session after scraping
        try:
            if self._context:
                state = await self._context.storage_state()
                self._save_storage_state(state)
        except Exception:
            pass

        return events

    async def _scrape_organizer_panel(self) -> list[dict]:
        """
        Scrape the /businesspage event list.

        Panel structure:
        - Heading: "ניהול אירועים"
        - Event rows with columns: שם האירוע (name), ID, תפקיד (role),
          רכישות כרטיסים (ticket purchases)
        - Each row is clickable → goes to event dashboard
        - Event dashboard has "צפייה באירוע החי" link with the public event URL
        """
        logger.info(f"[{self.account.account_id}] Loading organizer panel...")

        # Navigate to panel if not already there
        current = self._page.url
        if "/businesspage" not in current:
            await self._page.goto(GO_OUT_PANEL_URL, wait_until="domcontentloaded", timeout=30000)
            await self._page.wait_for_timeout(3000)

        # Scroll to load all events (in case of lazy loading)
        for _ in range(5):
            await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await self._page.wait_for_timeout(1000)
        await self._page.evaluate("window.scrollTo(0, 0)")
        await self._page.wait_for_timeout(1000)

        # Strategy 1: Extract event IDs and URLs from __NEXT_DATA__
        events = await self._extract_from_next_data()
        if events:
            logger.info(f"[{self.account.account_id}] Got {len(events)} events from __NEXT_DATA__")
            return events

        # Strategy 2: Parse event rows from the DOM and click into each
        events = await self._extract_by_clicking_rows()
        logger.info(f"[{self.account.account_id}] Got {len(events)} events by clicking rows")
        return events

    async def _extract_from_next_data(self) -> list[dict]:
        """Try to extract event data from __NEXT_DATA__ JSON."""
        events = []
        try:
            next_data = await self._page.evaluate("""
                () => {
                    const el = document.querySelector('#__NEXT_DATA__');
                    if (el) return JSON.parse(el.textContent);
                    if (window.__NEXT_DATA__) return window.__NEXT_DATA__;
                    return null;
                }
            """)
            if not next_data:
                return []

            # Walk the JSON to find event objects with Url or slug fields
            found = []
            self._walk_json_for_events(next_data, found, set())
            return found
        except Exception as exc:
            logger.warning(f"[{self.account.account_id}] __NEXT_DATA__ extraction failed: {exc}")
            return []

    def _walk_json_for_events(self, data: Any, events: list, seen: set):
        """Recursively walk JSON to find event-like objects."""
        if isinstance(data, dict):
            # Check if this looks like an event with an ID and URL/slug
            event_id = data.get("_id") or data.get("id") or data.get("Id")
            url_val = data.get("Url") or data.get("url") or data.get("slug") or data.get("Slug")

            if event_id and url_val and str(event_id) not in seen:
                seen.add(str(event_id))
                # Build the public event URL
                url_str = str(url_val)
                if url_str.startswith("http"):
                    event_url = url_str
                elif url_str.startswith("/"):
                    event_url = GO_OUT_BASE + url_str
                else:
                    event_url = f"{GO_OUT_EVENT_BASE}{url_str}"

                event_info = {
                    "url": event_url,
                    "go_out_id": str(event_id),
                    "source": "next_data",
                }
                # Grab any extra fields available
                for key in ("Title", "title", "Name", "name"):
                    if key in data and data[key]:
                        event_info["name"] = str(data[key])
                        break
                for key in ("StartingDate", "startingDate", "date"):
                    if key in data and data[key]:
                        event_info["date"] = str(data[key])
                        break
                for key in ("CoverImage", "coverImage"):
                    if key in data and isinstance(data[key], dict):
                        img_url = data[key].get("Url") or data[key].get("url")
                        if img_url:
                            event_info["image"] = img_url
                        break

                events.append(event_info)

            for v in data.values():
                self._walk_json_for_events(v, events, seen)
        elif isinstance(data, list):
            for item in data:
                self._walk_json_for_events(item, events, seen)

    async def _extract_by_clicking_rows(self) -> list[dict]:
        """
        Fallback: parse event rows from the DOM, click into each event's
        dashboard to get the live event URL.
        """
        events = []

        # First, collect event IDs from the visible rows
        # Each row shows #XXXXX as the event ID
        try:
            page_text = await self._page.content()
            # Find all event IDs like #43994, #43601, etc.
            id_matches = re.findall(r'#(\d{4,6})', page_text)
            event_ids = list(dict.fromkeys(id_matches))  # dedupe preserving order
            logger.info(f"[{self.account.account_id}] Found event IDs: {event_ids}")
        except Exception as exc:
            logger.error(f"[{self.account.account_id}] Failed to extract event IDs: {exc}")
            return []

        # For each event ID, navigate to its dashboard and extract the live URL
        for i, eid in enumerate(event_ids):
            try:
                event_info = await self._get_event_url_from_dashboard(eid, dump_first=(i == 0))
                if event_info:
                    events.append(event_info)
                await asyncio.sleep(1)  # Rate limit
            except Exception as exc:
                logger.warning(f"[{self.account.account_id}] Error processing event #{eid}: {exc}")
                continue

        return events

    async def _get_event_url_from_dashboard(self, event_id: str, dump_first: bool = False) -> dict | None:
        """
        Navigate to /businesspage/<event_id> and extract the live event URL
        from the "צפייה באירוע החי" link.
        """
        dashboard_url = f"{GO_OUT_PANEL_URL}/{event_id}"
        try:
            await self._page.goto(dashboard_url, wait_until="domcontentloaded", timeout=20000)
            await self._page.wait_for_timeout(3000)  # extra wait for JS to render
        except Exception as exc:
            logger.warning(f"[{self.account.account_id}] Failed to load dashboard for #{event_id}: {exc}")
            return None

        # Dump HTML for the first event so we can inspect the DOM structure
        if dump_first:
            try:
                import os
                os.makedirs('scratch', exist_ok=True)
                html = await self._page.content()
                with open(f'scratch/dashboard_{event_id}.html', 'w', encoding='utf-8') as f:
                    f.write(html)
                await self._page.screenshot(path=f'scratch/dashboard_{event_id}.png')
                logger.info(f"[{self.account.account_id}] Dumped dashboard HTML for #{event_id}")
            except Exception as e:
                logger.warning(f"Could not dump dashboard html: {e}")

        # Look for the "צפייה באירוע החי" (View Live Event) link
        live_url = None
        try:
            # Try finding the link by text
            link = await self._page.query_selector('a:has-text("צפייה באירוע החי")')
            if link:
                live_url = await link.get_attribute("href")

            # Also try English text
            if not live_url:
                link = await self._page.query_selector('a:has-text("View live event")')
                if link:
                    live_url = await link.get_attribute("href")

            # Fallback: find any link containing /event/
            if not live_url:
                all_links = await self._page.query_selector_all('a[href*="/event/"]')
                for a in all_links:
                    href = await a.get_attribute("href")
                    if href and "/event/" in href and "/businesspage" not in href:
                        live_url = href
                        break

            # Also try extracting from __NEXT_DATA__ on the dashboard page
            if not live_url:
                try:
                    next_data = await self._page.evaluate("""
                        () => {
                            const el = document.querySelector('#__NEXT_DATA__');
                            if (el) return JSON.parse(el.textContent);
                            return null;
                        }
                    """)
                    if next_data:
                        found = []
                        self._walk_json_for_events(next_data, found, set())
                        if found:
                            live_url = found[0].get("url")
                except Exception:
                    pass

        except Exception as exc:
            logger.warning(f"[{self.account.account_id}] Failed to extract URL for #{event_id}: {exc}")

        if not live_url:
            logger.warning(f"[{self.account.account_id}] No live URL found for #{event_id}")
            return None

        # Normalize URL
        if live_url.startswith("/"):
            live_url = GO_OUT_BASE + live_url

        # Extract event name from the dashboard
        name = None
        try:
            # The event name is usually displayed prominently on the dashboard
            name_el = await self._page.query_selector('h1, h2, [class*="event-name"], [class*="eventName"]')
            if name_el:
                name = await name_el.text_content()
                name = (name or "").strip()
        except Exception:
            pass

        logger.info(f"[{self.account.account_id}] Event #{event_id} → {live_url}")

        return {
            "url": live_url,
            "go_out_id": event_id,
            "name": name,
            "source": "dashboard_click",
        }


# ---------------------------------------------------------------------------
# High-level orchestrator (called from the scheduler)
# ---------------------------------------------------------------------------

def run_daily_scrape(accounts: list[GoOutAccount], db, telegram_mgr, app_context):
    """
    Synchronous entry point for the daily scrape job.
    Runs the async scraper in a new event loop.
    """
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            _async_daily_scrape(accounts, db, telegram_mgr, app_context)
        )
    except Exception as exc:
        logger.error(f"Daily scrape failed: {exc}")
        if telegram_mgr:
            telegram_mgr.send_message_sync(f"❌ Daily scrape error: {exc}")
    finally:
        loop.close()


async def _async_daily_scrape(accounts, db, telegram_mgr, app_context):
    """Async daily scrape across all accounts."""
    from app import (
        scrape_party_details, normalize_url, normalized_or_none_for_dedupe,
        apply_default_referral, slugify_party,
    )

    if telegram_mgr:
        telegram_mgr.send_message_sync("🔄 *Daily Go-Out scrape starting...*")

    total_new = 0
    pending_coll = db.goout_pending if db is not None else None
    parties_coll = db.parties if db is not None else None

    for account in accounts:
        scraper = GoOutScraper(account, db=db, telegram_mgr=telegram_mgr)
        try:
            session_ok = await scraper.ensure_session()
            if not session_ok:
                if telegram_mgr:
                    telegram_mgr.send_message_sync(
                        f"⚠️ Could not log in to *{account.account_id}*. Skipping."
                    )
                continue

            event_entries = await scraper.discover_events()
            new_count = 0

            for entry in event_entries:
                event_url = entry.get("url", "")
                if not event_url:
                    continue

                canonical = normalize_url(event_url)

                # Skip if already in parties collection
                if parties_coll is not None:
                    try:
                        existing = parties_coll.find_one({
                            "$or": [
                                {"canonicalUrl": canonical},
                                {"goOutUrl": canonical},
                            ]
                        })
                        if existing:
                            continue
                    except Exception:
                        pass

                # Skip if already pending or previously rejected
                if pending_coll is not None:
                    try:
                        existing_pending = pending_coll.find_one({
                            "goOutUrl": canonical,
                            "status": {"$in": ["pending", "approved"]},
                        })
                        if existing_pending:
                            continue
                    except Exception:
                        pass

                # Scrape full event details using existing scraper
                try:
                    party_data = scrape_party_details(event_url)
                except Exception as exc:
                    logger.warning(f"Failed to scrape {event_url}: {exc}")
                    continue

                if not party_data:
                    continue

                # Apply referral code for this account
                party_data["referralCode"] = account.referral
                apply_default_referral(party_data, account.referral)
                party_data.setdefault(
                    "slug", slugify_party(party_data.get("name"), party_data.get("date"))
                )

                # Store as pending
                if pending_coll is not None:
                    try:
                        pending_doc = {
                            "party_data": party_data,
                            "account_id": account.account_id,
                            "scraped_at": datetime.now(timezone.utc),
                            "status": "pending",
                            "goOutUrl": canonical,
                        }
                        pending_coll.insert_one(pending_doc)
                        new_count += 1

                        # Send to Telegram for approval
                        if telegram_mgr:
                            telegram_mgr.send_party_for_approval_sync(pending_doc)
                    except Exception as exc:
                        logger.error(f"Failed to store pending party: {exc}")

                # Rate limit between event page scrapes
                await asyncio.sleep(2)

            total_new += new_count
            logger.info(f"[{account.account_id}] Found {new_count} new events")

        except Exception as exc:
            logger.error(f"[{account.account_id}] Scrape error: {exc}")
            if telegram_mgr:
                telegram_mgr.send_message_sync(
                    f"❌ Error scraping *{account.account_id}*: {exc}"
                )
        finally:
            await scraper.close()

    if telegram_mgr:
        telegram_mgr.send_message_sync(
            f"✅ *Daily scrape complete!*\n"
            f"Found {total_new} new events across {len(accounts)} accounts."
        )
