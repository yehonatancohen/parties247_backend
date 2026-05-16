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
        
        # Listen to console logs from the page
        self._page.on("console", lambda msg: logger.debug(f"[{self.account.account_id}] PAGE CONSOLE: {msg.text}"))
        
        # Dismiss cookie dialogs globally
        await self._dismiss_cookie_dialog()

    async def _dismiss_cookie_dialog(self):
        """Try to find and click 'Allow all cookies' or similar buttons."""
        try:
            # Common CookieHub/UserWay/GDPR selectors
            selectors = [
                ".ch2-allow-all-btn",
                "button:has-text('Allow all')",
                "button:has-text('Allow all cookies')",
                "button:has-text('אשר הכל')",
                "button:has-text('אשר קוקיז')",
                "button:has-text('Got it')",
                "#hs-eu-confirmation-button",
            ]
            for sel in selectors:
                try:
                    btn = await self._page.query_selector(sel)
                    if btn and await btn.is_visible():
                        logger.info(f"[{self.account.account_id}] Dismissing cookie dialog: {sel}")
                        await btn.click()
                        await self._page.wait_for_timeout(1000)
                        break
                except Exception:
                    continue
        except Exception:
            pass

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
        api_events = [] # This is the list we'll populate
        events = []
        try:
            # We pass the list to be populated by the interceptor
            events = await self._scrape_organizer_panel(api_events)
            
            # Merge results: if _scrape_organizer_panel returned something, use it.
            # Otherwise, use what the API intercept caught.
            if not events and api_events:
                logger.info(f"[{self.account.account_id}] Falling back to {len(api_events)} intercepted events.")
                events = api_events
            elif events and api_events:
                # Merge and dedupe
                seen_ids = {e["go_out_id"] for e in events if "go_out_id" in e}
                for ae in api_events:
                    eid = ae.get("go_out_id")
                    if eid and eid not in seen_ids:
                        events.append(ae)
                        seen_ids.add(eid)

        except Exception as exc:
            logger.error(f"[{self.account.account_id}] Organizer panel discovery failed: {exc}")
            # The page may have timed out (e.g. networkidle never reached) but the
            # response interceptor already collected events — don't throw them away.
            if api_events and not events:
                logger.info(f"[{self.account.account_id}] Recovering {len(api_events)} intercepted events despite error.")
                events = api_events

        # Save session after scraping
        try:
            if self._context:
                state = await self._context.storage_state()
                self._save_storage_state(state)
        except Exception:
            pass

        logger.info(f"[{self.account.account_id}] Discovery complete. Total: {len(events)}")
        return events

    async def _scrape_organizer_panel(self, api_events_found: list) -> list[dict]:
        """
        Scrape the /businesspage event list.
        """
        logger.info(f"[{self.account.account_id}] Loading organizer panel...")

        # We must use a list we can modify from the callback
        async def intercept_response(response):
            try:
                # logger.debug(f"[{self.account.account_id}] Response: {response.url}")
                # Be more broad: check for anything that might be JSON
                ct = response.headers.get("content-type", "").lower()
                if response.ok and ("json" in ct or "text/plain" in ct or "application/octet-stream" in ct):
                    url = response.url
                    
                    # Skip noise
                    if "userway" in url or "facebook" in url or "google" in url or "tiktok" in url:
                        return

                    try:
                        data = await response.json()
                    except Exception:
                        # Maybe it's text that we can parse as JSON
                        try:
                            text = await response.text()
                            data = json.loads(text)
                        except Exception:
                            return

                    # Debug: Log URLs that look like they might contain event data
                    data_str = json.dumps(data)
                    logger.debug(f"[{self.account.account_id}] API Response from {url} (size: {len(data_str)})")
                    
                    if "events" in url or "business" in url or "myEvents" in url:
                        logger.info(f"[{self.account.account_id}] Potential Event API: {url}")

                    def find_slugs(obj):
                        if isinstance(obj, dict):
                            # Try to find event-like objects
                            eid = obj.get("EventSerial") or obj.get("_id") or obj.get("id") or obj.get("Id")
                            slug = obj.get("Url") or obj.get("url") or obj.get("slug") or obj.get("Slug") or obj.get("slugName")
                            
                            if not slug:
                                slug = obj.get("link") or obj.get("publicUrl")

                            if eid and slug and isinstance(slug, (str, int)) and "eventmanagement" not in str(slug):
                                slug_str = str(slug)
                                # Normalize slug: if it's just numbers but long (13 digits), it's a slug
                                if "go-out.co" in slug_str:
                                    full_url = slug_str
                                else:
                                    full_url = f"https://go-out.co/event/{slug_str}"

                                # Correct check using the reference passed to the method
                                already = False
                                for e in api_events_found:
                                    if e.get("go_out_id") == str(eid):
                                        already = True
                                        break

                                if not already:
                                    logger.info(f"[{self.account.account_id}] API INTERCEPT: Found slug '{slug_str}' for #{eid}")
                                    api_events_found.append({
                                        "url": full_url,
                                        "go_out_id": str(eid),
                                        "name": obj.get("Title") or obj.get("Name"),
                                        "source": "api_intercept"
                                    })
                            for v in obj.values():
                                if isinstance(v, (dict, list)):
                                    find_slugs(v)
                        elif isinstance(obj, list):
                            for item in obj:
                                find_slugs(item)

                    find_slugs(data)
            except Exception:
                pass

        self._page.on("response", intercept_response)

        # Always navigate fresh so the listener above catches all API responses.
        # Use domcontentloaded — networkidle never fires because analytics/tracking
        # (Meta Pixel, Stripe, TikTok) keep making requests indefinitely.
        await self._page.goto(GO_OUT_PANEL_URL, wait_until="domcontentloaded", timeout=30000)
        await self._page.wait_for_timeout(3000)

        # Ensure we are on the "Events" tab (sometimes it defaults to Promos or others)
        try:
            # Look for "אירועים" or "Events" tab/link
            # The icon is /assets/MyEvents/globeGrey.svg or globeRed.svg
            events_tab_selectors = [
                'div[role="button"]:has(img[src*="MyEvents/globe"])',
                'button:has(img[src*="MyEvents/globe"])',
                'a[href="/businesspage"]',
                'img[src*="MyEvents/globe"]',
                'text="אירועים"',
                'text="Events"',
            ]
            
            logger.info(f"[{self.account.account_id}] Current URL: {self._page.url}")
            
            # Try to ensure we are on the Events tab (not Promos/Dashboard)
            logger.info(f"[{self.account.account_id}] Ensuring Events tab is active...")
            for sel in events_tab_selectors:
                try:
                    tab = await self._page.query_selector(sel)
                    if tab:
                        await tab.click(force=True)
                        await self._page.wait_for_timeout(2000)
                        break  # Stop after first successful click
                except Exception:
                    pass
                    
            # Wait for table or rows to appear
            await self._page.wait_for_selector('tr, [role="row"], .MuiTableRow-root', timeout=15000, state="attached")
        except Exception as e:
            logger.warning(f"[{self.account.account_id}] Could not ensure Events tab or table: {e}")

        # Scroll to load all events (in case of lazy loading)
        for _ in range(3):
            await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await self._page.wait_for_timeout(2000)
        await self._page.evaluate("window.scrollTo(0, 0)")
        
        # Wait for API responses to come in - this is our best bet!
        logger.info(f"[{self.account.account_id}] Waiting for API responses (up to 72s)...")
        for i in range(24): # Wait up to 72 seconds total (24 × 3s)
            if len(api_events_found) >= 1:
                # Brief extra wait to let remaining events trickle in
                await self._page.wait_for_timeout(3000)
                logger.info(f"[{self.account.account_id}] Caught {len(api_events_found)} events via API. Returning FAST.")
                return api_events_found

            # Every ~21 seconds, try to kick the UI if no events found
            if i > 0 and i % 7 == 0:
                logger.info(f"[{self.account.account_id}] Still no API data. Trying to click Events tab again...")
                try:
                    # Capture state for debugging
                    import os
                    os.makedirs('scratch', exist_ok=True)
                    await self._page.screenshot(path=f'scratch/debug_{self.account.account_id}_{i}.png')
                    
                    # Try a very broad click on anything that looks like "Events"
                    await self._page.evaluate("""
                        () => {
                            const targets = Array.from(document.querySelectorAll('button, div, a, li, span, img'));
                            const eventsBtn = targets.find(el => {
                                const txt = (el.innerText || "").trim();
                                const src = (el.src || "").toLowerCase();
                                return (txt === "אירועים" || txt === "Events" || src.includes("globe"));
                            });
                            if (eventsBtn) {
                                eventsBtn.scrollIntoView();
                                eventsBtn.click();
                            }
                        }
                    """)
                except Exception: pass

            await self._page.wait_for_timeout(3000)
            if i % 7 == 0:
                logger.info(f"[{self.account.account_id}] Still waiting for API ({len(api_events_found)} found so far...)")
        
        # If we reached here, API intercept didn't find anything in 2m.
        # Check if we have anything at all
        if api_events_found:
            return api_events_found
        
        if api_events_found:
            logger.info(f"[{self.account.account_id}] Found {len(api_events_found)} events via API intercept.")
            return api_events_found

        # Strategy 1: Extract event IDs and URLs from __NEXT_DATA__
        events = await self._extract_from_next_data()
        if events:
            logger.info(f"[{self.account.account_id}] Got {len(events)} events from __NEXT_DATA__")

        # Strategy 2: Parse event rows from the DOM and click into each
        # We MUST do this for account2 if API didn't find much, as its API is unreliable
        seen_ids_from_api = {e["go_out_id"] for e in api_events_found if "go_out_id" in e}
        events_strategy2 = await self._extract_by_clicking_rows(seen_ids_from_api)
        
        # Merge all strategies
        all_found = api_events_found + events + events_strategy2
        seen = set()
        final = []
        for e in all_found:
            eid = e.get("go_out_id")
            if eid and eid not in seen:
                final.append(e)
                seen.add(eid)
        
        return final

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

    async def _extract_by_clicking_rows(self, already_found_ids: set) -> list[dict]:
        """
        Fallback: parse event rows from the DOM, click into each event's
        dashboard to get the live event URL.
        """
        events = []

        # First, collect event IDs and possibly names from the visible rows
        try:
            # We use a more flexible approach to find rows
            row_data = await self._page.evaluate("""
                () => {
                    const results = [];
                    // Find all elements that look like an ID (#XXXXX)
                    const elements = Array.from(document.querySelectorAll('*'));
                    for (const el of elements) {
                        try {
                            if (el.children.length === 0 && el.innerText && /#\d{5,6}/.test(el.innerText)) {
                                const idMatch = el.innerText.match(/#(\d{5,6})/);
                                const id = idMatch[1];
                                
                                // Find a suitable parent that acts as a row (e.g., has some height and width)
                                let row = el;
                                while (row && row.parentElement && row.offsetWidth < 200) {
                                    row = row.parentElement;
                                }
                                
                                if (row && !results.find(r => r.id === id)) {
                                    results.push({
                                        id: id,
                                        name: (row.innerText || "").split('\\n')[0].trim(),
                                        selector: row.className ? `.${row.className.split(' ').join('.')}` : null
                                    });
                                }
                            }
                        } catch(e) {}
                    }
                    return results;
                }
            """)
            
            event_ids = [r['id'] for r in row_data]
            # Dedupe preserving order
            event_ids = list(dict.fromkeys(event_ids))
            logger.info(f"[{self.account.account_id}] Found {len(event_ids)} event IDs via flexible search: {event_ids}")

        except Exception as exc:
            logger.error(f"[{self.account.account_id}] Failed to extract event IDs: {exc}")
            return []

        # For each event ID, try to get the URL (unless already found)
        for i, eid in enumerate(event_ids):
            if eid in already_found_ids:
                logger.debug(f"[{self.account.account_id}] Skipping #{eid} - already found via API")
                continue

            try:
                # 1. Try share button (if we can find it)
                event_info = await self._get_event_url_via_share(eid)
                
                # 2. Try clicking the row itself to go to dashboard
                if not event_info:
                    logger.info(f"[{self.account.account_id}] Share flow failed for #{eid}, trying direct row click...")
                    row_clicked = await self._page.evaluate(f"""
                        (id) => {{
                            const elements = Array.from(document.querySelectorAll('*'));
                            const el = elements.find(e => e.children.length === 0 && e.innerText && e.innerText.includes(id));
                            if (!el) return false;
                            
                            let row = el;
                            while (row && row.parentElement && row.offsetWidth < 200) {{
                                row = row.parentElement;
                            }}
                            if (row) {{
                                row.scrollIntoView();
                                row.click();
                                return true;
                            }}
                            return false;
                        }}
                    """, eid)
                    
                    if row_clicked:
                        await self._page.wait_for_timeout(5000)
                        logger.info(f"[{self.account.account_id}] After click, URL is: {self._page.url}")
                        event_info = await self._get_event_url_from_dashboard(eid)
                        # Go back to panel
                        await self._page.goto(GO_OUT_PANEL_URL, wait_until="domcontentloaded")
                        await self._page.wait_for_timeout(3000)
                
                # 3. Last resort: navigate directly to possible dashboard URLs
                if not event_info:
                    event_info = await self._get_event_url_from_dashboard(eid)
                
                if event_info:
                    events.append(event_info)
                
                await asyncio.sleep(1)
            except Exception as exc:
                logger.warning(f"[{self.account.account_id}] Error processing event #{eid}: {exc}")
                # Try to recover by going back to panel
                await self._page.goto(GO_OUT_PANEL_URL, wait_until="domcontentloaded")
                await self._page.wait_for_timeout(3000)
                continue

        return events

    async def _get_event_url_via_share(self, event_id: str) -> dict | None:
        """Try to get the public event URL by clicking the 'Share' option in the row menu."""
        try:
            # 1. Find and click more options button using evaluate
            clicked = await self._page.evaluate(f"""
                (id) => {{
                    const rows = Array.from(document.querySelectorAll('tr, [role="row"], .MuiTableRow-root, div'));
                    const row = rows.find(r => r.innerText.includes(id) && r.offsetWidth > 100);
                    if (row) {{
                        row.scrollIntoView();
                        const btns = Array.from(row.querySelectorAll('button, [role="button"]'));
                        // More options button often has no text but has an aria-haspopup or svg
                        const btn = btns.find(b => b.getAttribute('aria-haspopup') === 'true' || b.innerHTML.includes('svg') || b.innerText.includes('...'));
                        if (btn) {{
                            btn.click();
                            return true;
                        }}
                    }}
                    return false;
                }}
            """, event_id)

            if not clicked:
                logger.warning(f"[{self.account.account_id}] Row or menu button for #{event_id} not found")
                # Dump HTML of the suspected row area
                try:
                    area_html = await self._page.evaluate(f"""
                        (id) => {{
                            const rows = Array.from(document.querySelectorAll('tr, [role="row"], .MuiTableRow-root, div'));
                            const row = rows.find(r => r.innerText.includes(id) && r.offsetWidth > 100);
                            return row ? row.outerHTML : "ROW NOT FOUND";
                        }}
                    """, event_id)
                    logger.info(f"[{self.account.account_id}] Row HTML for #{event_id}: {area_html[:1000]}")
                except Exception: pass

                # Try clicking any button that contains the text '...' or looks like a menu
                await self._page.evaluate(f"""
                    () => {{
                        const btns = Array.from(document.querySelectorAll('button, [role="button"]'));
                        const btn = btns.find(b => b.innerText.includes('...') || b.getAttribute('aria-label') === 'more' || b.innerHTML.includes('svg'));
                        if (btn) btn.click();
                    }}
                """)

            await self._page.wait_for_timeout(2000)

            # 2. Find and click "Share" in the menu using evaluate
            share_clicked = await self._page.evaluate("""
                () => {
                    const elements = Array.from(document.querySelectorAll('li, [role="menuitem"], button, div, span'));
                    const share = elements.find(el => {
                        const txt = (el.innerText || "").toLowerCase();
                        return (txt.includes("שיתוף") || txt.includes("share")) && el.offsetWidth > 0;
                    });
                    if (share) {
                        share.click();
                        return true;
                    }
                    return false;
                }
            """)

            if not share_clicked:
                logger.warning(f"[{self.account.account_id}] Share button not found in menu for #{event_id}")
                # Dump menu HTML
                try:
                    menu_html = await self._page.evaluate("""
                        () => {
                            const menus = Array.from(document.querySelectorAll('[role="menu"], .MuiMenu-list, ul'));
                            return menus.map(m => m.outerHTML).join("\\n---\\n");
                        }
                    """)
                    logger.debug(f"[{self.account.account_id}] Menu HTML: {menu_html[:500]}...")
                except Exception: pass
                await self._page.mouse.click(0, 0)
                return None

            await self._page.wait_for_timeout(3000)

            # 3. Extract URL from the modal/menu
            live_url = await self._extract_url_from_share_dialog()

            # If still no URL, try to extract from window state
            if not live_url:
                live_url = await self._page.evaluate("""
                    () => {
                        try {
                            if (window.__NEXT_DATA__ && window.__NEXT_DATA__.props) {
                                // Search for slug in next data
                                const str = JSON.stringify(window.__NEXT_DATA__);
                                const match = str.match(/"slug":"([^"]+)"/);
                                if (match) return match[1];
                            }
                        } catch(e) {}
                        return null;
                    }
                """)
                if live_url and "/" not in live_url:
                    live_url = f"/event/{live_url}"

            # 4. Close dialog
            await self._page.keyboard.press("Escape")
            await self._page.wait_for_timeout(500)

            if live_url:
                if live_url.startswith("/"):
                    live_url = GO_OUT_BASE + live_url

                logger.info(f"[{self.account.account_id}] Got URL via share for #{event_id}: {live_url}")

                return {
                    "url": live_url,
                    "go_out_id": event_id,
                    "source": "share_button",
                }

        except Exception as exc:
            logger.warning(f"[{self.account.account_id}] Share button flow failed for #{event_id}: {exc}")

        return None


    async def _extract_url_from_share_dialog(self) -> str | None:
        """Helper to find a go-out.co URL in a modal or sharing menu."""
        # We prioritize URLs with slugs (non-digits) over URLs with just IDs
        return await self._page.evaluate("""
            () => {
                const results = [];
                
                // 1. Look in input fields
                const inputs = Array.from(document.querySelectorAll('input'));
                inputs.forEach(i => {
                    if (i.value.includes('go-out.co')) results.push(i.value);
                });

                // 2. Look for links (WhatsApp, etc.)
                const links = Array.from(document.querySelectorAll('a'));
                for (const a of links) {
                    const href = a.href || "";
                    if (href.includes('wa.me') || href.includes('whatsapp.com')) {
                        const match = href.match(/(https?:\\/\\/(?:www\\.)?go-out\\.co\\/(?:event|i)\\/[^&\\s]+)/);
                        if (match) results.push(match[1]);
                    }
                    if (href.includes('go-out.co/event/') || href.includes('go-out.co/i/')) {
                        if (!href.includes('eventmanagement') && !href.includes('businesspage')) {
                            results.push(href);
                        }
                    }
                }

                // 3. Look for ANY text that looks like a URL
                const bodyText = document.body.innerText;
                const matches = bodyText.match(/(https?:\\/\\/(?:www\\.)?go-out\\.co\\/(?:event|i)\\/[^\\s\\n\\r]+)/g);
                if (matches) results.push(...matches);

                if (results.length === 0) return null;

                // Prioritize: 
                // 1. URLs containing /i/ (short links often have slugs)
                // 2. URLs containing /event/ with non-digits (real slugs)
                // 3. Any other go-out link
                
                const slugs = results.filter(r => {
                    const parts = r.split('/');
                    const last = parts[parts.length - 1].split('?')[0];
                    return /[^0-9]/.test(last); // contains non-digits
                });
                
                if (slugs.length > 0) return slugs[0];
                return results[0];
            }
        """)

    async def _get_event_url_from_dashboard(self, event_id: str, dump_first: bool = False) -> dict | None:
        """
        Navigate to /businesspage/<event_id> and extract the live event URL.
        """
        # Try a few different URL patterns for the dashboard
        possible_urls = [
            f"https://www.go-out.co/businesspage/event/{event_id}",
            f"https://www.go-out.co/eventmanagement?eventId={event_id}",
            f"https://www.go-out.co/businesspage/{event_id}",
        ]

        dashboard_loaded = False
        for dashboard_url in possible_urls:
            try:
                logger.info(f"[{self.account.account_id}] Navigating to: {dashboard_url}")
                # We use networkidle here because the dashboard is very JS-heavy
                await self._page.goto(dashboard_url, wait_until="domcontentloaded", timeout=30000)
                await self._page.wait_for_timeout(10000) # Wait long for dashboard JS

                page_text = await self._page.evaluate("document.body.innerText")
                logger.debug(f"[{self.account.account_id}] Dashboard page text snippet: {page_text[:1000]}")
                
                # Check for "Manage" or "Share" which indicates we are in the dashboard
                if "Manage" in page_text or "ניהול" in page_text or "Share" in page_text or "שיתוף" in page_text:
                    # Also check if NEXT_DATA says we are on the right event
                    is_correct = await self._page.evaluate(f"""
                        (targetId) => {{
                            try {{
                                if (window.__NEXT_DATA__ && window.__NEXT_DATA__.query && window.__NEXT_DATA__.query.eventId) {{
                                    return String(window.__NEXT_DATA__.query.eventId) === String(targetId);
                                }}
                                // Check if the ID is mentioned in the body prominently
                                if (document.body.innerText.includes(targetId)) return true;
                            }} catch(e) {{}}
                            return true; // Fallback to true if we can't check
                        }}
                    """, event_id)
                    
                    if is_correct:
                        dashboard_loaded = True
                        break
                    else:
                        logger.warning(f"[{self.account.account_id}] Dashboard loaded but might be wrong page for #{event_id}")
            except Exception as e: 
                logger.warning(f"[{self.account.account_id}] Failed to load dashboard {dashboard_url}: {e}")
                continue

        if not dashboard_loaded:
            return None

        # Try extracting URL using evaluate for buttons and NEXT_DATA
        live_url = None
        try:
            # 1. Try NEXT_DATA first (most reliable if present)
            next_data_str = await self._page.evaluate("""
                () => {
                    const el = document.querySelector('#__NEXT_DATA__');
                    return el ? el.textContent : (window.__NEXT_DATA__ ? JSON.stringify(window.__NEXT_DATA__) : null);
                }
            """)
            if next_data_str:
                try:
                    import os
                    os.makedirs('scratch', exist_ok=True)
                    with open(f'scratch/next_data_{event_id}.json', 'w', encoding='utf-8') as f:
                        f.write(next_data_str)
                except Exception: pass

                # Look for slug or public URL
                match = re.search(r'"slug":"([^"]+)"', next_data_str) or re.search(r'"publicUrl":"([^"]+)"', next_data_str)
                if match:
                    live_url = match.group(1)

            # 2. Try to find and click Share on dashboard using evaluate
            if not live_url:
                share_clicked = await self._page.evaluate("""
                    () => {
                        const elements = Array.from(document.querySelectorAll('button, [role="button"], a, span, div'));
                        const share = elements.find(el => {
                            const txt = (el.innerText || "").trim();
                            return (txt === "שיתוף" || txt === "Share" || txt.includes("Share")) && el.offsetWidth > 0;
                        });
                        if (share) {
                            share.click();
                            return true;
                        }
                        return false;
                    }
                """)

                if share_clicked:
                    await self._page.wait_for_timeout(3000)
                    live_url = await self._extract_url_from_share_dialog()
                    await self._page.keyboard.press("Escape")

            # 3. Try direct links on dashboard using evaluate
            if not live_url:
                live_url = await self._page.evaluate("""
                    () => {
                        const links = Array.from(document.querySelectorAll('a, button, div[role="button"]'));
                        const live = links.find(el => {
                            const h = el.href || "";
                            const t = el.innerText || "";
                            return (t.includes("צפייה") || t.includes("View") || t.includes("Live")) && 
                                   (h.includes("/event/") || h.includes("/i/") || t.includes("View"));
                        });
                        if (live && live.href) return live.href;
                        
                        // If it's a button/div with text "View", it might open a new tab or we might need to find it differently
                        return null;
                    }
                """)

        except Exception as exc:
            logger.warning(f"[{self.account.account_id}] Dashboard extraction failed: {exc}")

        if not live_url:
            logger.warning(f"[{self.account.account_id}] No live URL found for #{event_id}")
            return None

        if live_url.startswith("/"):
            live_url = GO_OUT_BASE + live_url

        name = None
        try:
            name_el = await self._page.query_selector('h1, h2, [class*="name"]')
            if name_el:
                name = (await name_el.inner_text()).strip()
        except Exception: pass

        logger.info(f"[{self.account.account_id}] Got URL for #{event_id}: {live_url}")

        return {
            "url": live_url,
            "go_out_id": event_id,
            "name": name,
            "source": "dashboard_extraction",
        }





# ---------------------------------------------------------------------------
# High-level orchestrator (called from the scheduler)
# ---------------------------------------------------------------------------

def run_daily_scrape(accounts: list[GoOutAccount], db, telegram_mgr, app_context, force_send: bool = False):
    """
    Synchronous entry point for the daily scrape job.
    Runs the async scraper in a new event loop.
    """
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            _async_daily_scrape(accounts, db, telegram_mgr, app_context, force_send=force_send)
        )
    except Exception as exc:
        logger.error(f"Daily scrape failed: {exc}")
        if telegram_mgr:
            telegram_mgr.send_message_sync(f"❌ Daily scrape error: {exc}")
    finally:
        loop.close()


async def _async_daily_scrape(accounts, db, telegram_mgr, app_context, force_send: bool = False):
    """Async daily scrape across all accounts."""
    from app import (
        scrape_party_details, normalize_url, normalized_or_none_for_dedupe,
        apply_default_referral, slugify_party,
    )

    if telegram_mgr and force_send:
        telegram_mgr.send_message_sync("🔄 *Manual scan starting — sending all found parties...*")

    pending_coll = db.goout_pending if db is not None else None
    parties_coll = db.parties if db is not None else None

    async def _scrape_one_account(account: GoOutAccount) -> int:
        scraper = GoOutScraper(account, db=db, telegram_mgr=telegram_mgr)
        new_count = 0
        try:
            session_ok = await scraper.ensure_session()
            if not session_ok:
                if telegram_mgr:
                    telegram_mgr.send_message_sync(
                        f"⚠️ Could not log in to *{account.account_id}*. Skipping."
                    )
                return 0

            event_entries = await scraper.discover_events()
            logger.info(f"[{account.account_id}] Discovered {len(event_entries)} event(s)")

            for entry in event_entries:
                event_url = entry.get("url", "")
                if not event_url:
                    continue

                canonical = normalize_url(event_url)

                if not force_send:
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

                # Scrape full event details using existing scraper
                party_data = None
                try:
                    party_data = scrape_party_details(event_url)
                except Exception as exc:
                    logger.warning(f"Failed to scrape {event_url}: {exc}")

                # If full scrape failed, use metadata from discovery
                if not party_data:
                    logger.info(f"Using discovery metadata for {event_url}")
                    party_data = {
                        "name": entry.get("name") or "Unknown Event",
                        "goOutUrl": canonical,
                        "canonicalUrl": canonical,
                        "date": entry.get("date") or datetime.now().strftime("%Y-%m-%d"),
                        "image": entry.get("image") or "",
                        "source": "go-out",
                    }

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

                        if not force_send:
                            pending_coll.insert_one(pending_doc)

                        if telegram_mgr:
                            telegram_mgr.send_party_for_approval_sync(pending_doc)
                        new_count += 1
                    except Exception as exc:
                        logger.error(f"Failed to handle party: {exc}")

                await asyncio.sleep(2)

            logger.info(f"[{account.account_id}] Processed {new_count} new events")
        except Exception as exc:
            logger.error(f"[{account.account_id}] Scrape error: {exc}")
            if telegram_mgr:
                telegram_mgr.send_message_sync(
                    f"❌ Error scraping *{account.account_id}*: {exc}"
                )
        finally:
            await scraper.close()
        return new_count

    # Run all accounts in parallel — each has its own browser context so they don't interfere
    results = await asyncio.gather(*[_scrape_one_account(a) for a in accounts], return_exceptions=True)
    total_new = sum(r for r in results if isinstance(r, int))

    if telegram_mgr:
        # Use direct requests for the final summary to be sure it arrives 
        # even if there's a bot conflict (multiple instances).
        summary = (
            f"✅ *Daily scrape complete!*\n"
            f"Found {total_new} new events across {len(accounts)} accounts."
        )
        try:
            import requests
            token = telegram_mgr.bot_token
            chat_id = telegram_mgr.manager_chat_id
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": summary, "parse_mode": "Markdown"}, timeout=10)
        except Exception:
            # Fallback to standard sync send
            telegram_mgr.send_message_sync(summary)
