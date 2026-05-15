"""
Telegram Manager Bot for Parties 24/7.

Handles:
- 2FA code relay for Go-Out login
- New party approval notifications with inline buttons
- Manager commands (/status, /scrape, /pending, /approve_all, /sessions)
"""

import os
import json
import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Callable

from telegram import (
    Bot,
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State containers for cross-thread communication
# ---------------------------------------------------------------------------

_tfa_requests: dict[str, asyncio.Event] = {}
_tfa_codes: dict[str, str | None] = {}
_edit_sessions: dict[int, str] = {}  # chat_msg_id -> pending_id being edited


class TelegramManager:
    """Async Telegram bot running in its own thread alongside Flask."""

    def __init__(
        self,
        token: str,
        manager_chat_id: str,
        db_getter: Callable | None = None,
    ):
        self.token = token
        self.manager_chat_id = int(manager_chat_id)
        self._db_getter = db_getter
        self._app: Application | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = False
        # Callback for triggering a scrape from the /scrape command
        self.on_scrape_requested: Callable | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _pending_collection(self):
        if self._db_getter:
            db = self._db_getter()
            if db is not None:
                return db.goout_pending
        return None

    @property
    def _sessions_collection(self):
        if self._db_getter:
            db = self._db_getter()
            if db is not None:
                return db.goout_sessions
        return None

    @property
    def _parties_collection(self):
        if self._db_getter:
            db = self._db_getter()
            if db is not None:
                return db.parties
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_in_background(self):
        """Launch the Telegram bot in a background daemon thread."""
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._run_forever, daemon=True)
        self._thread.start()
        logger.info("Telegram bot started in background thread.")

    def _run_forever(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start_polling())

    async def _start_polling(self):
        builder = Application.builder().token(self.token)
        self._app = builder.build()

        # Register handlers
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("scrape", self._cmd_scrape))
        self._app.add_handler(CommandHandler("pending", self._cmd_pending))
        self._app.add_handler(CommandHandler("approve_all", self._cmd_approve_all))
        self._app.add_handler(CommandHandler("sessions", self._cmd_sessions))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(
            CallbackQueryHandler(self._handle_callback)
        )
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text)
        )

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

        # Keep running until cancelled
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass
        finally:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def _is_manager(self, update: Update) -> bool:
        return update.effective_chat and update.effective_chat.id == self.manager_chat_id

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_manager(update):
            await update.message.reply_text("⛔ Unauthorized.")
            return
        await update.message.reply_text(
            "🎉 *Parties 24/7 Manager Bot*\n\n"
            "Use /help to see available commands.",
            parse_mode="Markdown",
        )

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_manager(update):
            return
        await update.message.reply_text(
            "📋 *Available Commands*\n\n"
            "/status — Scraper status & pending count\n"
            "/scrape — Trigger immediate scrape\n"
            "/pending — List pending parties\n"
            "/approve\\_all — Approve all pending\n"
            "/sessions — Go-Out session status\n"
            "/help — Show this message",
            parse_mode="Markdown",
        )

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_manager(update):
            return
        pending_count = 0
        sessions_info = []
        try:
            coll = self._pending_collection
            if coll is not None:
                pending_count = coll.count_documents({"status": "pending"})
        except Exception as exc:
            logger.warning(f"Failed to count pending: {exc}")

        try:
            sess_coll = self._sessions_collection
            if sess_coll is not None:
                for doc in sess_coll.find({}):
                    sessions_info.append(
                        f"  • {doc.get('account_id', '?')}: "
                        f"{'✅ Valid' if doc.get('session_valid') else '❌ Expired'} "
                        f"(last: {doc.get('last_checked', 'never')})"
                    )
        except Exception as exc:
            logger.warning(f"Failed to read sessions: {exc}")

        sessions_text = "\n".join(sessions_info) if sessions_info else "  No sessions found."
        await update.message.reply_text(
            f"📊 *Scraper Status*\n\n"
            f"Pending parties: {pending_count}\n"
            f"Sessions:\n{sessions_text}",
            parse_mode="Markdown",
        )

    async def _cmd_scrape(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_manager(update):
            return
        await update.message.reply_text("🔄 Triggering scrape... This may take a few minutes.")
        if self.on_scrape_requested:
            try:
                self.on_scrape_requested()
            except Exception as exc:
                await update.message.reply_text(f"❌ Scrape failed: {exc}")

    async def _cmd_pending(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_manager(update):
            return
        coll = self._pending_collection
        if coll is None:
            await update.message.reply_text("⚠️ Database unavailable.")
            return
        try:
            docs = list(coll.find({"status": "pending"}).limit(20))
        except Exception as exc:
            await update.message.reply_text(f"❌ Error: {exc}")
            return
        if not docs:
            await update.message.reply_text("✅ No pending parties!")
            return
        for doc in docs:
            await self._send_pending_party_message(doc)

    async def _cmd_approve_all(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_manager(update):
            return
        coll = self._pending_collection
        if coll is None:
            await update.message.reply_text("⚠️ Database unavailable.")
            return
        try:
            docs = list(coll.find({"status": "pending"}))
        except Exception:
            docs = []
        if not docs:
            await update.message.reply_text("✅ No pending parties to approve.")
            return
        approved = 0
        for doc in docs:
            success = self._approve_pending_party(str(doc["_id"]))
            if success:
                approved += 1
        await update.message.reply_text(
            f"✅ Approved {approved}/{len(docs)} parties."
        )

    async def _cmd_sessions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_manager(update):
            return
        sess_coll = self._sessions_collection
        if sess_coll is None:
            await update.message.reply_text("⚠️ Database unavailable.")
            return
        try:
            docs = list(sess_coll.find({}))
        except Exception as exc:
            await update.message.reply_text(f"❌ Error: {exc}")
            return
        if not docs:
            await update.message.reply_text("No sessions stored yet.")
            return
        lines = []
        for doc in docs:
            valid = "✅ Valid" if doc.get("session_valid") else "❌ Expired"
            last_login = doc.get("last_login", "never")
            last_checked = doc.get("last_checked", "never")
            lines.append(
                f"*{doc.get('account_id', '?')}*\n"
                f"  Email: {doc.get('email', '?')}\n"
                f"  Status: {valid}\n"
                f"  Last login: {last_login}\n"
                f"  Last checked: {last_checked}"
            )
        await update.message.reply_text(
            "🔑 *Go-Out Sessions*\n\n" + "\n\n".join(lines),
            parse_mode="Markdown",
        )

    # ------------------------------------------------------------------
    # Callback query handler (inline buttons)
    # ------------------------------------------------------------------

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        if not query:
            return
        await query.answer()

        if not self._is_manager(update):
            await query.edit_message_text("⛔ Unauthorized.")
            return

        data = query.data or ""

        if data.startswith("approve:"):
            pending_id = data.split(":", 1)[1]
            success = self._approve_pending_party(pending_id)
            if success:
                await query.edit_message_text(
                    query.message.text + "\n\n✅ *APPROVED*",
                    parse_mode="Markdown",
                )
            else:
                await query.edit_message_text(
                    query.message.text + "\n\n❌ *Failed to approve*",
                    parse_mode="Markdown",
                )

        elif data.startswith("reject:"):
            pending_id = data.split(":", 1)[1]
            success = self._reject_pending_party(pending_id)
            if success:
                await query.edit_message_text(
                    query.message.text + "\n\n❌ *REJECTED*",
                    parse_mode="Markdown",
                )
            else:
                await query.edit_message_text(
                    query.message.text + "\n\n⚠️ *Failed to reject*",
                    parse_mode="Markdown",
                )

        elif data.startswith("edit:"):
            pending_id = data.split(":", 1)[1]
            _edit_sessions[update.effective_chat.id] = pending_id
            await query.edit_message_text(
                query.message.text
                + "\n\n✏️ *EDIT MODE*\n"
                "Send me the fields to change as JSON, e.g.:\n"
                '`{"name": "New Name", "location": "Tel Aviv"}`\n\n'
                "Available fields: name, description, date, location, "
                "tags, referralCode, musicType, eventType, region, age",
                parse_mode="Markdown",
            )

        elif data.startswith("2fa:"):
            # 2FA availability confirmation
            parts = data.split(":", 2)
            account_id = parts[1] if len(parts) > 1 else ""
            action = parts[2] if len(parts) > 2 else ""
            if action == "ready":
                await query.edit_message_text(
                    f"🔐 Great! I'll send the 2FA code request for *{account_id}* now.\n"
                    "Please reply with the 6-digit code when you receive it.",
                    parse_mode="Markdown",
                )
                # Signal that manager is available
                key = f"2fa_avail_{account_id}"
                if key in _tfa_requests:
                    _tfa_codes[key] = "ready"
                    _tfa_requests[key].set()
            elif action == "later":
                await query.edit_message_text(
                    f"⏳ OK, I'll try again later for *{account_id}*.",
                    parse_mode="Markdown",
                )
                key = f"2fa_avail_{account_id}"
                if key in _tfa_requests:
                    _tfa_codes[key] = None
                    _tfa_requests[key].set()

    # ------------------------------------------------------------------
    # Text message handler (for 2FA codes and edit JSON)
    # ------------------------------------------------------------------

    async def _handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_manager(update):
            return
        text = (update.message.text or "").strip()
        chat_id = update.effective_chat.id

        # Check if this is a 2FA code (6 digits)
        if text.isdigit() and len(text) == 6:
            # Find any pending 2FA request
            for key, event in list(_tfa_requests.items()):
                if key.startswith("2fa_code_") and not event.is_set():
                    _tfa_codes[key] = text
                    event.set()
                    await update.message.reply_text(
                        f"✅ 2FA code received: `{text}`. Attempting login...",
                        parse_mode="Markdown",
                    )
                    return
            await update.message.reply_text(
                "ℹ️ No pending 2FA request. Code ignored."
            )
            return

        # Check if this is an edit session
        if chat_id in _edit_sessions:
            pending_id = _edit_sessions.pop(chat_id)
            try:
                edits = json.loads(text)
                if not isinstance(edits, dict):
                    raise ValueError("Must be a JSON object")
                success = self._edit_pending_party(pending_id, edits)
                if success:
                    await update.message.reply_text("✅ Party updated and approved!")
                else:
                    await update.message.reply_text("❌ Failed to update party.")
            except (json.JSONDecodeError, ValueError) as exc:
                await update.message.reply_text(
                    f"❌ Invalid JSON: {exc}\n"
                    "Please send valid JSON or use /pending to try again."
                )
                return

    # ------------------------------------------------------------------
    # Public API — called from scraper / scheduler
    # ------------------------------------------------------------------

    def send_message_sync(self, text: str, parse_mode: str = "Markdown"):
        """Send a message to the manager (thread-safe)."""
        if not self._loop or not self._app:
            logger.warning("Telegram bot not started; cannot send message.")
            return
        asyncio.run_coroutine_threadsafe(
            self._send_text(text, parse_mode), self._loop
        )

    async def _send_text(self, text: str, parse_mode: str = "Markdown"):
        try:
            await self._app.bot.send_message(
                chat_id=self.manager_chat_id,
                text=text,
                parse_mode=parse_mode,
            )
        except Exception as exc:
            logger.error(f"Failed to send Telegram message: {exc}")

    def send_party_for_approval_sync(self, pending_doc: dict):
        """Send a pending party to the manager for approval (thread-safe)."""
        if not self._loop or not self._app:
            logger.warning("Telegram bot not started; cannot send party.")
            return
        asyncio.run_coroutine_threadsafe(
            self._send_pending_party_message(pending_doc), self._loop
        )

    async def _send_pending_party_message(self, pending_doc: dict):
        """Format and send a pending party message with inline buttons."""
        party = pending_doc.get("party_data", {})
        pending_id = str(pending_doc.get("_id", ""))
        account = pending_doc.get("account_id", "?")

        name = party.get("name", "Unknown Party")
        date_str = party.get("date", "Unknown Date")
        location = party.get("location", "Unknown Location")
        price = party.get("ticketPrice")
        sold_out = party.get("soldOut", False)
        url = party.get("originalUrl") or party.get("goOutUrl", "")
        image_url = party.get("imageUrl", "")

        price_text = "🎫 Sold Out" if sold_out else (
            f"💰 ₪{price:.0f}" if price else "💰 Free / Unknown"
        )

        text = (
            f"🎉 *New Party Found!*\n"
            f"Account: {account}\n\n"
            f"📛 *{self._escape_md(name)}*\n"
            f"📅 {self._escape_md(str(date_str))}\n"
            f"📍 {self._escape_md(location)}\n"
            f"{price_text}\n"
            f"🔗 [Go-Out Link]({url})"
        )

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Approve", callback_data=f"approve:{pending_id}"),
                InlineKeyboardButton("✏️ Edit", callback_data=f"edit:{pending_id}"),
                InlineKeyboardButton("❌ Reject", callback_data=f"reject:{pending_id}"),
            ]
        ])

        try:
            if image_url:
                try:
                    await self._app.bot.send_photo(
                        chat_id=self.manager_chat_id,
                        photo=image_url,
                        caption=text,
                        parse_mode="Markdown",
                        reply_markup=keyboard,
                    )
                    return
                except Exception:
                    pass  # Fall back to text-only
            await self._app.bot.send_message(
                chat_id=self.manager_chat_id,
                text=text,
                parse_mode="Markdown",
                reply_markup=keyboard,
                disable_web_page_preview=False,
            )
        except Exception as exc:
            logger.error(f"Failed to send party for approval: {exc}")

    # ------------------------------------------------------------------
    # 2FA coordination
    # ------------------------------------------------------------------

    def ask_2fa_availability_sync(self, account_id: str, timeout: float = 600) -> bool:
        """
        Ask the manager if they're available for 2FA. Returns True if ready.
        Blocks the calling thread until the manager responds or timeout.
        """
        if not self._loop or not self._app:
            return False
        future = asyncio.run_coroutine_threadsafe(
            self._ask_2fa_availability(account_id, timeout), self._loop
        )
        try:
            return future.result(timeout=timeout + 30)
        except Exception:
            return False

    async def _ask_2fa_availability(self, account_id: str, timeout: float) -> bool:
        key = f"2fa_avail_{account_id}"
        event = asyncio.Event()
        _tfa_requests[key] = event
        _tfa_codes[key] = None

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton(
                    "✅ I'm available",
                    callback_data=f"2fa:{account_id}:ready",
                ),
                InlineKeyboardButton(
                    "⏳ Not now",
                    callback_data=f"2fa:{account_id}:later",
                ),
            ]
        ])

        try:
            await self._app.bot.send_message(
                chat_id=self.manager_chat_id,
                text=(
                    f"🔐 *2FA Required for {account_id}*\n\n"
                    "The Go-Out session has expired and needs re-authentication.\n"
                    "Are you available to enter the 2FA code?"
                ),
                parse_mode="Markdown",
                reply_markup=keyboard,
            )
        except Exception as exc:
            logger.error(f"Failed to ask 2FA availability: {exc}")
            return False

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            _tfa_requests.pop(key, None)
            _tfa_codes.pop(key, None)
            return False

        result = _tfa_codes.pop(key, None)
        _tfa_requests.pop(key, None)
        return result == "ready"

    def request_2fa_code_sync(self, account_id: str, timeout: float = 600) -> str | None:
        """
        Request the 2FA code from the manager. Blocks until received or timeout.
        """
        if not self._loop or not self._app:
            return None
        future = asyncio.run_coroutine_threadsafe(
            self._request_2fa_code(account_id, timeout), self._loop
        )
        try:
            return future.result(timeout=timeout + 30)
        except Exception:
            return None

    async def _request_2fa_code(self, account_id: str, timeout: float) -> str | None:
        key = f"2fa_code_{account_id}"
        event = asyncio.Event()
        _tfa_requests[key] = event
        _tfa_codes[key] = None

        try:
            await self._app.bot.send_message(
                chat_id=self.manager_chat_id,
                text=(
                    f"🔢 *Enter 2FA Code for {account_id}*\n\n"
                    "Please check your email/SMS and reply with the 6-digit code."
                ),
                parse_mode="Markdown",
            )
        except Exception as exc:
            logger.error(f"Failed to request 2FA code: {exc}")
            return None

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            _tfa_requests.pop(key, None)
            _tfa_codes.pop(key, None)
            try:
                await self._app.bot.send_message(
                    chat_id=self.manager_chat_id,
                    text=f"⏰ 2FA code request for *{account_id}* timed out.",
                    parse_mode="Markdown",
                )
            except Exception:
                pass
            return None

        code = _tfa_codes.pop(key, None)
        _tfa_requests.pop(key, None)
        return code

    # ------------------------------------------------------------------
    # Party approval/rejection helpers
    # ------------------------------------------------------------------

    def _approve_pending_party(self, pending_id: str) -> bool:
        """Move a pending party into the main parties collection."""
        from bson.objectid import ObjectId
        coll = self._pending_collection
        parties_coll = self._parties_collection
        if coll is None or parties_coll is None:
            return False
        try:
            from app import (
                normalize_url, normalized_or_none_for_dedupe,
                apply_default_referral, slugify_party,
                normalize_event, notify_indexers, trigger_revalidation,
                event_related_paths,
            )
        except ImportError:
            logger.error("Cannot import app helpers for party approval")
            return False

        try:
            doc = coll.find_one({"_id": ObjectId(pending_id)})
        except Exception:
            return False
        if not doc or doc.get("status") != "pending":
            return False

        party_data = doc.get("party_data", {})
        if not party_data:
            return False

        # Ensure slug
        party_data.setdefault(
            "slug", slugify_party(party_data.get("name"), party_data.get("date"))
        )

        canonical = party_data.get("canonicalUrl")
        go_out_url = party_data.get("goOutUrl")

        # Check if already exists
        existing_query = {}
        or_clauses = []
        if canonical:
            or_clauses.append({"canonicalUrl": canonical})
        if go_out_url:
            or_clauses.append({"goOutUrl": go_out_url})
        if or_clauses:
            existing_query = {"$or": or_clauses}
        else:
            existing_query = {"name": party_data.get("name"), "date": party_data.get("date")}

        try:
            result = parties_coll.update_one(
                existing_query,
                {"$setOnInsert": party_data},
                upsert=True,
            )
        except Exception as exc:
            logger.error(f"Failed to insert approved party: {exc}")
            return False

        # Mark as approved
        try:
            coll.update_one(
                {"_id": ObjectId(pending_id)},
                {"$set": {
                    "status": "approved",
                    "approved_at": datetime.now(timezone.utc),
                }},
            )
        except Exception:
            pass

        # Trigger indexing
        try:
            event_view = normalize_event(party_data)
            notify_indexers([event_view.get("canonicalUrl")])
            trigger_revalidation(event_related_paths(event_view))
        except Exception as exc:
            logger.warning(f"Post-approval indexing failed: {exc}")

        return True

    def _reject_pending_party(self, pending_id: str) -> bool:
        """Mark a pending party as rejected."""
        from bson.objectid import ObjectId
        coll = self._pending_collection
        if coll is None:
            return False
        try:
            result = coll.update_one(
                {"_id": ObjectId(pending_id)},
                {"$set": {
                    "status": "rejected",
                    "rejected_at": datetime.now(timezone.utc),
                }},
            )
            return result.modified_count > 0
        except Exception:
            return False

    def _edit_pending_party(self, pending_id: str, edits: dict) -> bool:
        """Apply edits to a pending party and approve it."""
        from bson.objectid import ObjectId
        coll = self._pending_collection
        if coll is None:
            return False

        allowed_fields = {
            "name", "description", "date", "location", "tags",
            "referralCode", "musicType", "eventType", "region", "age",
        }
        filtered = {k: v for k, v in edits.items() if k in allowed_fields}
        if not filtered:
            return False

        try:
            doc = coll.find_one({"_id": ObjectId(pending_id)})
        except Exception:
            return False
        if not doc:
            return False

        party_data = doc.get("party_data", {})
        party_data.update(filtered)

        try:
            coll.update_one(
                {"_id": ObjectId(pending_id)},
                {"$set": {"party_data": party_data}},
            )
        except Exception:
            return False

        return self._approve_pending_party(pending_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _escape_md(text: str) -> str:
        """Escape Markdown special characters for Telegram."""
        if not text:
            return ""
        for ch in ("_", "*", "[", "]", "(", ")", "~", "`", ">", "#", "+", "-", "=", "|", "{", "}", ".", "!"):
            text = text.replace(ch, f"\\{ch}")
        return text
