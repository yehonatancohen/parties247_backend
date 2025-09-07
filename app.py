import os
import json
import logging
from functools import wraps
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient, errors
from pymongo.collection import Collection
from bson.objectid import ObjectId
import requests
from bs4 import BeautifulSoup

# --- Initial Setup ---
load_dotenv()
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

# --- URL canonicalization ---
TRACKING_PREFIXES = ("utm_",)
TRACKING_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid"}

def normalize_url(raw: str) -> str:
    p = urlparse((raw or "").strip())
    scheme = (p.scheme or "https").lower()
    netloc = p.netloc.lower()

    # drop default ports
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    if netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    # normalize path
    path = p.path.rstrip("/") or "/"

    # drop fragments
    fragment = ""

    # drop tracking params and keep stable order
    cleaned_qs = []
    for k, v in parse_qsl(p.query, keep_blank_values=True):
        kl = k.lower()
        if kl.startswith(TRACKING_PREFIXES) or kl in TRACKING_KEYS:
            continue
        cleaned_qs.append((k, v))
    query = urlencode(cleaned_qs, doseq=True)

    return urlunparse((scheme, netloc, path, "", query, fragment))


# --- MongoDB ---
try:
    client = MongoClient(os.environ.get("MONGODB_URI"))
    db = client["party247"]
    parties_collection: Collection = db.parties

    # Drop legacy unique index on originalUrl if present
    try:
        idx_info = parties_collection.index_information()
        for name, meta in idx_info.items():
            key = meta.get("key", [])
            if key == [("originalUrl", 1)] and meta.get("unique", False):
                parties_collection.drop_index(name)
                app.logger.info("Dropped legacy unique index on originalUrl")
    except Exception as e:
        app.logger.warning(f"Index inspection/drop skipped: {e}")

    # Create safe unique index on canonicalUrl (partial)
    parties_collection.create_index(
        "canonicalUrl",
        name="unique_canonicalUrl",
        unique=True,
        partialFilterExpression={"canonicalUrl": {"$exists": True, "$type": "string"}}
    )

    # Optional helper index for sorting by date
    parties_collection.create_index([("date", 1)], name="date_asc")

    # One-time light backfill of canonicalUrl for existing docs
    try:
        cursor = parties_collection.find(
            {"canonicalUrl": {"$exists": False}, "originalUrl": {"$type": "string"}},
            {"_id": 1, "originalUrl": 1},
            no_cursor_timeout=True,
        )
        for doc in cursor:
            try:
                parties_collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"canonicalUrl": normalize_url(doc.get("originalUrl", ""))}}
                )
            except Exception as inner:
                app.logger.warning(f"Backfill failed for {doc.get('_id')}: {inner}")
        cursor.close()
    except Exception as e:
        app.logger.warning(f"Backfill pass skipped: {e}")

    app.logger.info("Connected to MongoDB and ensured indexes.")
except Exception as e:
    app.logger.error(f"Error connecting to MongoDB Atlas: {e}")
    raise


# --- Security Decorator ---
def protect(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        provided_key = request.headers.get("x-admin-secret-key")
        if provided_key and provided_key == os.environ.get("ADMIN_SECRET_KEY"):
            return f(*args, **kwargs)
        return jsonify({"message": "Forbidden: Invalid or missing admin key."}), 403
    return decorated_function


# --- Classification Helpers ---
def get_region(location: str) -> str:
    south_keywords = ["באר שבע", "אילת", "אשדוד", "אשקלון", "דרום"]
    north_keywords = ["חיפה", "טבריה", "צפון", "קריות", "כרמיאל", "עכו"]
    center_keywords = ["תל אביב", "ירושלים", "ראשון לציון", "הרצליה", "נתניה", "מרכז", "י-ם", "tlv"]
    loc = (location or "").lower()
    if any(k in loc for k in south_keywords):
        return "דרום"
    if any(k in loc for k in north_keywords):
        return "צפון"
    if any(k in loc for k in center_keywords):
        return "מרכז"
    return "לא ידוע"

def get_music_type(text: str) -> str:
    techno_keywords = ["טכנו", "techno", "after", "אפטר", "house", "האוס", "electronic", "אלקטרונית"]
    trance_keywords = ["טראנס", "trance", "פסיי", "psy-trance", "psytrance"]
    mainstream_keywords = ["מיינסטרים", "mainstream", "היפ הופ", "hip hop", "רגאטון", "reggaeton", "pop", "פופ"]
    txt = (text or "").lower()
    if any(k in txt for k in techno_keywords):
        return "טכנו"
    if any(k in txt for k in trance_keywords):
        return "טראנס"
    if any(k in txt for k in mainstream_keywords):
        return "מיינסטרים"
    return "אחר"

def get_event_type(text: str) -> str:
    festival_keywords = ["פסטיבל", "festival"]
    nature_keywords = ["טבע", "nature", "יער", "forest", "חוף", "beach", "open air", "בחוץ"]
    club_keywords = ["מועדון", "club", "גגרין", "בלוק", "האומן 17", "gagarin", "block", "haoman 17", "rooftop", "גג"]
    txt = (text or "").lower()
    if any(k in txt for k in festival_keywords):
        return "פסטיבל"
    if any(k in txt for k in nature_keywords):
        return "מסיבת טבע"
    if any(k in txt for k in club_keywords):
        return "מסיבת מועדון"
    return "אחר"

def get_age(text: str, minimum_age: int) -> str:
    if minimum_age >= 21:
        return "21+"
    if minimum_age >= 18:
        return "18+"
    if "נוער" in (text or "").lower():
        return "נוער"
    if minimum_age > 0:
        return "18+"
    return "כל הגילאים"

def get_tags(text: str, location: str) -> list:
    tags = []
    tag_map = {
        "אלכוהול חופשי": ["אלכוהול חופשי", "free alcohol", "בר חופשי", "free bar"],
        "בחוץ": ["open air", "בחוץ", "טבע", "חוף", "יער", "rooftop", "גג"],
        "אילת": ["אילת", "eilat"],
        "תל אביב": ["תל אביב", "tel aviv", "tlv"],
    }
    combined_text = f"{text or ''} {location or ''}".lower()
    for tag, keywords in tag_map.items():
        if any(keyword in combined_text for keyword in keywords):
            tags.append(tag)
    return list(set(tags))


# --- Main Scraping Function with Logging ---
def scrape_party_details(url: str):
    app.logger.info(f"[SCRAPER_LOG] --- Starting scrape for URL: {url} ---")
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=15)
        app.logger.info(f"[SCRAPER_LOG] HTTP Response Status: {response.status_code}")
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        app.logger.info("[SCRAPER_LOG] Searching for '__NEXT_DATA__' script tag...")
        script_tag = soup.find("script", {"id": "__NEXT_DATA__"})
        if not script_tag:
            app.logger.error("[SCRAPER_ERROR] CRITICAL: Could not find '__NEXT_DATA__' script tag in HTML.")
            raise ValueError("Could not find party data script (__NEXT_DATA__).")
        app.logger.info("[SCRAPER_LOG] Found '__NEXT_DATA__' script tag.")

        app.logger.info("[SCRAPER_LOG] Parsing JSON data...")
        json_data = json.loads(script_tag.string)
        app.logger.info("[SCRAPER_LOG] JSON parsed successfully.")

        event_data = json_data.get("props", {}).get("pageProps", {}).get("event")
        if not event_data:
            app.logger.error("[SCRAPER_ERROR] CRITICAL: 'event' object not found in the expected JSON path.")
            raise ValueError("Event data not in expected format inside JSON.")
        app.logger.info("[SCRAPER_LOG] Found 'event' object in JSON data.")

        # Image extraction logic
        image_path = ""
        if event_data.get("CoverImage") and event_data["CoverImage"].get("Url"):
            image_path = event_data["CoverImage"]["Url"]
        elif event_data.get("WhatsappImage") and event_data["WhatsappImage"].get("Url"):
            image_path = event_data["WhatsappImage"]["Url"]

        app.logger.info(f"[SCRAPER_LOG] Initial image path found: {image_path if image_path else 'None'}")

        if image_path:
            cover_image_path = image_path.replace("_whatsappImage.jpg", "_coverImage.jpg")
            image_url = f"https://d15q6k8l9pfut7.cloudfront.net/{cover_image_path}"
        else:
            app.logger.warning("[SCRAPER_LOG] No image path in JSON, falling back to meta tag.")
            og_image_tag = soup.find("meta", {"property": "og:image"})
            og_image_url = og_image_tag["content"] if og_image_tag else ""
            if og_image_url:
                image_url = og_image_url.replace("_whatsappImage.jpg", "_coverImage.jpg")
            else:
                app.logger.error("[SCRAPER_ERROR] CRITICAL: Could not find image URL in JSON or meta tags.")
                raise ValueError("Could not find party image URL.")

        app.logger.info(f"[SCRAPER_LOG] Final image URL: {image_url}")

        description = event_data.get("Description", "")
        cleaned_desc = " ".join(list(filter(None, description.split("\n")))[:3]).strip()
        if len(cleaned_desc) > 250:
            cleaned_desc = cleaned_desc[:247] + "..."
        app.logger.info("[SCRAPER_LOG] Cleaned description successfully.")

        full_text = f"{event_data.get('Title', '')} {description}"
        location = event_data.get("Adress", "")

        party_details = {
            "name": event_data.get("Title"),
            "imageUrl": image_url,
            "date": event_data.get("StartingDate"),
            "location": location,
            "description": cleaned_desc or "No description available.",
            "region": get_region(location),
            "musicType": get_music_type(full_text),
            "eventType": get_event_type(full_text),
            "age": get_age(full_text, event_data.get("MinimumAge", 0)),
            "tags": get_tags(full_text, location),
            "originalUrl": url,
            "canonicalUrl": normalize_url(url),
        }
        app.logger.info(f"[SCRAPER_LOG] Assembled party object: {party_details['name']}")

        if not all([party_details["name"], party_details["date"], party_details["location"]]):
            app.logger.error("[SCRAPER_ERROR] CRITICAL: Final object is missing name, date, or location.")
            raise ValueError("Scraped data is missing critical fields.")

        return party_details

    except requests.exceptions.RequestException as e:
        app.logger.error(f"[SCRAPER_ERROR] HTTP Request failed: {e}")
        raise
    except Exception as e:
        app.logger.error(f"[SCRAPER_ERROR] An unexpected error occurred during scraping: {e}")
        raise


# --- API Routes ---
@app.route("/api/admin/add-party", methods=["POST"])
@protect
def add_party():
    data = request.get_json(silent=True) or {}
    url = data.get("url")
    if not url:
        return jsonify({"message": "URL is required."}), 400

    try:
        party_data = scrape_party_details(url)
        # Idempotent insert using canonicalUrl
        res = parties_collection.update_one(
            {"canonicalUrl": party_data["canonicalUrl"]},
            {"$setOnInsert": party_data},
            upsert=True,
        )

        if res.matched_count == 1 and res.upserted_id is None:
            existing = parties_collection.find_one(
                {"canonicalUrl": party_data["canonicalUrl"]},
                {"_id": 1}
            )
            return jsonify({
                "message": "This party has already been added.",
                "id": str(existing["_id"])
            }), 409

        party_data["_id"] = str(res.upserted_id)
        app.logger.info(f"[DB_LOG] Inserted party with ID: {party_data['_id']}")
        return jsonify({"message": "Party added successfully!", "party": party_data}), 201

    except errors.DuplicateKeyError as e:
        app.logger.warning(f"[DB_ERROR] DuplicateKeyError: {getattr(e, 'details', None)}")
        return jsonify({"message": "This party has already been added."}), 409
    except Exception as e:
        app.logger.error(f"[DB_ERROR] {str(e)}")
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500


@app.route("/api/admin/delete-party/<party_id>", methods=["DELETE"])
@protect
def delete_party(party_id):
    try:
        obj_id = ObjectId(party_id)
        result = parties_collection.delete_one({"_id": obj_id})
        if result.deleted_count == 0:
            return jsonify({"message": "Party not found."}), 404
        return jsonify({"message": "Party deleted successfully!"}), 200
    except Exception as e:
        return jsonify({"message": "Error deleting party", "error": str(e)}), 500


@app.route("/api/parties", methods=["GET"])
def get_parties():
    try:
        all_parties = []
        for party in parties_collection.find().sort("date", 1):
            party["_id"] = str(party["_id"])
            all_parties.append(party)
        return jsonify(all_parties), 200
    except Exception as e:
        return jsonify({"message": "Error fetching parties", "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 3001)))