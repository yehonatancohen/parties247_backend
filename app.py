import os
import json
import logging
import hmac
from datetime import datetime, timedelta
from functools import wraps
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
import socket
import ipaddress

import bcrypt
import jwt
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pymongo import MongoClient, errors
from pymongo.collection import Collection
from bson.objectid import ObjectId
import requests
from bs4 import BeautifulSoup

# --- App setup ---
load_dotenv()
app = Flask(__name__)
ALLOWED_ORIGINS = [
    "http://localhost:3000",  # Local development
    "https://parties247-website.vercel.app/",  # Staging
    "https://parties247.com",  # Production
]
app.config["RATELIMIT_HEADERS_ENABLED"] = True
CORS(app, origins=ALLOWED_ORIGINS, supports_credentials=False)
limiter = Limiter(get_remote_address, app=app)
logging.basicConfig(level=logging.INFO)

# --- URL canonicalization ---
TRACKING_PREFIXES = ("utm_",)
TRACKING_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid"}

def normalize_url(raw: str) -> str:
    p = urlparse((raw or "").strip())
    scheme = (p.scheme or "https").lower()
    netloc = p.netloc.lower()
    # strip default ports
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    if netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]
    path = p.path.rstrip("/") or "/"
    # drop tracking params
    cleaned_qs = []
    for k, v in parse_qsl(p.query, keep_blank_values=True):
        kl = k.lower()
        if kl.startswith(TRACKING_PREFIXES) or kl in TRACKING_KEYS:
            continue
        cleaned_qs.append((k, v))
    query = urlencode(cleaned_qs, doseq=True)
    return urlunparse((scheme, netloc, path, "", query, ""))

def normalized_or_none_for_dedupe(raw: str) -> str | None:
    n = normalize_url(raw)
    # require a host to be considered valid
    return n if urlparse(n).netloc else None

ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_PORTS = {80, 443}


def is_url_allowed(raw: str) -> bool:
    try:
        parsed = urlparse(raw)
    except Exception:
        return False

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return False

    if parsed.port and parsed.port not in ALLOWED_PORTS:
        return False

    host = parsed.hostname
    if not host:
        return False

    try:
        infos = socket.getaddrinfo(host, parsed.port or (80 if parsed.scheme == "http" else 443))
    except socket.gaierror:
        return False

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local or ip.is_multicast:
            return False

    return True

# --- Mongo helpers ---
def ensure_index(coll: Collection, keys, name: str, **kwargs):
    info = coll.index_information()
    if name in info:
        meta = info[name]
        same_keys = meta.get("key") == list(keys)
        same_unique = bool(meta.get("unique", False)) == bool(kwargs.get("unique", False))
        same_pfe = meta.get("partialFilterExpression") == kwargs.get("partialFilterExpression")
        if same_keys and same_unique and same_pfe:
            return
        try:
            coll.drop_index(name)
        except Exception as e:
            app.logger.warning(f"Drop index {name} failed: {e}")
    coll.create_index(list(keys), name=name, **kwargs)

# --- Mongo connection + index hygiene ---
try:
    client = MongoClient(os.environ.get("MONGODB_URI"))
    db = client["party247"]
    parties_collection: Collection = db.parties
    carousels_collection: Collection = db.carousels # New collection for carousels

    # Cleanup bad goOutUrl values before enforcing unique indexes
    try:
        parties_collection.update_many(
            {"$or": [{"goOutUrl": None}, {"goOutUrl": ""}]},
            {"$unset": {"goOutUrl": ""}}
        )
    except Exception as e:
        app.logger.warning(f"goOutUrl cleanup failed: {e}")

    # Drop any legacy unique index on originalUrl
    try:
        for name, meta in parties_collection.index_information().items():
            if meta.get("unique") and meta.get("key") == [("originalUrl", 1)]:
                parties_collection.drop_index(name)
                app.logger.info(f"Dropped legacy unique index on originalUrl: {name}")
    except Exception as e:
        app.logger.warning(f"Drop originalUrl index failed: {e}")

    # Drop any legacy indexes on goOutUrl (including ones with unsupported partial specs)
    try:
        for name, meta in parties_collection.index_information().items():
            if meta.get("key") == [("goOutUrl", 1)]:
                parties_collection.drop_index(name)
                app.logger.info(f"Dropped legacy index on goOutUrl: {name}")
    except Exception as e:
        app.logger.warning(f"Drop goOutUrl index failed: {e}")

    # Ensure canonicalUrl unique partial index (allow only when field exists and is string)
    ensure_index(
        parties_collection,
        [("canonicalUrl", 1)],
        name="unique_canonicalUrl",
        unique=True,
        partialFilterExpression={"canonicalUrl": {"$exists": True, "$type": "string"}}
    )

    # Ensure goOutUrl unique partial index WITHOUT $ne (avoid unsupported $not in partial index)
    ensure_index(
        parties_collection,
        [("goOutUrl", 1)],
        name="unique_goOutUrl",
        unique=True,
        partialFilterExpression={"goOutUrl": {"$exists": True, "$type": "string"}}
    )

    # Helper index for date sorting
    ensure_index(parties_collection, [("date", 1)], name="date_asc")
    
    # Index for carousels
    ensure_index(carousels_collection, [("title", 1)], name="title_asc")


    app.logger.info("Connected to MongoDB and ensured indexes.")
except Exception as e:
    app.logger.error(f"Error connecting to MongoDB Atlas: {e}")
    parties_collection = None
    carousels_collection = None

# --- Security ---
JWT_SECRET = os.environ.get("JWT_SECRET_KEY", "")
ADMIN_HASH = os.environ.get("ADMIN_PASSWORD_HASH", "").encode()


def protect(f):
    """Protect admin endpoints using a JWT token in the Authorization header."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1].strip()
            try:
                jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
                return f(*args, **kwargs)
            except jwt.ExpiredSignatureError:
                return jsonify({"message": "Token expired."}), 401
            except jwt.InvalidTokenError:
                return jsonify({"message": "Invalid token."}), 401
        return jsonify({"message": "Unauthorized."}), 401

    return decorated_function


@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    """Authenticate admin and issue a short-lived JWT."""
    data = request.get_json(silent=True) or {}
    password = (data.get("password") or "").encode()

    if not ADMIN_HASH:
        return jsonify({"message": "Server misconfigured."}), 500

    try:
        hashed_attempt = bcrypt.hashpw(password, ADMIN_HASH)
        if hmac.compare_digest(hashed_attempt, ADMIN_HASH):
            exp = datetime.utcnow() + timedelta(minutes=15)
            token = jwt.encode({"exp": exp}, JWT_SECRET, algorithm="HS256")
            if isinstance(token, bytes):
                token = token.decode()
            return jsonify({"token": token}), 200
    except ValueError:
        pass
    return jsonify({"message": "Invalid credentials."}), 401

# --- Classification helpers ---
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

# --- Scraper ---
def scrape_party_details(url: str):
    if not is_url_allowed(url):
        raise ValueError("URL is not allowed.")
    app.logger.info(f"[SCRAPER] start {url}")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=15)
        app.logger.info(f"[SCRAPER] status {response.status_code}")
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        script_tag = soup.find("script", {"id": "__NEXT_DATA__"})
        if not script_tag:
            raise ValueError("Could not find party data script (__NEXT_DATA__).")

        json_data = json.loads(script_tag.string)
        event_data = json_data.get("props", {}).get("pageProps", {}).get("event")
        if not event_data:
            raise ValueError("Event data not in expected format inside JSON.")

        image_path = ""
        if event_data.get("CoverImage") and event_data["CoverImage"].get("Url"):
            image_path = event_data["CoverImage"]["Url"]
        elif event_data.get("WhatsappImage") and event_data["WhatsappImage"].get("Url"):
            image_path = event_data["WhatsappImage"]["Url"]

        if image_path:
            cover_image_path = image_path.replace("_whatsappImage.jpg", "_coverImage.jpg")
            image_url = f"https://d15q6k8l9pfut7.cloudfront.net/{cover_image_path}"
        else:
            og_image_tag = soup.find("meta", {"property": "og:image"})
            og_image_url = og_image_tag["content"] if og_image_tag else ""
            if og_image_url:
                image_url = og_image_url.replace("_whatsappImage.jpg", "_coverImage.jpg")
            else:
                raise ValueError("Could not find party image URL.")

        description = event_data.get("Description", "")
        cleaned_desc = " ".join(list(filter(None, description.split("\n")))[:3]).strip()
        if len(cleaned_desc) > 250:
            cleaned_desc = cleaned_desc[:247] + "..."

        full_text = f"{event_data.get('Title', '')} {description}"
        location = event_data.get("Adress", "")

        canonical = normalize_url(url)
        go_out = normalized_or_none_for_dedupe(url)

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
            "canonicalUrl": canonical,
        }
        if go_out:
            party_details["goOutUrl"] = go_out

        if not all([party_details["name"], party_details["date"], party_details["location"]]):
            raise ValueError("Scraped data is missing critical fields.")

        return party_details

    except requests.exceptions.RequestException as e:
        app.logger.error(f"[SCRAPER] HTTP error: {e}")
        raise
    except Exception as e:
        app.logger.error(f"[SCRAPER] error: {e}")
        raise

# --- Routes ---

# Parties
@app.route("/api/admin/add-party", methods=["POST"])
@limiter.limit("10 per minute")
@protect
def add_party():
    data = request.get_json(silent=True) or {}
    url = data.get("url")
    if not url:
        return jsonify({"message": "URL is required."}), 400

    if not is_url_allowed(url):
        return jsonify({"message": "URL is not allowed."}), 400

    try:
        party_data = scrape_party_details(url)
        canonical = party_data["canonicalUrl"]

        res = parties_collection.update_one(
            {"$or": [{"canonicalUrl": canonical}, {"goOutUrl": canonical}]},
            {"$setOnInsert": party_data},
            upsert=True,
        )

        if res.matched_count == 1 and res.upserted_id is None:
            doc = parties_collection.find_one(
                {"$or": [{"canonicalUrl": canonical}, {"goOutUrl": canonical}]},
                {"_id": 1}
            )
            return jsonify({"message": "This party has already been added.", "id": str(doc["_id"])}), 409

        party_data["_id"] = str(res.upserted_id)
        return jsonify({"message": "Party added successfully!", "party": party_data}), 201

    except errors.DuplicateKeyError as e:
        app.logger.warning(f"[DB] DuplicateKeyError: {getattr(e, 'details', None)}")
        return jsonify({"message": "This party has already been added."}), 409
    except Exception as e:
        app.logger.error(f"[DB] {str(e)}")
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500

@app.route("/api/admin/delete-party/<party_id>", methods=["DELETE"])
@limiter.limit("10 per minute")
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
        
@app.route("/api/admin/update-party/<party_id>", methods=["PUT"])
@limiter.limit("10 per minute")
@protect
def update_party(party_id):
    try:
        data = request.get_json()
        obj_id = ObjectId(party_id)

        # Basic validation: ensure data is a dict
        if not isinstance(data, dict):
            return jsonify({"message": "Invalid payload format."}), 400

        # Prevent updating the _id field
        data.pop("_id", None)
        
        result = parties_collection.update_one({"_id": obj_id}, {"$set": data})

        if result.matched_count == 0:
            return jsonify({"message": "Party not found."}), 404
        
        return jsonify({"message": "Party updated successfully!"}), 200

    except Exception as e:
        app.logger.error(f"Error updating party {party_id}: {e}")
        return jsonify({"message": "An error occurred", "error": str(e)}), 500

@app.route("/api/parties", methods=["GET"])
@limiter.limit("100 per minute")
def get_parties():
    try:
        items = []
        for party in parties_collection.find().sort("date", 1):
            party["_id"] = str(party["_id"])
            items.append(party)
        return jsonify(items), 200
    except Exception as e:
        return jsonify({"message": "Error fetching parties", "error": str(e)}), 500
        
# Carousels
@app.route("/api/admin/carousels", methods=["POST"])
@limiter.limit("10 per minute")
@protect
def add_carousel():
    data = request.get_json()
    if not data or 'title' not in data:
        return jsonify({"message": "Title is required"}), 400
    
    try:
        carousel = {
            "title": data["title"],
            "partyIds": data.get("partyIds", [])
        }
        result = carousels_collection.insert_one(carousel)
        carousel["_id"] = str(result.inserted_id)
        return jsonify(carousel), 201
    except Exception as e:
        return jsonify({"message": "Error adding carousel", "error": str(e)}), 500


@app.route("/api/admin/carousels/<carousel_id>", methods=["PUT"])
@limiter.limit("10 per minute")
@protect
def update_carousel(carousel_id):
    data = request.get_json()
    if not data:
        return jsonify({"message": "Invalid data"}), 400
        
    try:
        obj_id = ObjectId(carousel_id)
        update_data = {}
        if 'title' in data:
            update_data['title'] = data['title']
        if 'partyIds' in data:
            update_data['partyIds'] = data['partyIds']

        result = carousels_collection.update_one(
            {"_id": obj_id},
            {"$set": update_data}
        )
        if result.matched_count == 0:
            return jsonify({"message": "Carousel not found"}), 404
        return jsonify({"message": "Carousel updated successfully!"}), 200
    except Exception as e:
        return jsonify({"message": "Error updating carousel", "error": str(e)}), 500


@app.route("/api/admin/carousels/<carousel_id>", methods=["DELETE"])
@limiter.limit("10 per minute")
@protect
def delete_carousel(carousel_id):
    try:
        obj_id = ObjectId(carousel_id)
        result = carousels_collection.delete_one({"_id": obj_id})
        if result.deleted_count == 0:
            return jsonify({"message": "Carousel not found."}), 404
        return jsonify({"message": "Carousel deleted successfully!"}), 200
    except Exception as e:
        return jsonify({"message": "Error deleting carousel", "error": str(e)}), 500


@app.route("/api/carousels", methods=["GET"])
@limiter.limit("100 per minute")
def get_carousels():
    try:
        items = []
        for carousel in carousels_collection.find():
            carousel["id"] = str(carousel["_id"])
            carousel.pop("_id")
            items.append(carousel)
        return jsonify(items), 200
    except Exception as e:
        return jsonify({"message": "Error fetching carousels", "error": str(e)}), 500

@app.route('/api/admin/verify-token', methods=['POST'])
@limiter.limit("10 per minute")
@protect
def verify_token():
    """Simple endpoint to verify that a provided JWT is valid."""
    return jsonify({"message": "Token is valid."}), 200

if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 3001)))