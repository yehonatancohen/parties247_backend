import os
import asyncio
from functools import wraps
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient, errors
from pymongo.collection import Collection
from bson.objectid import ObjectId
from playwright.async_api import async_playwright

# --- Initial Setup ---
load_dotenv() # Load environment variables from .env file

app = Flask(__name__)
# Allow requests from your frontend (adjust origin in production if needed)
CORS(app) 

# --- Database Connection ---
try:
    client = MongoClient(os.environ.get("MONGODB_URI"))
    db = client.get_default_database() # The DB name is usually in the URI
    parties_collection: Collection = db.parties
    # Create a unique index to prevent duplicate party URLs
    parties_collection.create_index("goOutUrl", unique=True)
    print("Successfully connected to MongoDB Atlas!")
except Exception as e:
    print(f"Error connecting to MongoDB Atlas: {e}")
    exit()


# --- Security Decorator ---
# This function checks for the admin secret key in the request headers
def protect(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        provided_key = request.headers.get('x-admin-secret-key')
        if provided_key and provided_key == os.environ.get("ADMIN_SECRET_KEY"):
            return f(*args, **kwargs)
        else:
            return jsonify({"message": "Forbidden: Invalid or missing admin key."}), 403
    return decorated_function


# --- Web Scraping Logic (Async) ---
async def scrape_party_data(url: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox'])
        page = await browser.new_page()
        await page.goto(url, wait_until='networkidle')

        # --- IMPORTANT: You must inspect go-out.co.il to find the correct selectors ---
        party_data = await page.evaluate('''() => {
            // These are EXAMPLE selectors. You need to replace them with the real ones.
            const title = document.querySelector('h1.event-title')?.innerText;
            const imageUrl = document.querySelector('.event-banner img')?.src;
            const date = document.querySelector('.event-date-time-class')?.innerText;
            const location = document.querySelector('.event-location-class')?.innerText;
            return { title, imageUrl, date, location };
        }''')
        
        await browser.close()
        return party_data


# --- API Routes ---

@app.route('/api/admin/add-party', methods=['POST'])
@protect
def add_party():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({"message": "URL is required."}), 400

    try:
        # Run the async scraping function within our sync Flask route
        party_data = asyncio.run(scrape_party_data(url))

        if not party_data or not party_data.get('title'):
             return jsonify({"message": "Could not scrape party data. Selectors might be wrong or page structure changed."}), 404
        
        # Prepare data for database insertion
        party_to_save = {
            **party_data,
            "goOutUrl": url
        }

        result = parties_collection.insert_one(party_to_save)
        # Convert ObjectId to string for JSON response
        party_to_save['_id'] = str(result.inserted_id)
        
        return jsonify({"message": "Party added successfully!", "party": party_to_save}), 201

    except errors.DuplicateKeyError:
        return jsonify({"message": "This party has already been added."}), 409
    except Exception as e:
        return jsonify({"message": "An error occurred during scraping", "error": str(e)}), 500


@app.route('/api/admin/delete-party/<party_id>', methods=['DELETE'])
@protect
def delete_party(party_id):
    try:
        # Convert string ID from URL to MongoDB ObjectId
        obj_id = ObjectId(party_id)
        result = parties_collection.delete_one({"_id": obj_id})

        if result.deleted_count == 0:
            return jsonify({"message": "Party not found."}), 404
        
        return jsonify({"message": "Party deleted successfully!"}), 200

    except Exception as e:
        return jsonify({"message": "Error deleting party", "error": str(e)}), 500

# Public route to get all parties (for your frontend to display)
@app.route('/api/parties', methods=['GET'])
def get_parties():
    try:
        all_parties = []
        for party in parties_collection.find().sort("createdAt", -1):
            party["_id"] = str(party["_id"]) # Convert ObjectId for JSON compatibility
            all_parties.append(party)
        return jsonify(all_parties), 200
    except Exception as e:
        return jsonify({"message": "Error fetching parties", "error": str(e)}), 500


# To run this app locally: `flask run`
# For deployment, a Gunicorn server is recommended.
if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get("PORT", 3001)))
