import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import app


def test_picks_up_changed_image_location_description():
    party = {
        "imageUrl": "https://old-image.example/a.jpg",
        "location": "Old Club, Tel Aviv",
        "description": "Old description.",
    }
    details = {
        "imageUrl": "https://new-image.example/b.jpg",
        "location": "New Club, Tel Aviv",
        "description": "New description.",
    }
    assert app._compute_content_changes(details, party) == {
        "imageUrl": "https://new-image.example/b.jpg",
        "location": "New Club, Tel Aviv",
        "description": "New description.",
    }


def test_no_changes_when_scrape_matches_stored_party():
    party = {
        "imageUrl": "https://img.example/a.jpg",
        "location": "Club X",
        "description": "Same.",
    }
    details = dict(party)
    assert app._compute_content_changes(details, party) == {}


def test_ignores_fallback_placeholder_values():
    """A network hiccup or page-shape change makes scrape_party_details() fall
    back to placeholder values — those must never overwrite good stored data."""
    party = {
        "imageUrl": "https://img.example/real.jpg",
        "location": "Real Venue",
        "description": "Real description.",
    }
    details = {
        "imageUrl": "https://via.placeholder.com/600x400?text=No+Image+Available",
        "location": "Unknown Location",
        "description": "No description available.",
    }
    assert app._compute_content_changes(details, party) == {}


def test_partial_change_only_includes_changed_fields():
    party = {
        "imageUrl": "https://img.example/a.jpg",
        "location": "Club X",
        "description": "Same description.",
    }
    details = {
        "imageUrl": "https://img.example/a.jpg",
        "location": "Club Y (renamed)",
        "description": "Same description.",
    }
    assert app._compute_content_changes(details, party) == {"location": "Club Y (renamed)"}


def test_missing_fields_in_scrape_are_skipped():
    party = {"imageUrl": "https://img.example/a.jpg", "location": "Club X", "description": "D."}
    details = {}
    assert app._compute_content_changes(details, party) == {}
