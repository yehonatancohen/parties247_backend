from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
import app

# Setup Mocks (Duplicate of previous file content to preserve context + new test)
class AdvancedFakeVisitorCollection:
    def __init__(self):
        self.docs = []

    def insert_one(self, doc):
        self.docs.append(doc)
        return SimpleNamespace(inserted_id="id")

    def find(self, query=None, *args, **kwargs):
        if not query:
            return list(self.docs)
        filtered = []
        created_filter = query.get("createdAt", {})
        start = created_filter.get("$gte") if isinstance(created_filter, dict) else None
        end = created_filter.get("$lte") if isinstance(created_filter, dict) else None
        for doc in self.docs:
            created = doc.get("createdAt")
            if not created: continue
            if start and created < start: continue
            if end and created > end: continue
            filtered.append(doc)
        return filtered

class FakeAnalyticsCollection:
    def __init__(self):
        self.docs = []

    def insert_one(self, doc):
        self.docs.append(doc)
        return SimpleNamespace(inserted_id="id")

    def find(self, query=None, *args, **kwargs):
        if not query: return list(self.docs)
        filtered = []
        
        start = query.get("createdAt", {}).get("$gte")
        end = query.get("createdAt", {}).get("$lte")
        category = query.get("category")
        actions = query.get("action", {}).get("$in", [])
        
        for doc in self.docs:
            if category and doc.get("category") != category: continue
            if actions and doc.get("action") not in actions: continue
            created = doc.get("createdAt")
            if not created: continue
            if start and created < start: continue
            if end and created > end: continue
            filtered.append(doc)
        return filtered

class FakePartiesCollection:
    def __init__(self):
        self.docs = []

    def find(self, query=None): return list(self.docs)
    
    def find_one(self, query=None):
        if not query: return None
        for doc in self.docs:
            match = True
            for k, v in query.items():
                if k == "_id":
                    if str(doc.get("_id")) != str(v): match = False
                elif k == "slug":
                    if doc.get("slug") != v: match = False
                elif doc.get(k) != v: match = False
            if match: return doc
        return None

class FakePartyAnalyticsCollection:
    def __init__(self):
        self.docs = []
    def update_one(self, *args, **kwargs): return SimpleNamespace(modified_count=1)
    def delete_one(self, *args): pass


# Tests
def test_build_time_series_analytics_basic(monkeypatch):
    now = datetime.now(timezone.utc)
    visitor_col = AdvancedFakeVisitorCollection()
    analytics_col = FakeAnalyticsCollection()
    parties_col = FakePartiesCollection()
    
    visitor_col.docs.append({"sessionId": "s1", "createdAt": now - timedelta(hours=1)})
    
    analytics_col.docs.append({
        "category": "party", "action": "view", 
        "label": "party1", "partyId": "id1", "createdAt": now - timedelta(hours=1)
    })
    
    monkeypatch.setattr(app, "visitor_analytics_collection", visitor_col)
    monkeypatch.setattr(app, "analytics_collection", analytics_col)
    monkeypatch.setattr(app, "parties_collection", parties_col)
    monkeypatch.setattr(app, "fetch_all_documents", lambda col: col.docs)
    
    results = app.build_time_series_analytics(now - timedelta(days=7), now, "day")
    assert len(results) > 0
    assert sum(r["partyViews"] for r in results) == 1

def test_controller_logic_lifecycle(monkeypatch):
    """Test full cycle: record_party_interaction logic (VIEW) -> stored -> retrieved"""
    now = datetime.now(timezone.utc)
    
    parties_col = FakePartiesCollection()
    party_analytics_col = FakePartyAnalyticsCollection()
    analytics_col = FakeAnalyticsCollection()
    visitor_col = AdvancedFakeVisitorCollection()
    
    parties_col.docs.append({"_id": "party123", "slug": "cool-party", "name": "Cool Party"})
    
    monkeypatch.setattr(app, "parties_collection", parties_col)
    monkeypatch.setattr(app, "party_analytics_collection", party_analytics_col)
    monkeypatch.setattr(app, "analytics_collection", analytics_col)
    monkeypatch.setattr(app, "visitor_analytics_collection", visitor_col)
    
    mock_request = MagicMock()
    mock_request.get_json.return_value = {"partyId": "party123"}
    monkeypatch.setattr(app, "request", mock_request)
    
    resp = app.record_party_interaction("views")
    assert resp[1] == 202
    
    assert len(analytics_col.docs) == 1
    event = analytics_col.docs[0]
    assert event["action"] == "view"
    
    results = app.build_time_series_analytics(now - timedelta(minutes=5), now + timedelta(minutes=5), "hour")
    assert len(results) == 1
    assert results[0]["partyViews"] == 1
    assert results[0]["purchases"] == 0

def test_controller_redirect_logic(monkeypatch):
    """Test full cycle: record_party_interaction logic (REDIRECT) -> stored -> retrieved"""
    now = datetime.now(timezone.utc)
    
    parties_col = FakePartiesCollection()
    party_analytics_col = FakePartyAnalyticsCollection()
    analytics_col = FakeAnalyticsCollection()
    visitor_col = AdvancedFakeVisitorCollection()
    
    parties_col.docs.append({"_id": "party999", "slug": "mega-party", "name": "Mega Party"})
    
    monkeypatch.setattr(app, "parties_collection", parties_col)
    monkeypatch.setattr(app, "party_analytics_collection", party_analytics_col)
    monkeypatch.setattr(app, "analytics_collection", analytics_col)
    monkeypatch.setattr(app, "visitor_analytics_collection", visitor_col)
    
    # Simulate /api/analytics/party-redirect
    mock_request = MagicMock()
    mock_request.get_json.return_value = {"partyId": "party999"}
    monkeypatch.setattr(app, "request", mock_request)
    
    # 1. Trigger Redirect
    resp = app.record_party_interaction("redirects")
    assert resp[1] == 202
    
    # 2. Verify Event
    assert len(analytics_col.docs) == 1
    event = analytics_col.docs[0]
    assert event["category"] == "party"
    assert event["action"] == "redirect"
    assert event["partyId"] == "party999"
    
    # 3. Verify Retrieval
    results = app.build_time_series_analytics(now - timedelta(minutes=5), now + timedelta(minutes=5), "hour")
    
    assert len(results) == 1
    # Purchases (redirects) should be 1
    assert results[0]["purchases"] == 1
    assert results[0]["partyViews"] == 0

def test_build_time_series_analytics_invalid_party(monkeypatch):
    """
    Test that the internal builder returns zero results for an invalid party filter.
    (The controller layer handles the 404).
    """
    now = datetime.now(timezone.utc)
    
    analytics_col = FakeAnalyticsCollection()
    visitor_col = AdvancedFakeVisitorCollection()
    
    # 1. Add some real data
    analytics_col.docs.append({
        "category": "party", "action": "view", 
        "label": "party1", "partyId": "id1", "createdAt": now - timedelta(hours=1)
    })
    visitor_col.docs.append({"sessionId": "s1", "createdAt": now - timedelta(hours=1)})
    
    monkeypatch.setattr(app, "visitor_analytics_collection", visitor_col)
    monkeypatch.setattr(app, "analytics_collection", analytics_col)
    monkeypatch.setattr(app, "parties_collection", FakePartiesCollection()) # needed for find_party_for_analytics check internally if used, though mocked
    monkeypatch.setattr(app, "find_party_for_analytics", lambda slug, id: None) # "hour" not found
    
    # 2. Call with the inputs seen in user's JSON
    # interval="day", partyId="hour"
    results = app.build_time_series_analytics(
        now - timedelta(days=7), 
        now, 
        interval="day", 
        party_slug="hour"
    )
    
    # 3. Assert correct "system behavior" (which is 0 results because of the filter)
    assert len(results) > 0 # Should have buckets now
    total_views = sum(r["partyViews"] for r in results)
    assert total_views == 0
    assert sum(r["visits"] for r in results) > 0

def test_controller_invalid_party(monkeypatch):
    """Test that the controller returns 404 if the partyId is invalid."""
    monkeypatch.setattr(app, "find_party_for_analytics", lambda slug, id: None)
    
    # 1. Mock Auth dependencies
    monkeypatch.setattr(app, "JWT_SECRET", "testsecret")
    monkeypatch.setattr(app.jwt, "decode", lambda *args, **kwargs: {"sub": "admin"})

    # 2. Mock Request object
    mock_request = MagicMock()
    # args for the route: ?partyId=hour
    mock_request.args = {"partyId": "hour"}
    # headers for @protect: Authorization: Bearer token
    mock_request.headers.get.side_effect = lambda k, default=None: "Bearer mock-token" if k == "Authorization" else default
    
    monkeypatch.setattr(app, "request", mock_request)

    # 3. Call the decorated function directly
    resp, code = app.analytics_detailed()
    
    assert code == 404
    data = resp.get_json() if hasattr(resp, "get_json") else resp
    assert "not found" in data["message"]

