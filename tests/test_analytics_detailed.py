from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import pytest
import app


class AdvancedFakeVisitorCollection:
    def __init__(self):
        self.docs = []

    def insert_one(self, doc):
        self.docs.append(doc)
        return SimpleNamespace(inserted_id="id")

    def find(self, query=None, *args, **kwargs):
        if not query:
            return list(self.docs)
        
        # Simple query filter simulation for testing
        filtered = []
        created_filter = query.get("createdAt", {})
        start = created_filter.get("$gte") if isinstance(created_filter, dict) else None
        end = created_filter.get("$lte") if isinstance(created_filter, dict) else None
        
        for doc in self.docs:
            created = doc.get("createdAt")
            if not created:
                continue
            if start and created < start:
                continue
            if end and created > end:
                continue
                
            filtered.append(doc)
        return filtered


class FakeAnalyticsCollection:
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
        
        category = query.get("category")
        actions = query.get("action", {}).get("$in", [])
        
        for doc in self.docs:
            if category and doc.get("category") != category:
                continue
            if actions and doc.get("action") not in actions:
                continue
                
            created = doc.get("createdAt")
            if not created:
                continue
            if start and created < start:
                continue
            if end and created > end:
                continue
                
            filtered.append(doc)
        return filtered


class FakePartiesCollection:
    def __init__(self):
        self.docs = []

    def find(self, query=None):
        return list(self.docs)


def test_build_time_series_analytics_basic(monkeypatch):
    now = datetime.now(timezone.utc)
    visitor_col = AdvancedFakeVisitorCollection()
    analytics_col = FakeAnalyticsCollection()
    parties_col = FakePartiesCollection()
    
    # Add visitor events
    visitor_col.docs.append({
        "sessionId": "s1",
        "createdAt": now - timedelta(hours=1)
    })
    
    visitor_col.docs.append({
        "sessionId": "s2",
        "createdAt": now - timedelta(hours=2)
    })
    
    # Add party events (view and redirect)
    analytics_col.docs.extend([
        {
            "category": "party", "action": "view", 
            "label": "party1", "partyId": "id1",
            "createdAt": now - timedelta(hours=1)
        },
        {
            "category": "party", "action": "view", 
            "label": "party1", "partyId": "id1",
            "createdAt": now - timedelta(hours=2) # 2nd view
        },
        {
            "category": "party", "action": "redirect", 
            "label": "party1", "partyId": "id1",
            "createdAt": now - timedelta(hours=1)
        }
    ])

    monkeypatch.setattr(app, "visitor_analytics_collection", visitor_col)
    monkeypatch.setattr(app, "analytics_collection", analytics_col)
    monkeypatch.setattr(app, "parties_collection", parties_col)
    monkeypatch.setattr(app, "fetch_all_documents", lambda col: col.docs)
    
    # Test 7 days, daily interval
    start = now - timedelta(days=7)
    end = now
    
    results = app.build_time_series_analytics(start, end, "day")
    
    # Should have results
    assert len(results) > 0
    
    # Total unique visits should be 2
    total_visits = sum(r["visits"] for r in results)
    assert total_visits == 2
    
    # Party views should be 2
    total_party_views = sum(r["partyViews"] for r in results)
    assert total_party_views == 2
    
    # Purchases (redirects) should be 1
    total_purchases = sum(r["purchases"] for r in results)
    assert total_purchases == 1


def test_build_time_series_analytics_hourly(monkeypatch):
    now = datetime.now(timezone.utc)
    visitor_col = AdvancedFakeVisitorCollection()
    analytics_col = FakeAnalyticsCollection()
    parties_col = FakePartiesCollection()
    
    # Add views separate hours
    analytics_col.docs.extend([
        {
            "category": "party", "action": "view",
            "createdAt": now - timedelta(minutes=30)
        },
        {
            "category": "party", "action": "redirect",
            "createdAt": now - timedelta(hours=2, minutes=30)
        }
    ])
    
    monkeypatch.setattr(app, "visitor_analytics_collection", visitor_col)
    monkeypatch.setattr(app, "analytics_collection", analytics_col)
    monkeypatch.setattr(app, "parties_collection", parties_col)
    
    start = now - timedelta(hours=4)
    end = now
    
    results = app.build_time_series_analytics(start, end, "hour")
    
    # Should have at least 2 buckets
    assert len(results) >= 2
    
    total_views = sum(r["partyViews"] for r in results)
    assert total_views == 1
    
    total_purchases = sum(r["purchases"] for r in results)
    assert total_purchases == 1


def test_filtered_by_party(monkeypatch):
    now = datetime.now(timezone.utc)
    visitor_col = AdvancedFakeVisitorCollection()
    analytics_col = FakeAnalyticsCollection()
    parties_col = FakePartiesCollection()
    
    # Add analytics for two parties
    analytics_col.docs.extend([
        {
            "category": "party", "action": "view",
            "label": "party-1", "partyId": "id1",
            "createdAt": now - timedelta(hours=1)
        },
        {
            "category": "party", "action": "redirect",
            "label": "party-1", "partyId": "id1",
            "createdAt": now - timedelta(hours=1)
        },
        {
            "category": "party", "action": "view",
            "label": "party-2", "partyId": "id2",
            "createdAt": now - timedelta(hours=1)
        }
    ])
    
    # Add party documents for resolution
    parties_col.docs.append({"_id": "id1", "slug": "party-1"})
    parties_col.docs.append({"_id": "id2", "slug": "party-2"})
    
    def fake_find_party(party_id=None, party_slug=None):
        for doc in parties_col.docs:
            if party_id and str(doc.get("_id")) == str(party_id):
                return doc
            if party_slug and doc.get("slug") == party_slug:
                return doc
        return None
    
    monkeypatch.setattr(app, "visitor_analytics_collection", visitor_col)
    monkeypatch.setattr(app, "analytics_collection", analytics_col)
    monkeypatch.setattr(app, "parties_collection", parties_col)
    monkeypatch.setattr(app, "find_party_for_analytics", fake_find_party)
    
    start = now - timedelta(days=1)
    end = now
    
    # Filter by party-1
    results = app.build_time_series_analytics(start, end, "day", party_slug="party-1")
    
    # Party views should only count party-1 (1 view)
    total_party_views = sum(r["partyViews"] for r in results)
    assert total_party_views == 1
    
    # Purchases should only count party-1 (1 redirect)
    total_purchases = sum(r["purchases"] for r in results)
    assert total_purchases == 1

