# Analytics Enhancement Documentation

## Overview
This update adds detailed time-series analytics to the Parties247 backend API, enabling admin dashboards to view historical trends for website visits, party views, and purchases.

## What Was Added

### 1. New Analytics Aggregation Function
**Location:** `app.py`, line ~634

**Function:** `build_time_series_analytics(start, end, interval, party_slug)`

**Purpose:** Aggregates analytics data into hourly or daily time buckets.

**Parameters:**
- `start` - Start datetime (timezone-aware)
- `end` - End datetime (timezone-aware)  
- `interval` - Aggregation granularity: "hour" or "day"
- `party_slug` - Optional party slug filter

**Returns:** List of time buckets with metrics:
```json
[
  {
    "timestamp": "2024-12-25",
    "visits": 150,
    "partyViews": 45,
    "purchases": 8
  }
]
```

### 2. New Admin Endpoint
**Route:** `GET /api/admin/analytics/detailed`

**Authentication:** Requires admin JWT token

**Query Parameters:**
- `range` - Preset: "24h", "7d", "30d" (default: "7d")
- `start` / `end` - Custom ISO 8601 date range
- `interval` - Bucket size: "hour" or "day" (default: "day")
- `partyId` - Optional party slug filter

**Example:**
```bash
curl -H "Authorization: Bearer <token>" \
  "http://localhost:3001/api/admin/analytics/detailed?range=7d&interval=day"
```

### 3. OpenAPI Documentation
The new endpoint is fully documented in the OpenAPI spec at `/openapi.json` and visible in the Swagger UI at `/docs`.

### 4. Comprehensive Tests
**Location:** `tests/test_analytics_detailed.py`

**Coverage:**
- Basic daily aggregation
- Hourly granularity
- Party-specific filtering
- Multiple time buckets

All tests passing ✓

### 5. Updated README
**Location:** `README.md`

Added section: "Detailed Time-Series Analytics (Admin)" with:
- Parameter descriptions
- Usage examples
- Response format
- Metrics explanation

## Metrics Explained

1. **visits** - Unique website sessions (counted by sessionId)
2. **partyViews** - Total party page views
3. **purchases** - Ticket link clicks (redirects to purchase pages)

## Implementation Details

### Data Sources
- **Visitor Analytics Collection** - Stores unique visitor sessions
- **Party Analytics Collection** - Tracks views and redirects per party

### Time Bucketing
- **Hour:** `YYYY-MM-DDTHH:00:00Z`
- **Day:** `YYYY-MM-DD`

### Party Filtering
When `partyId` is provided:
- Views and purchases are filtered to that specific party
- Visits remain global (all sessions in the time period)

## Testing the Feature

### Run All Analytics Tests
```bash
python -m pytest tests/test_analytics_detailed.py tests/test_analytics.py -v
```

### Manual Testing
1. Start the server: `python app.py`
2. Login to get admin token:
   ```bash
   curl -X POST http://localhost:3001/api/admin/login \
     -H "Content-Type: application/json" \
     -d '{"password":"your-password"}'
   ```
3. Query analytics:
   ```bash
   curl -H "Authorization: Bearer <token>" \
     "http://localhost:3001/api/admin/analytics/detailed?range=24h&interval=hour"
   ```

## Files Modified

1. **app.py**
   - Added `build_time_series_analytics()` function
   - Added `/api/admin/analytics/detailed` route
   - Updated OPENAPI_TEMPLATE with new endpoint

2. **README.md**
   - Added "Detailed Time-Series Analytics (Admin)" section

3. **tests/test_analytics_detailed.py** (new file)
   - 3 test cases covering all scenarios

## Next Steps

Consider adding:
- Export to CSV functionality
- Chart visualization endpoint
- Real-time WebSocket updates
- Custom metric definitions
