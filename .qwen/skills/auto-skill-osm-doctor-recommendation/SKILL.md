---
name: google-places-doctor-recommendation
description: Real-time doctor search using Google Places API with risk-level filtering
source: auto-skill
extracted_at: '2026-06-29T05:54:15.507Z'
---

# Google Places API - Doctor Recommendation System

## Overview
Uses **Google Places API** to find real-time healthcare providers based on user location and disease type. Features risk-level filtering: **HIGH risk → specialists only**, LOW/Moderate → all doctors.

## ⚠️ API Key Setup
1. Enable **Places API** in Google Cloud Console
2. Add to `backend/.env`:
   ```
   GOOGLE_PLACES_API_KEY=your_api_key
   ```

## Critical Bug Fix: api_key Scoping
**Symptom:** API returns 20 results (confirmed in debug logs) but doctors still show fallback data.

**Root Cause:** `api_key` used inside `build_doctor_data()` but NOT passed as parameter:
```python
# WRONG - api_key undefined in this scope
def build_doctor_data(place, user_lat, user_lon):
    place_details = get_place_details(api_key, place_id)  # NameError!
```

**Fix:** Pass `api_key` through the call chain:
```python
# search_doctors calls build_doctor_data
doctor = build_doctor_data(api_key, place, lat, lon)

# build_doctor_data receives it
def build_doctor_data(api_key, place, user_lat, user_lon, skip_details=False):
    place_details = get_place_details(api_key, place_id)  # Works!
```

## Risk-Level Filtering
| Risk Level | Behavior |
|-------------|----------|
| **HIGH** | Specialists only (Endocrinology, Cardiology, Neurology centers) |
| **LOW/MODERATE** | All doctors (general + specialists), sorted by distance |

### High Risk Specialist Scoring
```python
def calculate_specialist_score(doctor):
    score = 0
    name_lower = doctor.get('name', '').lower()

    # High value keywords (definite specialist)
    if any(k in name_lower for k in ['endocrin', 'diabetolog', 'cardiolog',
                                       'cardiac electrophys', 'neurolog', 'movement disorder']):
        score += 100

    # Medium value keywords (likely specialist)
    if any(k in name_lower for k in ['diabetes', 'cardio', 'heart', 'neuro',
                                       'parkinson', 'institute', 'center', 'clinic']):
        score += 50

    # Lower value (still specialist adjacent)
    if any(k in name_lower for k in ['specialist', 'specialty', 'nutrition']):
        score += 25

    return score
```

### ⚠️ Always Fetch Details (CRITICAL)
**NEVER** skip `get_place_details()` — it fetches phone and website needed for BookingButtons in the UI.

Skipping details for high risk causes: only "Get Directions" button shows, even when phone/website exist.

```python
# ALWAYS pass skip_details=False so phone + website are populated
doctor = build_doctor_data(api_key, place, lat, lon, skip_details=False)
```

## BookingButtons Logic (Frontend)
The `BookingButtons` component shows all available options as separate buttons:

```javascript
function BookingButtons({ doctor }) {
  const buttons = [];

  // Website → Book Online
  if (doctor.website) {
    buttons.push({ icon: '🌐', text: 'Book Online', url: doctor.website });
  }

  // WhatsApp → from booking_url if wa.me link
  if (doctor.booking_url?.includes('wa.me')) {
    buttons.push({ icon: '💬', text: 'WhatsApp', url: doctor.booking_url });
  }

  // Phone → Call Now
  if (doctor.phone) {
    buttons.push({ icon: '📞', text: 'Call Now', url: `tel:${doctor.phone}` });
  }

  // Get Directions (always last)
  buttons.push({ icon: '📍', text: 'Get Directions', url: doctor.maps_url });

  return <div className="grid...">{buttons.map(...)}</div>;
}
```

**Rule:** If data exists → show its button. If not → don't show. Never duplicate buttons.

## Backend Service (`doctor_service.py`)

### Disease Keywords
```python
DISEASE_KEYWORDS = {
    'diabetes': ['diabetes', 'endocrinologist', 'internal medicine', 'primary care'],
    'heart': ['cardiologist', 'cardiology', 'heart', 'cardiovascular'],
    'parkinsons': ['neurologist', 'parkinson', 'movement disorder'],
}
```

## API Endpoint
```
POST /api/doctors/nearby
Authorization: Bearer <jwt>
Body: {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "disease_type": "diabetes",  // diabetes, heart, parkinsons
    "risk_level": "high",         // low, moderate, high
    "limit": 5
}
```

## Troubleshooting
| Symptom | Cause | Fix |
|---------|-------|-----|
| API returns results but fallback shows | `api_key` not passed to `build_doctor_data` | Add `api_key` parameter |
| High risk returns fallback doctors | Filter too strict | Use scoring instead of filtering |
| Only "Get Directions" button shows, no phone/website | `skip_details=True` skips details API call | Always use `skip_details=False` |

## Files Modified
- `backend/services/doctor_service.py` - API integration + risk filtering
- `backend/routes/doctors.py` - Pass `risk_level` to service
- `frontend/src/components/UI/DoctorRecommendation.jsx` - Pass `risk_level` in request
