"""
Doctor recommendation service using Google Places API.
Provides real-time healthcare provider data with booking details.
"""
import os
import requests
import math
from dotenv import load_dotenv

# Load .env from backend directory
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_env_path = os.path.join(_backend_dir, '.env')
load_dotenv(_env_path, override=True)


DISEASE_KEYWORDS = {
    'diabetes': ['diabetes', 'endocrinologist', 'internal medicine', 'primary care'],
    'heart': ['cardiologist', 'cardiology', 'heart', 'cardiovascular'],
    'parkinsons': ['neurologist', 'parkinson', 'movement disorder'],
}


def get_google_places_api_key():
    """Get Google Places API key from environment."""
    # HARDCODED FOR TESTING
    key = 'AIzaSyB7vMB1pa3RcfVVUTEol_Cc1GoYSgabnDM'
    print(f"[DoctorService] Using API Key: {key[:15]}...")
    return key


def get_doctors_nearby(lat, lon, disease_type, radius_meters=15000, limit=5, risk_level='low'):
    """
    Find nearby doctors using Google Places API with full details.
    High risk = specialists only, Low/Moderate = all doctors.
    """
    api_key = get_google_places_api_key()

    if not api_key:
        print("[DoctorService] No Google API key found, using fallback")
        return generate_fallback_doctors(lat, lon, disease_type, limit)

    # High risk = use same keywords but filter strictly for specialists
    # Low/Moderate = all doctors including general physicians
    keywords = DISEASE_KEYWORDS.get(disease_type, ['doctor', 'medical'])

    for keyword in keywords:
        try:
            doctors = search_doctors(api_key, lat, lon, keyword, radius_meters, limit, risk_level)
            if doctors:
                return doctors[:limit]
        except Exception as e:
            print(f"[DoctorService] Failed for '{keyword}': {e}")
            continue

    print("[DoctorService] Google Places failed, using fallback")
    return generate_fallback_doctors(lat, lon, disease_type, limit)


def search_doctors(api_key, lat, lon, keyword, radius_meters, limit, risk_level='low'):
    """Search for doctors using Google Places Nearby Search API."""
    import sys
    print(f"[DEBUG] Searching: keyword={keyword}, lat={lat}, lon={lon}", file=sys.stderr)
    print(f"[DEBUG] Using API key: {api_key[:15]}...", file=sys.stderr)

    search_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    params = {
        'location': f'{lat},{lon}',
        'radius': radius_meters,
        'keyword': keyword,
        'type': 'doctor',
        'key': api_key,
    }

    print(f"[DEBUG] Params: {params}", file=sys.stderr)
    try:
        response = requests.get(search_url, params=params, timeout=30)
        print(f"[DEBUG] Response status: {response.status_code}", file=sys.stderr)
        data = response.json()
        print(f"[DEBUG] API status: {data.get('status')}", file=sys.stderr)
        print(f"[DEBUG] Results count: {len(data.get('results', []))}", file=sys.stderr)
    except Exception as e:
        print(f"[DEBUG] Exception: {e}", file=sys.stderr)
        return []

    if data.get('status') not in ['OK', 'ZERO_RESULTS']:
        print(f"[DoctorService] Search status: {data.get('status')}")
        return []

    places = data.get('results', [])
    if not places:
        return []

    doctors = []
    for place in places[:limit * 4]:  # Process more places for high risk filtering
        # Always fetch phone/website so UI buttons work for all risk levels
        doctor = build_doctor_data(api_key, place, lat, lon, skip_details=False)
        if doctor:
            # High risk = score doctors by specialist likelihood
            if risk_level == 'high':
                doctor['_specialist_score'] = calculate_specialist_score(doctor)
            doctors.append(doctor)

    # For high risk, sort by specialist score first, then distance
    if risk_level == 'high':
        result = sorted(doctors, key=lambda x: (-x.get('_specialist_score', 0), x['distance_m']))[:limit]
    else:
        result = sorted(doctors, key=lambda x: x['distance_m'])[:limit]

    # Remove internal score field from output
    for doc in result:
        doc.pop('_specialist_score', None)

    return result


def calculate_specialist_score(doctor):
    """Score how likely a doctor is a specialist (higher = more specialist)."""
    name_lower = doctor.get('name', '').lower()
    specialty = doctor.get('specialty', '')

    score = 0

    # High value keywords (definite specialist)
    if any(k in name_lower for k in ['endocrin', 'diabetolog', 'cardiolog', 'cardiac electrophys',
                                      'neurolog', 'movement disorder']):
        score += 100

    # Medium value keywords (likely specialist)
    if any(k in name_lower for k in ['diabetes', 'cardio', 'heart', 'neuro', 'parkinson',
                                       'institute', 'center', 'clinic']):
        score += 50

    # Lower value (still specialist adjacent)
    if any(k in name_lower for k in ['specialist', 'specialty', 'nutrition']):
        score += 25

    # If already marked as non-general specialty
    if specialty not in ['General Physician', 'Medical Center', 'Hospital', 'Health Center', '']:
        score += 30

    return score


def build_doctor_data(api_key, place, user_lat, user_lon, skip_details=False):
    """Build doctor data from Places API response."""
    import sys
    print(f"[DEBUG] Building doctor from place: {place.get('name', 'Unknown')}", file=sys.stderr)

    place_id = place.get('place_id', '')
    name = place.get('name', 'Medical Center')
    rating = place.get('rating', 0)
    review_count = place.get('user_ratings_total', 0)

    address = place.get('vicinity', place.get('formatted_address', ''))

    location = place.get('geometry', {}).get('location', {})
    lat = location.get('lat', user_lat)
    lon_val = location.get('lng', user_lon)

    distance = calculate_distance(user_lat, user_lon, lat, lon_val)

    types = place.get('types', [])
    specialty = determine_specialty(types, name)

    opening_hours = place.get('opening_hours', {})
    is_open_now = opening_hours.get('open_now') if opening_hours else None

    maps_url = f'https://www.google.com/maps/place/?q=place_id:{place_id}'

    doctor = {
        'id': place_id,
        'name': name,
        'specialty': specialty,
        'address': address,
        'phone': '',
        'rating': rating,
        'review_count': review_count,
        'distance_km': round(distance / 1000, 1),
        'distance_m': round(distance),
        'lat': lat,
        'lon': lon_val,
        'opening_hours': get_opening_text(is_open_now),
        'is_24h': is_open_now is True,
        'wheelchair_access': 'wheelchair' in types,
        'google_place_id': place_id,
        'website': '',
        'maps_url': maps_url,
        'direct_booking': False,
    }

    place_details = None if skip_details else get_place_details(api_key, place_id)
    if place_details:
        doctor['phone'] = place_details.get('formatted_phone_number', '') or place_details.get('international_phone_number', '')
        doctor['website'] = place_details.get('website', '')
        doctor['booking_url'] = determine_booking_url(
            doctor['website'],
            doctor['phone'],
            maps_url
        )
        if doctor['website']:
            doctor['direct_booking'] = True

    if not doctor.get('booking_url'):
        doctor['booking_url'] = maps_url

    return doctor


def get_place_details(api_key, place_id):
    """Get detailed info for a place including phone and website."""
    if not place_id:
        return {}

    details_url = "https://maps.googleapis.com/maps/api/place/details/json"

    params = {
        'place_id': place_id,
        'fields': 'formatted_phone_number,website,international_phone_number',
        'key': api_key,
    }

    try:
        response = requests.get(details_url, params=params, timeout=15)
        data = response.json()

        import sys
        print(f"[DETAILS] place_id={place_id}, status={data.get('status')}", file=sys.stderr)
        result = data.get('result', {})
        print(f"[DETAILS] phone={result.get('formatted_phone_number','')[:20]}, website={'yes' if result.get('website') else 'no'}", file=sys.stderr)

        if data.get('status') == 'OK':
            return result
    except Exception as e:
        print(f"[DoctorService] Details fetch failed: {e}", file=sys.stderr)

    return {}


def determine_specialty(types, name):
    """Determine specialty from place types and name."""
    name_lower = name.lower()

    # Check name FIRST for disease-specific keywords
    if 'cardio' in name_lower or 'heart' in name_lower or 'cardiac' in name_lower:
        return 'Cardiologist'
    if 'neuro' in name_lower or 'neurolo' in name_lower:
        return 'Neurologist'
    if 'ortho' in name_lower or 'orthop' in name_lower:
        return 'Orthopedic'
    if 'pediatric' in name_lower or 'child' in name_lower:
        return 'Pediatrician'
    if 'endo' in name_lower or 'diabete' in name_lower or 'thyroid' in name_lower:
        return 'Endocrinologist'
    if 'gastro' in name_lower or 'digest' in name_lower:
        return 'Gastroenterologist'
    if 'pulmo' in name_lower or 'respiratory' in name_lower or 'lung' in name_lower:
        return 'Pulmonologist'
    if 'urolog' in name_lower or 'kidney' in name_lower:
        return 'Urologist'
    if 'ophthal' in name_lower or 'eye' in name_lower:
        return 'Ophthalmologist'
    if 'psych' in name_lower or 'mental' in name_lower:
        return 'Psychiatrist'
    if 'oncolog' in name_lower or 'cancer' in name_lower:
        return 'Oncologist'
    if 'nephro' in name_lower or 'kidney' in name_lower:
        return 'Nephrologist'
    if 'rheuma' in name_lower or 'arthritis' in name_lower:
        return 'Rheumatologist'

    # Then check place types
    type_map = {
        'doctor': None,
        'hospital': 'Hospital',
        'health': 'Health Center',
        'medical': 'Medical Center',
        'clinic': 'Clinic',
        'dentist': 'Dentist',
        'pharmacy': 'Pharmacy',
    }

    for t in types:
        if t in type_map and type_map[t] is not None:
            return type_map[t]

    return 'Medical Specialist'


def determine_booking_url(website, phone, maps_url):
    """Determine the best booking URL."""
    if website:
        return website

    if phone:
        clean_phone = phone.replace(' ', '').replace('-', '').replace('(', '').replace(')', '').replace('+', '')
        if clean_phone.startswith('1') and len(clean_phone) > 10:
            clean_phone = clean_phone[1:]
        return f'https://wa.me/{clean_phone}?text=Hi, I would like to book an appointment'

    # Always return maps_url so UI has at least one booking option
    return maps_url


def get_opening_text(is_open_now):
    """Get formatted opening hours text."""
    if is_open_now is True:
        return "Open Now"
    elif is_open_now is False:
        return "Closed"
    return "Check on Google Maps"


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in meters using Haversine formula."""
    R = 6371000

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def generate_fallback_doctors(lat, lon, disease_type, count):
    """Fallback when API is unavailable."""
    import random

    specialties_map = {
        'diabetes': ['Endocrinologist', 'Diabetes Clinic', 'Internal Medicine'],
        'heart': ['Cardiologist', 'Heart Hospital', 'Cardiac Care'],
        'parkinsons': ['Neurologist', 'Movement Disorder Center', "Parkinson's Clinic"],
    }

    names_map = {
        'diabetes': ['Dr. Sarah Mitchell', 'Metro Endocrine Clinic', 'City Diabetes Center'],
        'heart': ['Dr. James Wilson', 'Heart & Vascular Institute', 'Metro Cardiac Care'],
        'parkinsons': ['Dr. Emily Chen', 'Advanced Neurology Center', "Regional Parkinson's Clinic"],
    }

    specialties = specialties_map.get(disease_type, ['Medical Center'])
    fallback_names = names_map.get(disease_type, ['Medical Center'])

    doctors = []
    for i in range(count):
        search_url = f'https://www.google.com/maps/search/{disease_type}+specialist+near+me'
        doctors.append({
            'id': f'fallback_{disease_type}_{i}',
            'name': fallback_names[i % len(fallback_names)],
            'specialty': specialties[i % len(specialties)],
            'address': 'Search on Google Maps',
            'phone': '',
            'rating': round(random.uniform(3.5, 5.0), 1),
            'review_count': random.randint(50, 300),
            'distance_km': round(random.uniform(1.5, 8.0), 1),
            'distance_m': round(random.uniform(1500, 8000)),
            'lat': lat + random.uniform(-0.01, 0.01),
            'lon': lon + random.uniform(-0.01, 0.01),
            'opening_hours': 'Check online',
            'is_24h': False,
            'wheelchair_access': True,
            'fallback': True,
            'website': '',
            'booking_url': search_url,
            'maps_url': search_url,
            'direct_booking': False,
        })

    return doctors
