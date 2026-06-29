import requests
import json

api_key = 'AIzaSyB7vMB1pa3RcfVVUTEol_Cc1GoYSgabnDM'

print("Testing Google Places API (Nearby Search)...")

search_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
params = {
    'location': '40.7128,-74.0060',
    'radius': 15000,
    'keyword': 'cardiologist',
    'type': 'doctor',
    'key': api_key,
}

r = requests.get(search_url, params=params, timeout=15)
print(f"Status: {r.status_code}")
data = r.json()

with open('api_test.json', 'w') as f:
    json.dump(data, f, indent=2)

if data.get('status') == 'OK':
    places = data.get('results', [])
    print(f"Found {len(places)} doctors:")
    for p in places[:5]:
        print(f"  - {p.get('name')} | Rating: {p.get('rating')} | Vicinity: {p.get('vicinity', 'N/A')[:50]}...")

    place_id = places[0].get('place_id') if places else None
    if place_id:
        print(f"\nFetching details for first doctor...")
        details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        details_params = {
            'place_id': place_id,
            'fields': 'formatted_phone_number,website',
            'key': api_key,
        }
        dr = requests.get(details_url, params=details_params, timeout=10)
        details = dr.json()
        if details.get('status') == 'OK':
            result = details.get('result', {})
            print(f"  Phone: {result.get('formatted_phone_number', 'N/A')}")
            print(f"  Website: {result.get('website', 'N/A')}")
        else:
            print(f"  Details error: {details.get('status')}")
else:
    print("Error:", data.get('status'), data.get('error_message', ''))
