"""
Doctor recommendation routes - location-based doctor search.
"""
from flask import Blueprint, request, jsonify
from backend.routes.auth import token_required
from backend.services.doctor_service import get_doctors_nearby

doctors_bp = Blueprint('doctors', __name__, url_prefix='/api')


@doctors_bp.route('/doctors/nearby', methods=['POST'])
@token_required
def find_nearby_doctors(current_user):
    """
    Find doctors near user's location based on disease type.

    Request body:
    {
        "latitude": 40.7128,
        "longitude": -74.0060,
        "disease_type": "diabetes",  // diabetes, heart, parkinsons
        "radius": 15000,  // optional, in meters
        "limit": 5  // optional, number of doctors to return
    }
    """
    data = request.get_json(silent=True) or {}

    latitude = data.get('latitude')
    longitude = data.get('longitude')
    disease_type = data.get('disease_type')
    radius = data.get('radius', 15000)
    limit = data.get('limit', 5)
    risk_level = data.get('risk_level', 'low')

    if not latitude or not longitude:
        return jsonify({'error': 'Location required (latitude, longitude)'}), 400

    if not disease_type:
        return jsonify({'error': 'Disease type required'}), 400

    if disease_type not in ['diabetes', 'heart', 'parkinsons']:
        return jsonify({'error': 'Invalid disease type. Must be diabetes, heart, or parkinsons'}), 400

    try:
        doctors = get_doctors_nearby(
            lat=float(latitude),
            lon=float(longitude),
            disease_type=disease_type,
            radius_meters=int(radius),
            limit=int(limit),
            risk_level=risk_level
        )

        return jsonify({
            'success': True,
            'doctors': doctors,
            'count': len(doctors),
            'search_params': {
                'latitude': latitude,
                'longitude': longitude,
                'disease_type': disease_type,
                'radius_meters': radius,
                'risk_level': risk_level,
            }
        }), 200

    except Exception as e:
        return jsonify({'error': f'Failed to find doctors: {str(e)}'}), 500
