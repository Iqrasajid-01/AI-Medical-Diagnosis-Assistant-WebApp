"""
Appointment booking routes.
"""
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify
from backend.models.db_models import db, Appointment
from backend.routes.auth import token_required
from backend.services.email_service import send_booking_confirmation_email

appointments_bp = Blueprint('appointments', __name__, url_prefix='/api')


@appointments_bp.route('/appointments/book', methods=['POST'])
@token_required
def book_appointment(current_user):
    """
    Book an appointment with a doctor.

    Request body:
    {
        "prediction_id": 42,  // optional
        "doctor_name": "Dr. John Smith",
        "doctor_specialty": "Cardiologist",
        "doctor_address": "123 Medical Center, City",
        "doctor_lat": 40.7128,
        "doctor_lon": -74.0060,
        "doctor_phone": "+1-555-1234",
        "doctor_rating": 4.5,
        "appointment_date": "2026-06-25",
        "appointment_time": "10:30 AM",
        "notes": "Follow-up for heart assessment"  // optional
    }
    """
    data = request.get_json(silent=True) or {}

    required_fields = ['doctor_name', 'doctor_specialty', 'appointment_date', 'appointment_time']
    for field in required_fields:
        if not data.get(field):
            return jsonify({'error': f'{field} is required'}), 400

    try:
        appointment_date = datetime.strptime(data['appointment_date'], '%Y-%m-%d')

        if appointment_date.date() < datetime.now().date():
            return jsonify({'error': 'Cannot book appointments in the past'}), 400

    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400

    appointment = Appointment(
        user_id=current_user.id,
        prediction_id=data.get('prediction_id'),
        doctor_name=data['doctor_name'],
        doctor_specialty=data['doctor_specialty'],
        doctor_address=data.get('doctor_address'),
        doctor_lat=data.get('doctor_lat'),
        doctor_lon=data.get('doctor_lon'),
        doctor_phone=data.get('doctor_phone'),
        doctor_rating=data.get('doctor_rating'),
        appointment_date=appointment_date,
        appointment_time=data['appointment_time'],
        notes=data.get('notes'),
        status='confirmed'
    )

    db.session.add(appointment)
    db.session.commit()

    email_data = {
        'appointment_id': appointment.id,
        'doctor_name': appointment.doctor_name,
        'doctor_specialty': appointment.doctor_specialty,
        'doctor_address': appointment.doctor_address,
        'doctor_phone': appointment.doctor_phone,
        'appointment_date': appointment.appointment_date.strftime('%B %d, %Y'),
        'appointment_time': appointment.appointment_time,
        'notes': appointment.notes,
    }

    try:
        send_booking_confirmation_email(current_user.email, email_data)
        email_sent = True
    except Exception as e:
        print(f"[Appointments] Email failed: {e}")
        email_sent = False

    return jsonify({
        'success': True,
        'message': 'Appointment booked successfully',
        'appointment': appointment.to_dict(),
        'email_sent': email_sent,
    }), 201


@appointments_bp.route('/appointments', methods=['GET'])
@token_required
def get_appointments(current_user):
    """Get all appointments for the current user."""
    appointments = (
        Appointment.query
        .filter_by(user_id=current_user.id)
        .order_by(Appointment.appointment_date.desc())
        .all()
    )

    return jsonify({
        'appointments': [a.to_dict() for a in appointments],
        'count': len(appointments)
    }), 200


@appointments_bp.route('/appointments/<int:appointment_id>', methods=['GET'])
@token_required
def get_appointment(current_user, appointment_id):
    """Get a specific appointment."""
    appointment = Appointment.query.get(appointment_id)

    if not appointment:
        return jsonify({'error': 'Appointment not found'}), 404

    if appointment.user_id != current_user.id and current_user.role != 'admin':
        return jsonify({'error': 'Access denied'}), 403

    return jsonify({'appointment': appointment.to_dict()}), 200


@appointments_bp.route('/appointments/<int:appointment_id>/cancel', methods=['POST'])
@token_required
def cancel_appointment(current_user, appointment_id):
    """Cancel an appointment."""
    appointment = Appointment.query.get(appointment_id)

    if not appointment:
        return jsonify({'error': 'Appointment not found'}), 404

    if appointment.user_id != current_user.id:
        return jsonify({'error': 'Access denied'}), 403

    if appointment.status == 'cancelled':
        return jsonify({'error': 'Appointment already cancelled'}), 400

    appointment.status = 'cancelled'
    db.session.commit()

    return jsonify({
        'success': True,
        'message': 'Appointment cancelled',
        'appointment': appointment.to_dict()
    }), 200
