"""
SQLAlchemy database models for the Medical AI application.
"""
from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model):
    """User account model."""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(10), nullable=False, default='user')  # 'admin' or 'user'
    created_at = db.Column(
        db.DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    predictions = db.relationship(
        'PredictionHistory', backref='user', lazy=True,
        cascade='all, delete-orphan'
    )

    def to_dict(self):
        """Serialize user to dictionary (excludes password)."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'created_at': self.created_at.isoformat(),
        }

    def __repr__(self):
        return f'<User {self.username}>'


class PredictionHistory(db.Model):
    """Stores each prediction made by a user."""
    __tablename__ = 'prediction_history'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer, db.ForeignKey('users.id'), nullable=False
    )
    disease_type = db.Column(db.String(50), nullable=False)  # diabetes, heart, parkinsons
    input_data = db.Column(db.JSON, nullable=False)
    prediction_result = db.Column(db.Integer, nullable=False)  # 0 or 1
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(
        db.DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self):
        """Serialize prediction to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'disease_type': self.disease_type,
            'input_data': self.input_data,
            'prediction_result': self.prediction_result,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat(),
        }

    def __repr__(self):
        return f'<Prediction {self.disease_type} user={self.user_id}>'


class Appointment(db.Model):
    """Stores appointment bookings."""
    __tablename__ = 'appointments'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    prediction_id = db.Column(db.Integer, db.ForeignKey('prediction_history.id'), nullable=True)
    doctor_name = db.Column(db.String(200), nullable=False)
    doctor_specialty = db.Column(db.String(100), nullable=False)
    doctor_address = db.Column(db.String(500), nullable=True)
    doctor_lat = db.Column(db.Float, nullable=True)
    doctor_lon = db.Column(db.Float, nullable=True)
    doctor_phone = db.Column(db.String(50), nullable=True)
    doctor_rating = db.Column(db.Float, nullable=True)
    appointment_date = db.Column(db.DateTime, nullable=False)
    appointment_time = db.Column(db.String(20), nullable=False)
    status = db.Column(db.String(20), nullable=False, default='confirmed')  # confirmed, cancelled, completed
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(
        db.DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )

    user = db.relationship('User', backref='appointments')
    prediction = db.relationship('PredictionHistory', backref='appointments')

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'prediction_id': self.prediction_id,
            'doctor_name': self.doctor_name,
            'doctor_specialty': self.doctor_specialty,
            'doctor_address': self.doctor_address,
            'doctor_lat': self.doctor_lat,
            'doctor_lon': self.doctor_lon,
            'doctor_phone': self.doctor_phone,
            'doctor_rating': self.doctor_rating,
            'appointment_date': self.appointment_date.isoformat() if self.appointment_date else None,
            'appointment_time': self.appointment_time,
            'status': self.status,
            'notes': self.notes,
            'created_at': self.created_at.isoformat(),
        }

    def __repr__(self):
        return f'<Appointment {self.id} - {self.doctor_name}>'
