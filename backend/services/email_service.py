"""
Email service — supports prediction reports and appointment confirmations.
Uses SMTP settings from environment variables for production.
Falls back to console logging for development.
"""
import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


def _get_smtp_config():
    """Get SMTP configuration from environment."""
    return {
        'host': os.getenv('SMTP_HOST', ''),
        'port': int(os.getenv('SMTP_PORT', 587)),
        'user': os.getenv('SMTP_USER', ''),
        'password': os.getenv('SMTP_PASSWORD', ''),
        'from_email': os.getenv('SMTP_FROM', 'noreply@aimedicalassistant.com'),
        'use_tls': os.getenv('SMTP_USE_TLS', 'true').lower() == 'true',
    }


def _send_real_email(to_email, subject, html_body, text_body):
    """Send real email via SMTP."""
    config = _get_smtp_config()

    if not config['host'] or not config['user']:
        return False

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = config['from_email']
    msg['To'] = to_email

    msg.attach(MIMEText(text_body, 'plain'))
    msg.attach(MIMEText(html_body, 'html'))

    try:
        with smtplib.SMTP(config['host'], config['port']) as server:
            if config['use_tls']:
                server.starttls()
            server.login(config['user'], config['password'])
            server.sendmail(config['from_email'], to_email, msg.as_string())
        return True
    except Exception as e:
        print(f"[Email] SMTP error: {e}")
        return False


def send_report_email(to_email, prediction_data):
    """
    Send prediction report via email.

    Parameters
    ----------
    to_email : str
        Recipient email address.
    prediction_data : dict
        Prediction result data to include in the email.

    Returns
    -------
    dict
        Status message.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    disease = prediction_data.get('disease', 'Unknown').title()
    result = 'Positive (At Risk)' if prediction_data.get('prediction') == 1 else 'Negative (Low Risk)'
    confidence = prediction_data.get('confidence', 0) * 100
    risk_level = prediction_data.get('risk_level', 'N/A')

    subject = f'AI Medical Diagnosis Report - {disease}'
    text_body = f"""Dear Patient,

Your medical prediction report is ready.

Disease: {disease}
Result: {result}
Confidence: {confidence:.1f}%
Risk Level: {risk_level}

⚠️ This is for educational purposes only.
Please consult a healthcare professional.

— AI Medical Diagnosis Assistant
"""

    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px;">
            <h1 style="color: white; margin: 0;">🩺 AI Medical Assistant</h1>
        </div>
        <div style="padding: 20px; background: #f9fafb;">
            <h2 style="color: #1f2937;">Diagnosis Report</h2>
            <table style="width: 100%; border-collapse: collapse;">
                <tr><td style="padding: 10px; border-bottom: 1px solid #e5e7eb;"><strong>Disease:</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #e5e7eb;">{disease}</td></tr>
                <tr><td style="padding: 10px; border-bottom: 1px solid #e5e7eb;"><strong>Result:</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #e5e7eb;">{result}</td></tr>
                <tr><td style="padding: 10px; border-bottom: 1px solid #e5e7eb;"><strong>Confidence:</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #e5e7eb;">{confidence:.1f}%</td></tr>
                <tr><td style="padding: 10px; border-bottom: 1px solid #e5e7eb;"><strong>Risk Level:</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #e5e7eb;">{risk_level}</td></tr>
            </table>
            <div style="margin-top: 20px; padding: 15px; background: #fef3c7; border-radius: 8px;">
                <p style="margin: 0; color: #92400e;">⚠️ <strong>Disclaimer:</strong> This is for educational purposes only. Please consult a healthcare professional.</p>
            </div>
        </div>
    </body>
    </html>
    """

    sent = _send_real_email(to_email, subject, html_body, text_body)

    if not sent:
        print("=" * 60)
        print(f"📧 EMAIL LOG — {timestamp}")
        print("=" * 60)
        print(f"  To:      {to_email}")
        print(f"  Subject: {subject}")
        print(f"  Status:  {'Sent (SMTP)' if sent else 'Logged (mock mode)'}")
        print("=" * 60)

    return {
        'success': True,
        'message': f'Report email {"sent" if sent else "logged"} for {to_email}',
        'timestamp': timestamp,
    }


def send_booking_confirmation_email(to_email, booking_data):
    """
    Send appointment booking confirmation via email.

    Parameters
    ----------
    to_email : str
        Recipient email address.
    booking_data : dict
        Booking details including doctor info and appointment time.

    Returns
    -------
    dict
        Status message.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    appointment_id = booking_data.get('appointment_id', 'N/A')
    doctor_name = booking_data.get('doctor_name', 'Doctor')
    specialty = booking_data.get('doctor_specialty', 'Specialist')
    address = booking_data.get('doctor_address', 'Address not provided')
    phone = booking_data.get('doctor_phone', 'Phone not provided')
    date = booking_data.get('appointment_date', 'Date not set')
    time = booking_data.get('appointment_time', 'Time not set')
    notes = booking_data.get('notes', '')

    subject = f'✅ Appointment Confirmed - {doctor_name}'
    text_body = f"""Dear Patient,

Your appointment has been confirmed! 🎉

Booking ID: #{appointment_id}

Doctor: {doctor_name}
Specialty: {specialty}
Date: {date}
Time: {time}
Address: {address}
Phone: {phone}
{("Notes: " + notes) if notes else ""}

Please arrive 15 minutes early and bring your ID.

If you need to cancel, please do so at least 24 hours in advance.

— AI Medical Assistant
"""

    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 20px; border-radius: 10px;">
            <h1 style="color: white; margin: 0;">✅ Appointment Confirmed!</h1>
        </div>
        <div style="padding: 20px; background: #f9fafb;">
            <p style="color: #6b7280;">Booking ID: <strong>#{appointment_id}</strong></p>

            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h2 style="color: #1f2937; margin-top: 0;">👨‍⚕️ {doctor_name}</h2>
                <p style="color: #6b7280;">{specialty}</p>

                <table style="width: 100%; margin-top: 15px;">
                    <tr><td style="padding: 8px 0;"><strong>📅 Date:</strong></td>
                        <td>{date}</td></tr>
                    <tr><td style="padding: 8px 0;"><strong>⏰ Time:</strong></td>
                        <td>{time}</td></tr>
                    <tr><td style="padding: 8px 0;"><strong>📍 Address:</strong></td>
                        <td>{address}</td></tr>
                    <tr><td style="padding: 8px 0;"><strong>📞 Phone:</strong></td>
                        <td>{phone}</td></tr>
                    {f'<tr><td style="padding: 8px 0;"><strong>📝 Notes:</strong></td><td>{notes}</td></tr>' if notes else ''}
                </table>
            </div>

            <div style="margin-top: 20px; padding: 15px; background: #dbeafe; border-radius: 8px;">
                <p style="margin: 0; color: #1e40af;">💡 <strong>Tip:</strong> Please arrive 15 minutes early and bring your ID.</p>
            </div>

            <div style="margin-top: 20px; padding: 15px; background: #fef3c7; border-radius: 8px;">
                <p style="margin: 0; color: #92400e;">⚠️ To cancel or reschedule, please contact us at least 24 hours in advance.</p>
            </div>
        </div>
    </body>
    </html>
    """

    sent = _send_real_email(to_email, subject, html_body, text_body)

    if not sent:
        print("=" * 60)
        print(f"📧 BOOKING CONFIRMATION — {timestamp}")
        print("=" * 60)
        print(f"  To:      {to_email}")
        print(f"  Subject: {subject}")
        print(f"  Doctor:  {doctor_name} ({specialty})")
        print(f"  Date:    {date} at {time}")
        print(f"  Address: {address}")
        print(f"  Phone:   {phone}")
        print("=" * 60)

    return {
        'success': True,
        'message': f'Booking confirmation {"sent" if sent else "logged"} for {to_email}',
        'timestamp': timestamp,
    }
