"""
PDF report generation service using ReportLab.

Generates professional medical prediction reports with patient info,
prediction results, confidence levels, and disclaimers.
"""
import os
import tempfile
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


# Theme colors
CYAN = HexColor('#0891b2')
DARK_BG = HexColor('#1e293b')
TEXT_COLOR = HexColor('#334155')
LIGHT_GRAY = HexColor('#f1f5f9')
WHITE = HexColor('#ffffff')
RED = HexColor('#ef4444')
GREEN = HexColor('#10b981')

DISEASE_NAMES = {
    'diabetes': 'Type 2 Diabetes',
    'heart': 'Heart Disease',
    'parkinsons': "Parkinson's Disease",
}


def generate_prediction_pdf(prediction_record, user):
    """
    Generate a PDF report for a prediction and return the file path.

    Parameters
    ----------
    prediction_record : PredictionHistory
        The database record for the prediction.
    user : User
        The user who made the prediction.

    Returns
    -------
    str
        Path to the generated PDF file.
    """
    # Create temp file
    fd, pdf_path = tempfile.mkstemp(suffix='.pdf')
    os.close(fd)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=22,
        textColor=CYAN,
        spaceAfter=6 * mm,
        alignment=TA_CENTER,
    )

    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=TEXT_COLOR,
        spaceAfter=4 * mm,
        alignment=TA_CENTER,
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=CYAN,
        spaceBefore=6 * mm,
        spaceAfter=3 * mm,
    )

    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        textColor=TEXT_COLOR,
        spaceAfter=2 * mm,
    )

    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=HexColor('#94a3b8'),
        spaceAfter=2 * mm,
        alignment=TA_CENTER,
    )

    # Build content
    elements = []

    # Title
    elements.append(Paragraph("AI Medical Diagnosis Report", title_style))
    elements.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        subtitle_style
    ))
    elements.append(HRFlowable(width="100%", thickness=2, color=CYAN))
    elements.append(Spacer(1, 6 * mm))

    # Patient Info
    elements.append(Paragraph("Patient Information", heading_style))
    patient_data = [
        ['Username:', user.username],
        ['Email:', user.email],
        ['Report ID:', f'#{prediction_record.id}'],
        ['Date:', prediction_record.created_at.strftime('%Y-%m-%d %H:%M UTC')],
    ]
    patient_table = Table(patient_data, colWidths=[4 * cm, 12 * cm])
    patient_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), CYAN),
        ('TEXTCOLOR', (1, 0), (1, -1), TEXT_COLOR),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 6 * mm))

    # Prediction Result
    elements.append(Paragraph("Prediction Result", heading_style))

    disease_name = DISEASE_NAMES.get(prediction_record.disease_type, prediction_record.disease_type)
    is_positive = prediction_record.prediction_result == 1
    result_text = "POSITIVE (At Risk)" if is_positive else "NEGATIVE (Low Risk)"
    result_color = RED if is_positive else GREEN

    confidence_pct = f"{prediction_record.confidence * 100:.1f}%"

    # Risk level
    conf = prediction_record.confidence
    if conf >= 0.80:
        risk = 'High'
    elif conf >= 0.40:
        risk = 'Moderate'
    else:
        risk = 'Low'

    result_data = [
        ['Disease:', disease_name],
        ['Prediction:', result_text],
        ['Confidence:', confidence_pct],
        ['Risk Level:', risk],
    ]
    result_table = Table(result_data, colWidths=[4 * cm, 12 * cm])
    result_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('TEXTCOLOR', (0, 0), (0, -1), CYAN),
        ('TEXTCOLOR', (1, 1), (1, 1), result_color),
        ('FONTNAME', (1, 1), (1, 1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_GRAY),
        ('BOX', (0, 0), (-1, -1), 1, CYAN),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, HexColor('#e2e8f0')),
    ]))
    elements.append(result_table)
    elements.append(Spacer(1, 6 * mm))

    # Input Data Summary
    elements.append(Paragraph("Input Data Summary", heading_style))
    input_data = prediction_record.input_data or {}
    if input_data:
        input_rows = [['Parameter', 'Value']]
        for key, val in input_data.items():
            display_val = f"{val:.4f}" if isinstance(val, float) else str(val)
            input_rows.append([str(key), display_val])

        input_table = Table(input_rows, colWidths=[8 * cm, 8 * cm])
        input_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
            ('BACKGROUND', (0, 0), (-1, 0), CYAN),
            ('BACKGROUND', (0, 1), (-1, -1), LIGHT_GRAY),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOX', (0, 0), (-1, -1), 1, CYAN),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, HexColor('#e2e8f0')),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ]))
        elements.append(input_table)
    else:
        elements.append(Paragraph("No input data recorded.", normal_style))

    elements.append(Spacer(1, 10 * mm))

    # Disclaimer
    elements.append(HRFlowable(width="100%", thickness=1, color=HexColor('#cbd5e1')))
    elements.append(Spacer(1, 4 * mm))
    elements.append(Paragraph(
        "⚠️ DISCLAIMER: This prediction is generated by an AI model for educational "
        "and research purposes only. It should NOT be used as a substitute for professional "
        "medical diagnosis. Always consult a qualified healthcare provider for medical advice.",
        disclaimer_style,
    ))
    elements.append(Paragraph(
        "AI Medical Diagnosis Assistant — ANN Lab Course Project",
        disclaimer_style,
    ))

    doc.build(elements)
    return pdf_path
