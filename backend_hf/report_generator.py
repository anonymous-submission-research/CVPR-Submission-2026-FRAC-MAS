import os
import base64
import logging
from io import BytesIO
from datetime import datetime
from typing import Dict, Any, Optional

from PIL import Image as PILImage
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, PageBreak, KeepTogether, Flowable
)
from reportlab.platypus.flowables import BalancedColumns
from reportlab.pdfgen import canvas as rl_canvas

logger = logging.getLogger(__name__)


# ─── Design Tokens ────────────────────────────────────────────────────────────
C_NAVY    = colors.HexColor('#0f172a')   # header / footer background
C_BLUE    = colors.HexColor('#1d4ed8')   # accent stripe / headings
C_BLUE_LT = colors.HexColor('#dbeafe')  # light accent background
C_SLATE   = colors.HexColor('#1e293b')  # primary text
C_MUTED   = colors.HexColor('#64748b')  # secondary text
C_BORDER  = colors.HexColor('#cbd5e1')  # card borders
C_CARD    = colors.HexColor('#f8fafc')  # card background
C_RED     = colors.HexColor('#dc2626')  # fracture / danger
C_RED_LT  = colors.HexColor('#fef2f2')  # fracture banner bg
C_GREEN   = colors.HexColor('#16a34a')  # healthy
C_GREEN_LT = colors.HexColor('#f0fdf4') # healthy banner bg
C_WHITE   = colors.white
C_RULE    = colors.HexColor('#e2e8f0')  # horizontal rule


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _b64_to_pil(b64: str) -> Optional[PILImage.Image]:
    try:
        return PILImage.open(BytesIO(base64.b64decode(b64))).convert('RGB')
    except Exception:
        return None


def _pil_to_rl_image(pil_img: PILImage.Image, width: float, height: float) -> Image:
    """Convert a PIL Image to a ReportLab Image flowable via BytesIO."""
    buf = BytesIO()
    pil_img.save(buf, format='JPEG', quality=92)
    buf.seek(0)
    return Image(buf, width=width, height=height)


def _bytes_to_rl_image(img_bytes: bytes, width: float, height: float) -> Optional[Image]:
    try:
        pil = PILImage.open(BytesIO(img_bytes)).convert('RGB')
        return _pil_to_rl_image(pil, width, height)
    except Exception:
        return None


# ─── Styles ───────────────────────────────────────────────────────────────────

def _build_styles() -> dict:
    return {
        'report_title': ParagraphStyle(
            'report_title',
            fontName='Helvetica-Bold',
            fontSize=11,
            textColor=C_WHITE,
            leading=14,
        ),
        'report_subtitle': ParagraphStyle(
            'report_subtitle',
            fontName='Helvetica',
            fontSize=8,
            textColor=colors.HexColor('#93c5fd'),
            leading=11,
        ),
        'header_meta': ParagraphStyle(
            'header_meta',
            fontName='Helvetica',
            fontSize=8,
            textColor=colors.HexColor('#cbd5e1'),
            leading=11,
            alignment=TA_RIGHT,
        ),
        'confidential': ParagraphStyle(
            'confidential',
            fontName='Helvetica-Bold',
            fontSize=7.5,
            textColor=colors.HexColor('#f87171'),
            leading=10,
            alignment=TA_RIGHT,
        ),
        'banner_positive': ParagraphStyle(
            'banner_positive',
            fontName='Helvetica-Bold',
            fontSize=13,
            textColor=C_RED,
            leading=16,
        ),
        'banner_negative': ParagraphStyle(
            'banner_negative',
            fontName='Helvetica-Bold',
            fontSize=13,
            textColor=C_GREEN,
            leading=16,
        ),
        'banner_conf': ParagraphStyle(
            'banner_conf',
            fontName='Helvetica-Bold',
            fontSize=12,
            leading=16,
            alignment=TA_RIGHT,
        ),
        'section_heading': ParagraphStyle(
            'section_heading',
            fontName='Helvetica-Bold',
            fontSize=10,
            textColor=C_SLATE,
            leading=14,
            spaceBefore=4,
        ),
        'section_heading_blue': ParagraphStyle(
            'section_heading_blue',
            fontName='Helvetica-Bold',
            fontSize=10,
            textColor=C_BLUE,
            leading=14,
            spaceBefore=4,
        ),
        'label': ParagraphStyle(
            'label',
            fontName='Helvetica-Bold',
            fontSize=8,
            textColor=C_MUTED,
            leading=11,
        ),
        'value': ParagraphStyle(
            'value',
            fontName='Helvetica',
            fontSize=8.5,
            textColor=C_SLATE,
            leading=12,
        ),
        'body': ParagraphStyle(
            'body',
            fontName='Helvetica',
            fontSize=8.5,
            textColor=C_SLATE,
            leading=13,
            spaceBefore=2,
        ),
        'body_small': ParagraphStyle(
            'body_small',
            fontName='Helvetica',
            fontSize=7.5,
            textColor=C_SLATE,
            leading=11,
        ),
        'caption': ParagraphStyle(
            'caption',
            fontName='Helvetica-Oblique',
            fontSize=7,
            textColor=C_MUTED,
            leading=10,
            alignment=TA_CENTER,
        ),
        'image_label': ParagraphStyle(
            'image_label',
            fontName='Helvetica-Bold',
            fontSize=8,
            textColor=C_SLATE,
            leading=11,
            alignment=TA_CENTER,
        ),
        'footer_left': ParagraphStyle(
            'footer_left',
            fontName='Helvetica',
            fontSize=6.5,
            textColor=colors.HexColor('#94a3b8'),
            leading=9,
        ),
        'footer_center': ParagraphStyle(
            'footer_center',
            fontName='Helvetica',
            fontSize=6,
            textColor=colors.HexColor('#475569'),
            leading=9,
            alignment=TA_CENTER,
        ),
        'footer_right': ParagraphStyle(
            'footer_right',
            fontName='Helvetica-Oblique',
            fontSize=6.5,
            textColor=colors.HexColor('#94a3b8'),
            leading=9,
            alignment=TA_RIGHT,
        ),
        'page2_title': ParagraphStyle(
            'page2_title',
            fontName='Helvetica-Bold',
            fontSize=14,
            textColor=C_SLATE,
            leading=18,
            spaceBefore=6,
        ),
        'page2_subtitle': ParagraphStyle(
            'page2_subtitle',
            fontName='Helvetica-Oblique',
            fontSize=8,
            textColor=C_MUTED,
            leading=11,
            spaceAfter=8,
        ),
        'page2_body': ParagraphStyle(
            'page2_body',
            fontName='Helvetica',
            fontSize=8.5,
            textColor=C_SLATE,
            leading=13,
            spaceBefore=3,
            spaceAfter=2,
        ),
        'page2_subheading': ParagraphStyle(
            'page2_subheading',
            fontName='Helvetica-Bold',
            fontSize=9,
            textColor=C_BLUE,
            leading=13,
            spaceBefore=8,
            spaceAfter=2,
        ),
    }


# ─── Page Template Callbacks ───────────────────────────────────────────────────

class _PageTemplate:
    """Draws the fixed header and footer on every page via canvas callbacks."""

    def __init__(self, styles: dict, date_str: str, time_str: str,
                 inference_id: str, page_label: str = ''):
        self.styles = styles
        self.date_str = date_str
        self.time_str = time_str
        self.inference_id = inference_id
        self.page_label = page_label  # e.g. 'Page 1 of 2'

    def _draw_header(self, c: rl_canvas.Canvas, doc):
        W, H = letter
        HEADER_H = 0.75 * inch

        # Navy background
        c.setFillColor(C_NAVY)
        c.rect(0, H - HEADER_H, W, HEADER_H, fill=1, stroke=0)

        # Accent stripe below header
        c.setFillColor(C_BLUE)
        c.rect(0, H - HEADER_H - 2, W, 3, fill=1, stroke=0)

        # Left: icon + title
        c.setFillColor(C_WHITE)
        c.setFont('Helvetica-Bold', 14)
        c.drawString(0.5 * inch, H - 0.32 * inch, '◈')

        c.setFont('Helvetica-Bold', 11)
        c.drawString(0.72 * inch, H - 0.32 * inch, 'Automated Radiographic Analysis Report')

        c.setFillColor(colors.HexColor('#93c5fd'))
        c.setFont('Helvetica', 7.5)
        c.drawString(0.72 * inch, H - 0.52 * inch, 'AI-Assisted Diagnostic Imaging Report  ·  Confidential Medical Document')

        # Right: metadata
        right_x = W - 0.5 * inch
        c.setFillColor(colors.HexColor('#f87171'))
        c.setFont('Helvetica-Bold', 7.5)
        c.drawRightString(right_x, H - 0.28 * inch, 'CONFIDENTIAL')

        c.setFillColor(colors.HexColor('#cbd5e1'))
        c.setFont('Helvetica', 7.5)
        c.drawRightString(right_x, H - 0.44 * inch, f'Date: {self.date_str}')
        if self.time_str:
            c.drawRightString(right_x, H - 0.58 * inch, f'Time: {self.time_str}')

        if self.page_label:
            c.setFillColor(colors.HexColor('#94a3b8'))
            c.setFont('Helvetica', 7)
            c.drawRightString(right_x, H - 0.70 * inch, self.page_label)

    def _draw_footer(self, c: rl_canvas.Canvas, doc):
        W, _ = letter
        FOOTER_H = 0.35 * inch

        c.setFillColor(C_NAVY)
        c.rect(0, 0, W, FOOTER_H, fill=1, stroke=0)

        # Accent stripe above footer
        c.setFillColor(C_BLUE)
        c.rect(0, FOOTER_H, W, 2, fill=1, stroke=0)

        c.setFillColor(colors.HexColor('#94a3b8'))
        c.setFont('Helvetica', 6.5)
        c.drawString(0.5 * inch, 0.13 * inch, '◈  Research Team')

        c.setFillColor(colors.HexColor('#64748b'))
        c.setFont('Helvetica-Oblique', 6.5)
        c.drawRightString(W - 0.5 * inch, 0.13 * inch,
                          'AI-assisted analysis — not a substitute for professional medical advice')

        c.setFillColor(colors.HexColor('#475569'))
        c.setFont('Helvetica', 6)
        c.drawCentredString(W / 2, 0.05 * inch, f'Report ID: {self.inference_id}')

    def __call__(self, c: rl_canvas.Canvas, doc):
        c.saveState()
        self._draw_header(c, doc)
        self._draw_footer(c, doc)
        c.restoreState()


# ─── Card Table Helper ─────────────────────────────────────────────────────────

def _card_table(content_rows, col_widths, bg=C_CARD, border=C_BORDER,
                padding=8, roundness=4) -> Table:
    """Wrap content in a styled table that looks like a card."""
    tbl = Table(content_rows, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg),
        ('ROUNDEDCORNERS', [roundness]),
        ('BOX', (0, 0), (-1, -1), 0.5, border),
        ('TOPPADDING', (0, 0), (-1, -1), padding),
        ('BOTTOMPADDING', (0, 0), (-1, -1), padding),
        ('LEFTPADDING', (0, 0), (-1, -1), padding),
        ('RIGHTPADDING', (0, 0), (-1, -1), padding),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    return tbl


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def _make_pdf_report(payload: Dict[str, Any], original_image_bytes: bytes) -> BytesIO:
    """Create a professional multi-page PDF report from the diagnosis payload."""

    buf = BytesIO()
    styles = _build_styles()

    # ── Extract payload ──────────────────────────────────────────────────
    pred_data   = payload.get('prediction', {})
    ensemble    = payload.get('ensemble', {})
    explanation = payload.get('explanation', {})
    edu         = payload.get('educational', {}) or {}
    kb          = payload.get('knowledge_base', {}) or {}
    conformal   = payload.get('conformal', {}) or {}
    audit       = payload.get('audit', {}) or {}

    top_class   = pred_data.get('top_class', pred_data.get('class', 'Unknown'))
    top_conf    = pred_data.get('confidence_score', pred_data.get('confidence', 0.0))
    is_fracture = top_class.lower() != 'healthy'

    inference_id = audit.get('inference_id', 'N/A')
    timestamp    = audit.get('timestamp', datetime.utcnow().isoformat())
    try:
        dt_obj   = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        date_str = dt_obj.strftime('%B %d, %Y')
        time_str = dt_obj.strftime('%H:%M UTC')
    except Exception:
        date_str = timestamp
        time_str = ''

    anonymize = payload.get('anonymize', True)
    if anonymize:
        inference_id_display = 'REDACTED'
        time_str = ''
    else:
        inference_id_display = inference_id

    W, H = letter
    MARGIN   = 0.5 * inch
    CONTENT_W = W - 2 * MARGIN

    # ── Document setup ───────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=0.95 * inch,   # leaves room for fixed header
        bottomMargin=0.55 * inch, # leaves room for fixed footer
    )

    # ── Check if page 2 is needed ────────────────────────────────────────
    gemini_text = kb.get('gemini_explanation')
    if not gemini_text:
        gemini_text = (explanation.get('text') or explanation.get('explanation')
                       or edu.get('detailed_explanation'))
    if not gemini_text:
        kb_parts = []
        for key in ('Type_Definition', 'Clinical_Notes'):
            if kb.get(key):
                kb_parts.append(kb[key])
        if kb_parts:
            gemini_text = '\n\n'.join(kb_parts)

    total_pages = 2 if gemini_text else 1

    # ── Page callbacks ───────────────────────────────────────────────────
    page1_cb = _PageTemplate(styles, date_str, time_str, inference_id_display,
                             page_label=f'Page 1 of {total_pages}')
    page2_cb = _PageTemplate(styles, date_str, time_str, inference_id_display,
                             page_label=f'Page 2 of {total_pages}')

    story = []

    # ════════════════════════════════════════════════════════════════════
    #  PAGE 1
    # ════════════════════════════════════════════════════════════════════

    # ── Diagnosis Banner ─────────────────────────────────────────────────
    status_text = 'FRACTURE DETECTED' if is_fracture else 'NO FRACTURE DETECTED'
    banner_bg  = C_RED_LT if is_fracture else C_GREEN_LT
    banner_clr = C_RED if is_fracture else C_GREEN
    banner_bdr = colors.HexColor('#fca5a5') if is_fracture else colors.HexColor('#86efac')

    conf_style = ParagraphStyle('conf_dyn', parent=styles['banner_conf'],
                                textColor=banner_clr)
    banner_tbl = Table(
        [[
            Paragraph(status_text, ParagraphStyle('st', parent=styles['banner_positive' if is_fracture else 'banner_negative'])),
            Paragraph(f'Confidence: {top_conf * 100:.1f}%', conf_style),
        ]],
        colWidths=[CONTENT_W * 0.6, CONTENT_W * 0.4],
    )
    banner_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), banner_bg),
        ('ROUNDEDCORNERS', [6]),
        ('BOX', (0, 0), (-1, -1), 1, banner_bdr),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 14),
        ('RIGHTPADDING', (0, 0), (-1, -1), 14),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(banner_tbl)
    story.append(Spacer(1, 10))

    # ── Images: Original Radiograph + Grad-CAM ───────────────────────────
    IMG_W = (CONTENT_W - 10) / 2
    IMG_H = 1.9 * inch

    orig_rl = _bytes_to_rl_image(original_image_bytes, IMG_W, IMG_H)
    orig_cell = [
        Paragraph('Original Radiograph', styles['image_label']),
        Spacer(1, 4),
        orig_rl if orig_rl else Paragraph('[Image Unavailable]', styles['caption']),
    ]

    cam_b64 = explanation.get('heatmap_b64') or explanation.get('primary_heatmap_b64')
    cam_pil = _b64_to_pil(cam_b64) if cam_b64 else None
    if cam_pil:
        cam_rl = _pil_to_rl_image(cam_pil, IMG_W, IMG_H)
        cam_cell = [
            Paragraph('AI Attention Map (Grad-CAM)', styles['image_label']),
            Spacer(1, 4),
            cam_rl,
        ]
    else:
        cam_cell = [
            Paragraph('AI Attention Map (Grad-CAM)', styles['image_label']),
            Spacer(1, 4),
            Paragraph('[Heatmap Unavailable]', styles['caption']),
        ]

    img_tbl = Table([[orig_cell, cam_cell]], colWidths=[IMG_W, IMG_W])
    img_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), C_CARD),
        ('BOX', (0, 0), (0, 0), 0.5, C_BORDER),
        ('BOX', (1, 0), (1, 0), 0.5, C_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('INNERGRID', (0, 0), (-1, -1), 0, C_WHITE),
        ('ROUNDEDCORNERS', [4]),
    ]))
    story.append(img_tbl)
    story.append(Spacer(1, 10))

    # ── Clinical Findings ─────────────────────────────────────────────────
    diag_name      = kb.get('Diagnosis', top_class)
    icd_code       = kb.get('ICD_Code', 'N/A')
    severity       = kb.get('Severity_Rating', 'N/A')
    definition_raw = (kb.get('Type_Definition') or kb.get('Definition')
                      or 'No definition available.')

    findings_header = Table(
        [[
            Paragraph(f'<b>Classification:</b> {diag_name}', styles['value']),
            Paragraph(f'<b>ICD-10:</b> {icd_code}', styles['value']),
            Paragraph(f'<b>Severity:</b> {severity}', styles['value']),
        ]],
        colWidths=[CONTENT_W * 0.4, CONTENT_W * 0.3, CONTENT_W * 0.3],
    )
    findings_header.setStyle(TableStyle([
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    findings_content = [
        [Paragraph('Clinical Findings', styles['section_heading'])],
        [findings_header],
        [HRFlowable(width='100%', thickness=0.5, color=C_RULE, spaceAfter=4)],
        [Paragraph('<b>Definition:</b>', styles['label'])],
        [Paragraph(definition_raw, styles['body'])],
    ]
    findings_card = _card_table(findings_content, [CONTENT_W], padding=10)
    story.append(findings_card)
    story.append(Spacer(1, 10))

    # ── Simplified Explanation ────────────────────────────────────────────
    pat_summary = edu.get('patient_summary', '')
    action_plan = edu.get('next_steps_action_plan', 'Consult your physician.')

    explanation_rows = [
        [Paragraph('Simplified Explanation', styles['section_heading_blue'])],
        [Paragraph(pat_summary, styles['body'])],
        [Spacer(1, 4)],
        [Paragraph('<b>Next Steps / Action Plan</b>', styles['label'])],
        [Paragraph(action_plan, styles['body'])],
    ]
    explanation_card = _card_table(explanation_rows, [CONTENT_W],
                                   bg=colors.HexColor('#f8fafc'),
                                   border=C_BLUE_LT, padding=10)
    story.append(explanation_card)
    story.append(Spacer(1, 10))

    # ── Technical Details: Conformal + Ensemble ────────────────────────
    HALF_W = (CONTENT_W - 10) / 2

    # -- Conformal Set --
    c_set = conformal.get('conformal_set', [])
    conformal_rows = [[Paragraph('Conformal Prediction Set', styles['section_heading'])],
                      [Paragraph('Statistically guaranteed inclusion set (90% coverage)',
                                 styles['label'])],
                      [Spacer(1, 4)]]
    if c_set:
        for item in c_set[:10]:
            if isinstance(item, dict):
                c_name = item.get('class', 'Unknown')
                c_prob = item.get('probability', None)
                text = f'• {c_name} — {c_prob*100:.1f}%' if c_prob is not None else f'• {c_name}'
            else:
                text = f'• {str(item)}'
            conformal_rows.append([Paragraph(text, styles['body_small'])])
    else:
        conformal_rows.append([Paragraph('Conformal set not generated.', styles['body_small'])])
    conformal_card = _card_table(conformal_rows, [HALF_W], padding=8)

    # -- Ensemble --
    indiv_preds = ensemble.get('individual_predictions', {})
    ensemble_rows = [[Paragraph('Ensemble Composition', styles['section_heading'])],
                     [Paragraph('Individual model predictions', styles['label'])],
                     [Spacer(1, 4)]]
    if indiv_preds:
        header_row = Table(
            [[Paragraph('<b>Model</b>', styles['body_small']),
              Paragraph('<b>Prediction</b>', styles['body_small']),
              Paragraph('<b>Confidence</b>', styles['body_small'])]],
            colWidths=[HALF_W * 0.45, HALF_W * 0.35, HALF_W * 0.20],
        )
        header_row.setStyle(TableStyle([
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('LINEBELOW', (0, 0), (-1, 0), 0.5, C_BORDER),
        ]))
        ensemble_rows.append([header_row])

        for mname, mdata in list(indiv_preds.items())[:8]:
            display_name = mname.replace('_', ' ').replace('best ', '').title()
            m_class = mdata.get('class', '?')
            m_conf  = mdata.get('confidence', 0) or 0
            row_tbl = Table(
                [[Paragraph(display_name, styles['body_small']),
                  Paragraph(m_class, styles['body_small']),
                  Paragraph(f'{m_conf*100:.1f}%', styles['body_small'])]],
                colWidths=[HALF_W * 0.45, HALF_W * 0.35, HALF_W * 0.20],
            )
            row_tbl.setStyle(TableStyle([
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ]))
            ensemble_rows.append([row_tbl])
    else:
        ensemble_rows.append([Paragraph('No ensemble data available.', styles['body_small'])])

    ensemble_card = _card_table(ensemble_rows, [HALF_W], padding=8)

    tech_tbl = Table([[conformal_card, ensemble_card]],
                     colWidths=[HALF_W, HALF_W])
    tech_tbl.setStyle(TableStyle([
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (0, -1), 5),
        ('RIGHTPADDING', (1, 0), (1, -1), 0),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(tech_tbl)

    # ════════════════════════════════════════════════════════════════════
    #  PAGE 2: Detailed Clinical Analysis
    # ════════════════════════════════════════════════════════════════════

    if gemini_text:
        story.append(PageBreak())

        story.append(Paragraph('Detailed Clinical Analysis', styles['page2_title']))
        story.append(Paragraph(f'Diagnosis: {top_class}', styles['page2_subtitle']))
        story.append(HRFlowable(width='100%', thickness=0.8, color=C_BLUE, spaceAfter=12))

        # Clean markdown artifacts
        clean = (gemini_text
                 .replace('**', '')
                 .replace('###', '')
                 .replace('##', '')
                 .replace('#', ''))

        # Parse and render paragraphs with sub-heading detection
        for para in clean.split('\n'):
            para = para.strip()
            if not para:
                story.append(Spacer(1, 4))
                continue
            # Treat lines ending in ':' as sub-headings if short enough
            if para.endswith(':') and len(para) < 80:
                story.append(Paragraph(para, styles['page2_subheading']))
            elif para.startswith('* ') or para.startswith('- '):
                story.append(Paragraph('&nbsp;&nbsp;' + para[2:], styles['page2_body']))
            else:
                story.append(Paragraph(para, styles['page2_body']))

    # ── Build ────────────────────────────────────────────────────────────
    def _on_page(c, doc):
        # Choose callback by page number
        if doc.page == 1:
            page1_cb(c, doc)
        else:
            page2_cb(c, doc)

    doc.build(story, onFirstPage=page1_cb, onLaterPages=page2_cb)
    buf.seek(0)
    return buf
