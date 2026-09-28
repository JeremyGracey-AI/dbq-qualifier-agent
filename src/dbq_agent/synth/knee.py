"""Generate fillable synthetic knee DBQ PDFs (AcroForm) from a `KneeTruth`.

Everything here is fictional. The PDF is a structural replica: the same sections and question
types as VA Form 21-0960M-9, so ingest/extract can be developed and tested without any real
veteran record. Swapping in the real form later is a field-name mapping, nothing more.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

from dbq_agent.synth.form_spec import ALL_WIDGETS, PAGES, Widget

PAGE_W, PAGE_H = int(letter[0]), int(letter[1])
MARGIN = 54
LINE = 16
LABEL_FONT = ("Helvetica", 9)
HEAD_FONT = ("Helvetica-Bold", 11)
NOTE_FONT = ("Helvetica-Oblique", 8)


class KneeTruth(BaseModel):
    """Field values for one synthetic case. Keys are form_spec field names."""

    case_id: str
    values: dict[str, str] = Field(default_factory=dict)

    def get(self, name: str) -> str:
        return self.values.get(name, "")


def _draw_text_widget(c: canvas.Canvas, w: Widget, value: str, x: int, y: int) -> int:
    c.setFont(*LABEL_FONT)
    c.drawString(x, y, w.label)
    fx = x + 240
    if w.kind == "textarea":
        lines = simpleSplit(value, "Helvetica", 8, 420) if value else [""]
        height = max(3, min(len(lines), 8)) * 11 + 6
        c.acroForm.textfield(
            name=w.name,
            value=value,
            x=x,
            y=y - height - 6,
            width=480,
            height=height,
            fontSize=8,
            borderWidth=1,
            fieldFlags="multiline",
            forceBorder=True,
        )
        return y - height - 6 - LINE
    c.acroForm.textfield(
        name=w.name,
        value=value,
        x=fx,
        y=y - 5,
        width=w.width,
        height=14,
        fontSize=8,
        borderWidth=1,
        forceBorder=True,
    )
    return y - LINE - 6


def _draw_radio(c: canvas.Canvas, w: Widget, value: str, x: int, y: int) -> int:
    c.setFont(*LABEL_FONT)
    label_lines = simpleSplit(w.label, LABEL_FONT[0], LABEL_FONT[1], 300)
    for i, line in enumerate(label_lines):
        c.drawString(x, y - i * 11, line)
    ry = y - (len(label_lines) - 1) * 11
    ox = x + 320
    for opt_value, opt_label in w.options:
        c.acroForm.radio(
            name=w.name,
            value=opt_value,
            x=ox,
            y=ry - 3,
            size=10,
            selected=(value == opt_value),
            borderWidth=1,
            forceBorder=True,
        )
        c.setFont("Helvetica", 8)
        short = opt_label if len(opt_label) <= 22 else opt_label[:21] + "…"
        c.drawString(ox + 13, ry - 1, short)
        ox += 14 + max(30, int(len(short) * 4.2))
        if ox > PAGE_W - MARGIN - 40:
            ox = x + 320
            ry -= 12
    return ry - LINE - 4


def _draw_checkbox(c: canvas.Canvas, w: Widget, value: str, x: int, y: int) -> int:
    c.acroForm.checkbox(
        name=w.name,
        x=x + 8,
        y=y - 3,
        size=10,
        checked=(value == "Yes"),
        borderWidth=1,
        forceBorder=True,
    )
    c.setFont(*LABEL_FONT)
    c.drawString(x + 24, y, w.label)
    return y - LINE


def build_pdf(truth: KneeTruth, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_path), pagesize=letter)
    c.setTitle(f"Synthetic knee DBQ {truth.case_id}")
    for page_no, widgets in enumerate(PAGES, start=1):
        y = PAGE_H - MARGIN
        x = MARGIN
        c.setFont(*NOTE_FONT)
        c.drawRightString(
            PAGE_W - MARGIN, PAGE_H - 30, f"synthetic case {truth.case_id} — page {page_no}"
        )
        for w in widgets:
            if w.kind == "heading":
                y -= 6
                c.setFont(*HEAD_FONT)
                c.drawString(x, y, w.label)
                y -= LINE + 2
            elif w.kind == "note":
                c.setFont(*NOTE_FONT)
                for line in simpleSplit(w.label, NOTE_FONT[0], NOTE_FONT[1], PAGE_W - 2 * MARGIN):
                    c.drawString(x, y, line)
                    y -= 11
                y -= 4
            elif w.kind in ("text", "textarea"):
                y = _draw_text_widget(c, w, truth.get(w.name), x, y)
            elif w.kind == "radio":
                y = _draw_radio(c, w, truth.get(w.name), x, y)
            elif w.kind == "checkbox":
                y = _draw_checkbox(c, w, truth.get(w.name), x, y)
            if y < MARGIN + 40:
                raise RuntimeError(f"page {page_no} overflowed at field {w.name}; split the page")
        c.showPage()
    c.save()
    return out_path


def blank_values() -> dict[str, str]:
    """Every field present, empty (radios unselected, checkboxes Off)."""
    return {name: ("Off" if w.kind == "checkbox" else "") for name, w in ALL_WIDGETS.items()}
