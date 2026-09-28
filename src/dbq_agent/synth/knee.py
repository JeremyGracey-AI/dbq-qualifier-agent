"""Generate synthetic knee DBQ PDFs from a `KneeTruth`.

Everything here is fictional. The PDF is a structural replica: the same sections and question
types as VA Form 21-0960M-9, so ingest/extract can be developed and tested without any real
veteran record. Swapping in the real form later is a field-name mapping, nothing more.

Two renditions share one layout walk:
- `build_pdf`: fillable AcroForm (what a clinician's PDF tool produces).
- `build_scanned_pdf`: a print rendition (values as text, filled dots, checked boxes) that is
  rasterized and degraded like a scan — noise, skew, JPEG — into an image-only PDF with no text
  layer. This is the fixture for the OCR ingest path.
"""

from __future__ import annotations

import io
import random
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase.pdfmetrics import stringWidth
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


def _draw_text_widget(
    c: canvas.Canvas, w: Widget, value: str, x: int, y: int, flat: bool = False
) -> int:
    c.setFont(*LABEL_FONT)
    c.drawString(x, y, w.label)
    fx = x + 240
    if w.kind == "textarea":
        lines = simpleSplit(value, "Helvetica", 8, 420) if value else [""]
        height = max(3, min(len(lines), 8)) * 11 + 6
        if flat:
            c.setLineWidth(0.8)
            c.rect(x, y - height - 6, 480, height, stroke=1, fill=0)
            c.setFont("Helvetica", 8)
            ty = y - 6 - 10
            for line in lines[:8]:
                c.drawString(x + 4, ty, line)
                ty -= 11
            return y - height - 6 - LINE
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
    if flat:
        c.setLineWidth(0.8)
        c.rect(fx, y - 5, w.width, 14, stroke=1, fill=0)
        # a printed form fits the value to its box; shrink the font rather than overflow
        size = 8.0
        while size > 5 and stringWidth(value, "Helvetica", size) > w.width - 6:
            size -= 0.5
        c.setFont("Helvetica", size)
        c.drawString(fx + 3, y - 1, value)
        return y - LINE - 6
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


def _draw_radio(c: canvas.Canvas, w: Widget, value: str, x: int, y: int, flat: bool = False) -> int:
    c.setFont(*LABEL_FONT)
    label_lines = simpleSplit(w.label, LABEL_FONT[0], LABEL_FONT[1], 300)
    for i, line in enumerate(label_lines):
        c.drawString(x, y - i * 11, line)
    ry = y - (len(label_lines) - 1) * 11
    ox = x + 320
    for opt_value, opt_label in w.options:
        if flat:
            c.setLineWidth(0.8)
            c.circle(ox + 5, ry + 2, 5, stroke=1, fill=0)
            if value == opt_value:
                c.circle(ox + 5, ry + 2, 2.6, stroke=0, fill=1)
        else:
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


def _draw_checkbox(
    c: canvas.Canvas, w: Widget, value: str, x: int, y: int, flat: bool = False
) -> int:
    if flat:
        c.setLineWidth(0.8)
        c.rect(x + 8, y - 3, 10, 10, stroke=1, fill=0)
        if value == "Yes":
            c.setLineWidth(1.4)
            c.line(x + 10, y - 1, x + 16, y + 5)
            c.line(x + 10, y + 5, x + 16, y - 1)
    else:
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


def build_pdf(truth: KneeTruth, out_path: Path, flat: bool = False) -> Path:
    """Write the form. `flat=True` draws a print rendition instead of AcroForm widgets."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # invariant: no timestamps or random IDs, so regenerating unchanged cases is a no-op in git
    c = canvas.Canvas(str(out_path), pagesize=letter, invariant=True)
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
                y = _draw_text_widget(c, w, truth.get(w.name), x, y, flat)
            elif w.kind == "radio":
                y = _draw_radio(c, w, truth.get(w.name), x, y, flat)
            elif w.kind == "checkbox":
                y = _draw_checkbox(c, w, truth.get(w.name), x, y, flat)
            if y < MARGIN + 40:
                raise RuntimeError(f"page {page_no} overflowed at field {w.name}; split the page")
        c.showPage()
    c.save()
    return out_path


def build_scanned_pdf(
    truth: KneeTruth, out_path: Path, dpi: int = 200, seed: int | None = None
) -> Path:
    """Print rendition → raster → scan-like degradation → image-only PDF (no text layer)."""
    import pypdfium2 as pdfium
    from PIL import Image, ImageFilter

    rng = random.Random(seed if seed is not None else truth.case_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        flat = build_pdf(truth, Path(tmp) / "flat.pdf", flat=True)
        pdf = pdfium.PdfDocument(str(flat))
        pages: list[Image.Image] = []
        scale: float = dpi / 72
        for i in range(len(pdf)):
            page = pdf[i].render(scale=scale)  # pyright: ignore[reportArgumentType]  # untyped
            img = page.to_pil().convert("L")
            img = img.rotate(
                rng.uniform(-0.4, 0.4), resample=Image.Resampling.BICUBIC, fillcolor=255
            )
            img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
            px = img.load()
            w, h = img.size
            for _ in range(w * h // 400):  # sparse speckle
                x, y = rng.randrange(w), rng.randrange(h)
                px[x, y] = max(0, px[x, y] - rng.randint(20, 90))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=rng.randint(55, 70))
            pages.append(Image.open(io.BytesIO(buf.getvalue())).convert("L"))
        pages[0].save(
            out_path,
            format="PDF",
            save_all=True,
            append_images=pages[1:],
            resolution=dpi,
        )
    return out_path


def blank_values() -> dict[str, str]:
    """Every field present, empty (radios unselected, checkboxes Off)."""
    return {name: ("Off" if w.kind == "checkbox" else "") for name, w in ALL_WIDGETS.items()}
