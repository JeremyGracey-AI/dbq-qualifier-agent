"""Read a fillable DBQ PDF into typed field values with page provenance.

Deterministic on purpose: AcroForm field values are read directly (no OCR, no model). Each
field records the page its widget sits on so every downstream claim can point back to a
page + field. A scanned-DBQ path (OCR + layout) is out of scope for this slice and would plug
in here as a second `IngestedDoc` producer.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from dbq_agent.models import FieldKind, FieldValue, IngestedDoc


def _norm_value(raw: Any) -> str:
    if raw is None:
        return ""
    s = str(raw)
    if s.startswith("/"):
        s = s[1:]
    return s.strip()


def _widget_field_name(obj: Any) -> str | None:
    """A widget's field name, following /Parent for radio kids."""
    seen = 0
    while obj is not None and seen < 8:
        name = obj.get("/T")
        if name is not None:
            return str(name)
        parent = obj.get("/Parent")
        obj = parent.get_object() if parent is not None else None
        seen += 1
    return None


def _field_kind(obj: Any) -> FieldKind:
    ft = obj.get("/FT")
    if ft is None:
        parent = obj.get("/Parent")
        ft = parent.get_object().get("/FT") if parent is not None else None
    if ft == "/Btn":
        flags = int(obj.get("/Ff", 0) or 0)
        if flags == 0:
            parent = obj.get("/Parent")
            if parent is not None:
                flags = int(parent.get_object().get("/Ff", 0) or 0)
        return "radio" if flags & (1 << 15) else "checkbox"
    return "text"


def ingest_pdf(path: Path, case_id: str | None = None) -> IngestedDoc:
    reader = PdfReader(str(path))
    pages: dict[str, int] = {}
    kinds: dict[str, FieldKind] = {}
    for page_no, page in enumerate(reader.pages, start=1):
        annots = page.get("/Annots") or []
        for annot in annots:
            obj = annot.get_object()
            if obj.get("/Subtype") != "/Widget":
                continue
            name = _widget_field_name(obj)
            if name is None:
                continue
            pages.setdefault(name, page_no)
            kinds.setdefault(name, _field_kind(obj))

    fields: dict[str, FieldValue] = {}
    raw_fields = reader.get_fields() or {}
    for name, fld in raw_fields.items():
        value = _norm_value(fld.get("/V"))
        kind = kinds.get(name, "text")
        if kind == "checkbox" and value == "":
            value = "Off"
        fields[name] = FieldValue(name=name, page=pages.get(name, 1), value=value, kind=kind)

    cid = case_id or re.sub(r"\.pdf$", "", path.name, flags=re.IGNORECASE)
    return IngestedDoc(
        case_id=cid,
        source_path=str(path),
        page_count=len(reader.pages),
        fields=fields,
    )
