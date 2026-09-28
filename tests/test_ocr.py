"""Scanned-DBQ ingest.

Two layers: the recovery logic on synthetic inputs (no OCR engine needed), and end-to-end
recovery of image-only renditions through tesseract (skipped when tesseract is not
installed). The end-to-end layer is the contract: a scan of a synthetic case must yield the
same structured values as its fillable original, and the pipeline must reach the same
outcome from it.
"""

from __future__ import annotations

import json
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("PIL")
pytest.importorskip("numpy")
pytest.importorskip("pypdfium2")

from PIL import Image, ImageDraw

from dbq_agent.ingest import ingest_any
from dbq_agent.ingest_ocr import (
    Line,
    OptionHit,
    TesseractEngine,
    Word,
    _match_option,
    align_labels,
    erase_rules,
    ink_runs,
    label_match,
    mark_fill,
)
from dbq_agent.pipeline import Context, run
from dbq_agent.report import build_report, to_markdown
from dbq_agent.synth.form_spec import ALL_WIDGETS, PAGE_OF
from dbq_agent.synth.knee import KneeTruth, build_scanned_pdf

SCANNED_CASES = ("knee_08", "knee_11")  # adequate ROM case + DC 5257 instability case
needs_tesseract = pytest.mark.skipif(
    not TesseractEngine.available(), reason="tesseract binary not installed"
)


def _line(*texts: str, y: int = 100, x: int = 100, h: int = 20) -> Line:
    words: list[Word] = []
    for t in texts:
        words.append(Word(t, x, y, x + 12 * len(t), y + h, 0.9))
        x += 12 * len(t) + 8
    return Line(words)


# --------------------------------------------------------------------------------------
# recovery logic, no engine
# --------------------------------------------------------------------------------------


def test_label_match_spans_ocr_tokens_not_ours() -> None:
    ln = _line("Flexion", "(0-140),", "degrees", "60")
    score, n = label_match("Flexion (0-140), degrees", ln)
    assert score > 0.95
    assert n == 3  # the value word is not part of the label


def test_alignment_survives_a_dropped_line_without_cascading() -> None:
    labels = ["Veteran name", "Date of birth", "SSN", "Date of examination", "Examiner"]
    # OCR dropped the "Veteran name" line entirely; a later free-text line starts with
    # "Veteran" and would win a greedy first-match search
    lines = [
        _line("Date", "of", "birth", "01/02/1970", y=100),
        _line("SSN", "900-11-2222", y=160),
        _line("Date", "of", "examination", "08/12/2026", y=220),
        _line("Examiner", "Dr.", "A.", "Fictional,", "MD", y=280),
        _line("Veteran", "reports", "right", "knee", "pain", y=400),
    ]
    hits = align_labels(labels, lines)
    assert 0 not in hits  # left unmatched rather than mis-anchored downstream
    assert {k: v[0] for k, v in hits.items()} == {1: 0, 2: 1, 3: 2, 4: 3}


def test_alignment_respects_order_for_repeated_labels() -> None:
    labels = ["Describe (frequency, duration, severity, functional loss)", "Describe"]
    lines = [
        _line("Describe", "(frequency,", "duration,", "severity,", "functional", "loss)", y=100),
        _line("Describe", y=300),
    ]
    hits = align_labels(labels, lines)
    assert hits[0][0] == 0 and hits[1][0] == 1


def test_option_matching_accepts_ocr_shapes() -> None:
    # glyph-only word, glued glyph, merged tokens, truncated label
    words = _line(
        "©", "Yes", "@yYes", "ONo", "Nottested", "Unable", "to", "say", "without..."
    ).words
    assert _match_option(words, 0, "Yes") == OptionHit(1, 2, False)
    assert _match_option(words, 2, "Yes") == OptionHit(2, 3, True)
    assert _match_option(words, 3, "No") == OptionHit(3, 4, True)
    assert _match_option(words, 4, "Not tested") == OptionHit(4, 5, False)
    assert _match_option(words, 5, "Unable to say without") == OptionHit(5, 9, False)
    assert _match_option(words, 0, "Severe") is None


def _darkest(img: Image.Image) -> int:
    lo, _ = cast(tuple[int, int], img.getextrema())  # "L" mode: (min, max)
    return lo


def _mark_image(filled: bool, kind: str = "circle") -> Image.Image:
    img = Image.new("L", (120, 60), 255)
    d = ImageDraw.Draw(img)
    if kind == "circle":
        d.ellipse((20, 15, 48, 43), outline=0, width=2)
        if filled:
            d.ellipse((27, 22, 41, 36), fill=0)
    else:
        d.rectangle((20, 15, 48, 43), outline=0, width=2)
        if filled:
            d.line((24, 19, 44, 39), fill=0, width=3)
            d.line((24, 39, 44, 19), fill=0, width=3)
    d.text((60, 20), "Yes", fill=0)
    return img


@pytest.mark.parametrize("kind", ["circle", "box"])
def test_mark_fill_separates_filled_from_empty(kind: str) -> None:
    empty, _ = mark_fill(_mark_image(False, kind), (10, 5, 55, 55))
    filled, box = mark_fill(_mark_image(True, kind), (10, 5, 55, 55))
    assert empty < 0.05
    assert filled > 0.5
    assert box is not None and box[0] >= 18 and box[2] <= 50  # the mark, not the window


def test_ink_runs_find_the_mark_and_the_label_separately() -> None:
    runs = ink_runs(_mark_image(True), (0, 0, 120, 60))
    assert len(runs) >= 2
    assert runs[0][0] >= 18 and runs[0][1] <= 50  # the circle
    assert runs[-1][0] >= 58  # the label text


def test_erase_rules_removes_box_borders_but_keeps_glyphs() -> None:
    img = Image.new("L", (200, 60), 255)
    d = ImageDraw.Draw(img)
    d.rectangle((10, 8, 190, 52), outline=0, width=2)  # a 44 px tall, 180 px wide box
    d.text((20, 20), "60", fill=0)
    out = erase_rules(img, min_horizontal=60, min_vertical=30)
    assert _darkest(out.crop((0, 0, 200, 12))) == 255  # top border gone
    assert _darkest(out.crop((0, 0, 14, 60))) == 255  # left border gone
    assert _darkest(out.crop((18, 18, 40, 34))) < 128  # glyph ink kept


# --------------------------------------------------------------------------------------
# end to end, tesseract
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def scanned_dir(tmp_path_factory: pytest.TempPathFactory, truth: dict[str, Any]) -> Path:
    out = tmp_path_factory.mktemp("scanned")
    for cid in SCANNED_CASES:
        build_scanned_pdf(KneeTruth(case_id=cid, values=truth[cid]["values"]), out / f"{cid}.pdf")
    (out / "truth.json").write_text(json.dumps({c: truth[c] for c in SCANNED_CASES}))
    return out


@needs_tesseract
def test_scan_recovers_every_field(scanned_dir: Path, truth: dict[str, Any]) -> None:
    for cid in SCANNED_CASES:
        doc = ingest_any(scanned_dir / f"{cid}.pdf")
        assert doc.source_kind == "ocr" and doc.dpi == 200
        assert set(doc.fields) == set(ALL_WIDGETS)
        for name, w in ALL_WIDGETS.items():
            if w.kind in ("heading", "note"):
                continue
            expected = truth[cid]["values"].get(name, "Off" if w.kind == "checkbox" else "")
            got = doc.fields[name]
            assert got.page == PAGE_OF[name], f"{cid}: page for {name}"
            assert got.bbox is not None, f"{cid}: no provenance for {name}"
            if w.kind == "textarea":
                sim = SequenceMatcher(None, expected, got.value).ratio()
                assert sim >= 0.9, f"{cid}: {name} similarity {sim:.2f}: {got.value!r}"
            else:
                assert got.value == expected, f"{cid}: {name}: {got.value!r} != {expected!r}"


@needs_tesseract
def test_scan_reaches_the_same_outcome_as_the_fillable_form(
    scanned_dir: Path, cases_dir: Path, truth: dict[str, Any], ctx: Context
) -> None:
    for cid in SCANNED_CASES:
        claim = date.fromisoformat(truth[cid]["claim_date"] or "2026-09-01")
        scanned = build_report(run(scanned_dir / f"{cid}.pdf", claim, ctx), ctx.kb)
        fillable = build_report(run(cases_dir / f"{cid}.pdf", claim, ctx), ctx.kb)
        assert scanned.adequate is fillable.adequate
        assert {dc: ln.pct for dc, ln in scanned.provisional.items()} == {
            dc: ln.pct for dc, ln in fillable.provisional.items()
        }
        assert {g.rule_id for g in scanned.gaps} == {g.rule_id for g in fillable.gaps}
        assert scanned.pipeline.source_kind == "ocr" and not fillable.ocr_boxes
        # every evidence field on the scan points at ink a reviewer can look at
        for line in scanned.provisional.values():
            for span in line.evidence:
                assert span.field in scanned.ocr_boxes
        md = to_markdown(scanned, ctx.kb)
        assert "scanned, OCR at 200 dpi" in md and "[px " in md


def test_ingest_any_uses_form_fields_when_present(cases_dir: Path) -> None:
    doc = ingest_any(cases_dir / "knee_01.pdf")
    assert doc.source_kind == "acroform" and doc.dpi is None
