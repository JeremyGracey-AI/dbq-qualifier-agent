"""OCR ingest for scanned (image-only) DBQs.

The AcroForm path reads values by field name. A scan has no fields, only ink, so this module
recovers the same logical fields from the page image in two passes:

1. **Anchor pass.** Rasterize each page (pypdfium2), OCR the whole page into words with boxes
   (tesseract in sparse-text mode, which keeps every label on a form that block mode drops),
   group the words into lines, and align the form spec's labels to those lines with a
   monotone dynamic-programming alignment: labels appear in reading order, so the alignment
   that maximises total label similarity while keeping order is the anchoring. One dropped
   or garbled line then costs one field, not every field after it.
2. **Zone pass.** Read each answer from a small region relative to its anchor, by widget kind:
   - text: the strip right of the label is re-OCR'd on its own (single-line mode, box borders
     erased, upscaled). Page-level OCR routinely drops a lone "60" inside a bordered box; a
     targeted crop does not.
   - textarea: the page-level lines under the label, up to the next anchored label; the box
     is re-OCR'd if the page pass saw nothing there.
   - radio: the option strip is re-OCR'd so option labels come back as clean words, then the
     mark beside each label is read from pixels (OCR cannot read a filled circle; ink can).
   - checkbox: the mark left of the label, from pixels.
   A label with no aligned line (dropped by OCR) is looked for once more in the strip between
   its neighbours' anchors before the field is given up as blank.

Provenance is page + pixel bbox on every recovered value, so a reviewer can look at exactly
the ink a claim rests on. The rest of the pipeline is unchanged: it sees an `IngestedDoc`
keyed by the same field names, with `source_kind="ocr"`.

Layout assumptions (true of the synthetic form; documented so a different form is a known
change, not a surprise): a text answer sits right of its label; radio options sit on the
label's last line or the lines just below it; a checkbox sits left of its label.

The engine is behind `OcrEngine` so tesseract can be swapped without touching the recovery
logic. Optional extra: `uv sync --extra ocr` plus a tesseract binary on PATH.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from dbq_agent.models import FieldKind, FieldValue, IngestedDoc
from dbq_agent.synth.form_spec import PAGES, Widget

if TYPE_CHECKING:
    from PIL.Image import Image

BBox = tuple[int, int, int, int]
OcrMode = Literal["sparse", "line"]


@dataclass(frozen=True)
class Word:
    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    conf: float  # 0-1

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class Line:
    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def y0(self) -> int:
        return min(w.y0 for w in self.words)

    @property
    def y1(self) -> int:
        return max(w.y1 for w in self.words)

    @property
    def cy(self) -> float:
        return sum(w.cy for w in self.words) / len(self.words)

    @property
    def h(self) -> float:
        return sum(w.h for w in self.words) / len(self.words)


class OcrEngine(Protocol):
    name: str

    def words(self, image: Image, mode: OcrMode = "sparse") -> list[Word]: ...


@dataclass
class TesseractEngine:
    name: str = "tesseract"

    def __post_init__(self) -> None:
        # pytesseract hands tesseract a temp file. With TMPDIR unset on macOS that lands in
        # /tmp/..., a symlink Leptonica can fail to open from a sandboxed shell, while the
        # real path (/private/tmp/...) always opens. Resolve the default temp dir once.
        tempfile.tempdir = os.path.realpath(tempfile.gettempdir())

    @staticmethod
    def available() -> bool:
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
        except Exception:
            return False
        return True

    def words(self, image: Image, mode: OcrMode = "sparse") -> list[Word]:
        import pytesseract

        psm = 11 if mode == "sparse" else 7
        data = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT, config=f"--psm {psm}"
        )
        out: list[Word] = []
        for i, text in enumerate(data["text"]):
            t = str(text).strip()
            if not t:
                continue
            conf = float(data["conf"][i])
            x, y = int(data["left"][i]), int(data["top"][i])
            w, h = int(data["width"][i]), int(data["height"][i])
            out.append(Word(t, x, y, x + w, y + h, max(conf, 0.0) / 100.0))
        return out


# --------------------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------------------

_NORM = re.compile(r"[^a-z0-9 ]+")


def _norm(text: str) -> str:
    return _NORM.sub(" ", text.lower()).strip()


def _squash(text: str) -> str:
    return _norm(text).replace(" ", "")


def group_lines(words: list[Word]) -> list[Line]:
    """Cluster words into lines by vertical centre; tolerant of slight skew."""
    lines: list[Line] = []
    for w in sorted(words, key=lambda w: (w.cy, w.x0)):
        placed = False
        for ln in lines:
            if abs(ln.cy - w.cy) <= max(6.0, 0.55 * max(ln.h, w.h)):
                ln.words.append(w)
                placed = True
                break
        if not placed:
            lines.append(Line([w]))
    for ln in lines:
        ln.words.sort(key=lambda w: w.x0)
    lines.sort(key=lambda ln: ln.cy)
    return lines


def label_match(label: str, line: Line) -> tuple[float, int]:
    """How well the line *starts with* the label, and how many OCR words the label spans.

    Returns (score, n_words). OCR tokenisation differs from ours ("(0-140)," is one OCR word
    but two normalised tokens), so the span is found by growing a prefix, not by counting.
    """
    lab = _norm(label)
    if not lab or not line.words:
        return 0.0, 0
    best_score, best_n = 0.0, 0
    limit = min(len(line.words), len(lab.split()) + 3)
    for n in range(1, limit + 1):
        head = _norm(" ".join(w.text for w in line.words[:n]))
        score = SequenceMatcher(None, lab, head).ratio()
        if score > best_score:
            best_score, best_n = score, n
        if len(head) > len(lab) + 8:
            break
    return best_score, best_n


LABEL_THRESHOLD = 0.72


def align_labels(labels: list[str], lines: list[Line]) -> dict[int, tuple[int, int]]:
    """Monotone alignment of labels (in form order) to OCR lines (in reading order).

    Maximises the total similarity of matched pairs, counting only pairs at or above
    `LABEL_THRESHOLD`; a label may stay unmatched and a line may be skipped. Returns
    label index -> (line index, label word count).
    """
    n, m = len(labels), len(lines)
    scores = [[label_match(lab, ln) for ln in lines] for lab in labels]
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]  # 0 skip label, 1 skip line, 2 match
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            b, c = best[i - 1][j], 0
            if best[i][j - 1] > b:
                b, c = best[i][j - 1], 1
            s = scores[i - 1][j - 1][0]
            if s >= LABEL_THRESHOLD and best[i - 1][j - 1] + s > b:
                b, c = best[i - 1][j - 1] + s, 2
            best[i][j], back[i][j] = b, c
    out: dict[int, tuple[int, int]] = {}
    i, j = n, m
    while i > 0 and j > 0:
        c = back[i][j]
        if c == 2:
            out[i - 1] = (j - 1, scores[i - 1][j - 1][1])
            i, j = i - 1, j - 1
        elif c == 1:
            j -= 1
        else:
            i -= 1
    return out


# --------------------------------------------------------------------------------------
# Pixel helpers
# --------------------------------------------------------------------------------------


def _ink(v: int) -> int:
    return 255 if v < 128 else 0


def _clip(image: Image, window: BBox) -> BBox | None:
    x0, y0, x1, y1 = window
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(image.width, x1), min(image.height, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def erase_rules(image: Image, min_horizontal: int, min_vertical: int) -> Image:
    """Whiten straight dark runs (box borders, underlines) longer than the given lengths.

    Glyph strokes are short in both directions; a border or rule is long in one. Erasing
    only the long runs (plus 2 px either side) leaves the text untouched.
    """
    import numpy as np
    from PIL import Image as PILImage

    a = np.asarray(image, dtype=np.uint8).copy()
    dark = a < 200  # include anti-aliased edges so a skewed rule is one run, not fragments
    for axis, min_len in ((1, min_horizontal), (0, min_vertical)):
        arr = dark if axis == 1 else dark.T
        for i in range(arr.shape[0]):
            row = arr[i]
            if not row.any():
                continue
            edges = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
            for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
                if e - s >= min_len:
                    if axis == 1:
                        a[max(0, i - 2) : i + 3, s:e] = 255
                    else:
                        a[s:e, max(0, i - 2) : i + 3] = 255
    return PILImage.fromarray(a)


def _runs(flags: list[bool], offset: int, min_gap: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    gap = 0
    for i, on in enumerate(flags):
        if on:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                runs.append((offset + start, offset + i - gap + 1))
                start, gap = None, 0
    if start is not None:
        runs.append((offset + start, offset + len(flags) - gap))
    return runs


def ink_runs(image: Image, window: BBox, min_gap: int = 2) -> list[tuple[int, int]]:
    """Horizontal extents (page x0, x1) of the ink blobs in `window`, left to right."""
    import numpy as np

    clipped = _clip(image, window)
    if clipped is None:
        return []
    dark = np.asarray(image.crop(clipped)) < 128
    return _runs([bool(v) for v in dark.sum(axis=0) >= 2], clipped[0], min_gap)


def ink_rows(image: Image, window: BBox) -> list[tuple[int, int]]:
    """Vertical extents (page y0, y1) of the ink blobs in `window`, top to bottom."""
    import numpy as np

    clipped = _clip(image, window)
    if clipped is None:
        return []
    dark = np.asarray(image.crop(clipped)) < 128
    return _runs([bool(v) for v in dark.sum(axis=1) >= 2], clipped[1], 1)


def mark_fill(image: Image, window: BBox) -> tuple[float, BBox | None]:
    """Ink fraction *inside* the mark drawn in `window` (a radio circle or checkbox), and the
    mark's tight bbox.

    Finds the dark outline's bounding box inside the window, shrinks it by 30% per side and
    measures ink there: an empty mark is ~0, a filled dot or an X is well above 0.5. Robust
    to the exact size and position of the mark, unlike a raw density over the window.
    """
    clipped = _clip(image, window)
    if clipped is None:
        return 0.0, None
    x0, y0, _, _ = clipped
    region = image.crop(clipped).point(_ink)
    box = region.getbbox()
    if box is None:
        return 0.0, None
    bx0, by0, bx1, by1 = box
    bw, bh = bx1 - bx0, by1 - by0
    if bw < 6 or bh < 6:
        return 0.0, None
    ix0, iy0 = bx0 + int(0.3 * bw), by0 + int(0.3 * bh)
    ix1, iy1 = bx1 - int(0.3 * bw), by1 - int(0.3 * bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0, None
    inner = region.crop((ix0, iy0, ix1, iy1))
    fill = inner.histogram()[255] / float((ix1 - ix0) * (iy1 - iy0))
    return fill, (x0 + bx0, y0 + by0, x0 + bx1, y0 + by1)


# --------------------------------------------------------------------------------------
# Field recovery
# --------------------------------------------------------------------------------------

MARK_THRESHOLD = 0.2  # ink fraction inside a mark's interior that counts as filled
MIN_WORD_CONF = 0.5  # zone OCR words below this are treated as noise
ZONE_SCALE = 2
_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1", "S": "5"})


@dataclass
class _PageCtx:
    image: Image
    lines: list[Line]
    engine: OcrEngine
    dpi: int = 200

    def pt(self, points: float) -> int:
        return int(points * self.dpi / 72)

    @property
    def left_margin(self) -> int:
        return int(0.03 * self.image.width)

    @property
    def right_edge(self) -> int:
        return self.image.width - int(0.01 * self.image.width)

    def band(self, j: int, pad: float) -> tuple[int, int]:
        """Vertical extent of line `j` plus `pad` px each side, never reaching into the
        neighbouring lines (wrapped option rows sit only a few px apart)."""
        ln = self.lines[j]
        y0, y1 = int(ln.y0 - pad), int(ln.y1 + pad)
        if j > 0:
            prev = self.lines[j - 1]
            y0 = max(y0, min(ln.y0, (prev.y1 + ln.y0) // 2))
        if j + 1 < len(self.lines):
            nxt = self.lines[j + 1]
            y1 = min(y1, max(ln.y1, (ln.y1 + nxt.y0) // 2))
        return y0, y1


@dataclass(frozen=True)
class _Anchor:
    idx: int  # line index of the label
    n_label: int  # OCR words the label spans on that line
    stop: int  # line index of the next anchored label (exclusive bound for answers)


def _zone(
    ctx: _PageCtx, window: BBox, *, h: float, erase: bool = False, mode: OcrMode = "line"
) -> list[Word] | None:
    """Re-OCR one strip of the page; words come back in page coordinates.

    Returns None when the strip holds no ink at all (a blank answer), so callers can tell
    "nothing written" from "written but unreadable".
    """
    from PIL import Image as PILImage

    clipped = _clip(ctx.image, window)
    if clipped is None:
        return None
    x0, y0, _, _ = clipped
    crop = ctx.image.crop(clipped)
    if erase:  # a box border is >= 12 pt tall / 20 pt wide; an 8-10 pt glyph stroke is not
        crop = erase_rules(crop, min_horizontal=ctx.pt(20), min_vertical=ctx.pt(9))
    ink = crop.point(_ink).getbbox()
    if ink is None:
        return None
    # tighten to the ink (plus a margin) so an almost-empty strip does not skew line finding
    tx0, ty0 = max(0, ink[0] - int(0.6 * h)), max(0, ink[1] - int(0.3 * h))
    tx1, ty1 = min(crop.width, ink[2] + int(0.6 * h)), min(crop.height, ink[3] + int(0.3 * h))
    crop = crop.crop((tx0, ty0, tx1, ty1))
    x0, y0 = x0 + tx0, y0 + ty0
    crop = crop.resize(
        (crop.width * ZONE_SCALE, crop.height * ZONE_SCALE), PILImage.Resampling.LANCZOS
    )
    return [
        Word(
            w.text,
            x0 + w.x0 // ZONE_SCALE,
            y0 + w.y0 // ZONE_SCALE,
            x0 + w.x1 // ZONE_SCALE,
            y0 + w.y1 // ZONE_SCALE,
            w.conf,
        )
        for w in ctx.engine.words(crop, mode)
    ]


def _label_end(line: Line, n_label: int) -> int:
    return line.words[min(n_label, len(line.words)) - 1].x1


def _fix_code(text: str) -> str:
    """OCR confusions in an alphanumeric code (ICD-10: letter, digits, '.', alnum)."""
    text = re.sub(r"^[$5]", "S", text)
    return re.sub(r"(?<=\d)[Oo](?=[\d.]|$)", "0", text)


def _read_text(ctx: _PageCtx, a: _Anchor, w: Widget) -> FieldValue:
    line = ctx.lines[a.idx]
    h = line.h
    x0 = _label_end(line, a.n_label) + int(0.4 * h)
    y0, y1 = ctx.band(a.idx, 0.6 * h)
    window = (x0, y0, ctx.right_edge, y1)
    zone = _zone(ctx, window, h=h, erase=True)
    vals: list[Word] = []
    if zone is not None:
        vals = [x for x in zone if x.conf >= MIN_WORD_CONF and (x.x1 - x.x0) >= 3]
        if not vals:  # ink but nothing read: fall back to what the page-level pass saw there
            vals = [x for x in line.words[a.n_label :] if x.x0 >= x0 and not _is_noise(x)]
    text = " ".join(x.text for x in vals).strip()
    if w.numeric:
        m = re.search(r"-?\d{1,3}", text.translate(_DIGIT_FIX))
        text = m.group(0) if m else ""
    elif w.code:
        text = _fix_code(text)
    return FieldValue(
        name=w.name,
        page=0,
        value=text,
        kind="text",
        bbox=_bbox(vals) if vals else window,
        confidence=(sum(x.conf for x in vals) / len(vals)) if vals else None,
    )


def _is_noise(word: Word) -> bool:
    return word.conf < 0.3 or not _norm(word.text)


def _read_textarea(ctx: _PageCtx, a: _Anchor, w: Widget) -> FieldValue:
    """Lines under the label up to the next anchored label, or a visibly larger gap."""
    line = ctx.lines[a.idx]
    collected: list[Line] = []
    for j in range(a.idx + 1, a.stop):
        ln = ctx.lines[j]
        if collected and (ln.y0 - collected[-1].y1) > 2.2 * line.h:
            break
        collected.append(ln)
    words = [x for ln in collected for x in ln.words if not _is_noise(x)]
    if not words:  # the page pass saw nothing in the box; look once more, on its own
        y_end = ctx.lines[a.stop].y0 - 2 if a.stop < len(ctx.lines) else ctx.image.height
        zone = _zone(
            ctx,
            (ctx.left_margin, line.y1 + 2, ctx.right_edge, y_end),
            h=line.h,
            erase=True,
            mode="sparse",
        )
        words = [x for x in (zone or []) if not _is_noise(x)]
        words.sort(key=lambda x: (x.cy // max(1, int(line.h)), x.x0))
    text = " ".join(x.text for x in words).strip()
    return FieldValue(
        name=w.name,
        page=0,
        value=text,
        kind="text",
        bbox=_bbox(words) if words else _bbox(line.words),
        confidence=(sum(x.conf for x in words) / len(words)) if words else None,
    )


def _tok_eq(got: str, want: str) -> bool:
    if got == want:
        return True
    return len(want) >= 4 and SequenceMatcher(None, got, want).ratio() >= 0.8


def _glued_glyph(word: Word, want: str) -> bool:
    """OCR glued a mark glyph onto this word: "@yYes", "©O)No", "OcCane" for yes/no/cane."""
    got = _squash(word.text)
    return got != want and got.endswith(want) and 0 < len(got) - len(want) <= 3


def _starts_with_glyph(word: Word) -> bool:
    raw = word.text.strip()
    return bool(raw) and not raw[0].isalnum()


@dataclass(frozen=True)
class OptionHit:
    start: int  # index of the first label word
    end: int  # index after the last label word
    merged: bool  # the mark glyph is inside the first word's box


def _match_option(words: list[Word], start: int, label: str) -> OptionHit | None:
    """The words spelling `label`, searching from `start`; first match in reading order.

    OCR may merge adjacent tokens ("Nottested"), glue the mark glyph onto the first token
    ("@yYes"), or emit the glyph as its own word ("©"); all three are accepted. Glyph-only
    words are skipped between tokens.
    """
    toks = _norm(label).split()
    if not toks:
        return None
    for i in range(start, len(words)):
        j, k, merged = i, 0, False
        while k < len(toks) and j < len(words):
            got = _squash(words[j].text)
            if not got:
                if j == i:
                    break  # a match must start on a real word
                j += 1
                continue
            want, kk = "", k
            while kk < len(toks) and len(want) < len(got):
                want += toks[kk]
                kk += 1
            if _tok_eq(got, want):
                merged = merged or (j == i and _starts_with_glyph(words[j]))
            elif j == i and _glued_glyph(words[j], want):
                merged = True
            else:
                break
            k, j = kk, j + 1
        if k == len(toks):
            return OptionHit(i, j, merged)
    return None


def _mark_near(
    ctx: _PageCtx, words: list[Word], i: int, h: float, y0: int, y1: int, *, merged: bool
) -> tuple[float, BBox]:
    """Read the mark that belongs to the option/checkbox label whose first word is words[i].

    The mark is the ink blob just left of the label's letters. When OCR glued the mark's glyph
    onto the word (`merged`), or emitted it as a glyph-only word touching the label, it is the
    leftmost blob from the glyph's start instead.
    """
    word = words[i]
    prev = words[i - 1] if i > 0 and not _squash(words[i - 1].text) else None
    if merged:
        runs = ink_runs(ctx.image, (word.x0 - 2, y0, word.x1, y1))
        pick = runs[0] if runs else None
    elif prev is not None and prev.x1 >= word.x0 - 0.3 * h:
        runs = ink_runs(ctx.image, (prev.x0 - 2, y0, word.x1, y1))
        pick = runs[0] if runs else None
    else:
        runs = [
            r
            for r in ink_runs(ctx.image, (word.x0 - int(2.4 * h), y0, word.x0 - 1, y1))
            if r[1] - r[0] >= 0.35 * h
        ]
        pick = runs[-1] if runs else None
    if pick is None:
        return 0.0, (word.x0 - int(1.5 * h), y0, word.x0, y1)
    rows = [r for r in ink_rows(ctx.image, (pick[0], y0, pick[1], y1)) if r[1] - r[0] >= 0.35 * h]
    ry0, ry1 = (y0, y1)
    if rows:  # the blob nearest the label's centre line, so a neighbouring row's mark is ignored
        ry0, ry1 = min(rows, key=lambda r: abs((r[0] + r[1]) / 2 - word.cy))
        ry0, ry1 = ry0 - 1, ry1 + 1
    fill, tight = mark_fill(ctx.image, (pick[0] - 1, ry0, pick[1] + 1, ry1))
    return fill, tight or (pick[0], ry0, pick[1], ry1)


def _continuation_end(label: str, words: list[Word]) -> int:
    """Index just after the label's last token, when a wrapped label shares the line with
    its options; 0 when the line holds no label text."""
    toks = _norm(label).split()
    if not toks:
        return 0
    last = toks[-1]
    for i, w in enumerate(words[: max(1, len(words) - 1)]):
        if _squash(w.text).endswith(last) and len(_squash(w.text)) >= len(last):
            return i + 1
    return 0


def _read_radio(ctx: _PageCtx, a: _Anchor, w: Widget) -> FieldValue:
    line = ctx.lines[a.idx]
    h = line.h
    remaining = list(w.options)
    found: dict[str, tuple[float, BBox]] = {}  # value -> (fill, mark bbox)
    for j in range(a.idx, min(a.idx + 4, a.stop)):
        if not remaining:
            break
        x0 = _label_end(line, a.n_label) + int(0.4 * h) if j == a.idx else ctx.left_margin
        y0, y1 = ctx.band(j, 0.5 * h)
        words = _zone(ctx, (x0, y0, ctx.right_edge, y1), h=h) or []
        pos = _continuation_end(w.label, words) if j > a.idx else 0
        hits = 0
        for value, label in list(remaining):
            short = label if len(label) <= 22 else label[:21]
            hit = _match_option(words, pos, short)
            if hit is None:
                continue
            found[value] = _mark_near(ctx, words, hit.start, h, y0, y1, merged=hit.merged)
            remaining.remove((value, label))
            pos = hit.end
            hits += 1
        if j > a.idx and hits == 0:
            break
    if not found:
        return FieldValue(name=w.name, page=0, value="", kind="radio", bbox=_bbox(line.words))
    ranked = sorted(found.items(), key=lambda kv: kv[1][0], reverse=True)
    top_value, (top_fill, top_box) = ranked[0]
    second = ranked[1][1][0] if len(ranked) > 1 else 0.0
    selected = top_fill >= MARK_THRESHOLD and top_fill >= second + 0.1
    return FieldValue(
        name=w.name,
        page=0,
        value=top_value if selected else "",
        kind="radio",
        bbox=top_box,
        confidence=min(1.0, (top_fill - second) / 0.5) if selected else None,
    )


def _read_checkbox(ctx: _PageCtx, a: _Anchor, w: Widget) -> FieldValue:
    line = ctx.lines[a.idx]
    h = line.h
    y0, y1 = ctx.band(a.idx, 0.5 * h)
    first_tok = w.label.split()[0] if w.label.split() else w.label
    hit = _match_option(line.words, 0, first_tok)
    if hit is None:  # label unreadable; the mark still sits left of the first real word
        first = next((i for i, x in enumerate(line.words) if _squash(x.text)), 0)
        hit = OptionHit(first, first + 1, False)
    fill, box = _mark_near(ctx, line.words, hit.start, h, y0, y1, merged=hit.merged)
    return FieldValue(
        name=w.name,
        page=0,
        value="Yes" if fill >= MARK_THRESHOLD else "Off",
        kind="checkbox",
        bbox=box,
        confidence=min(1.0, abs(fill - MARK_THRESHOLD) / MARK_THRESHOLD),
    )


def _bbox(words: list[Word]) -> BBox:
    return (
        min(w.x0 for w in words),
        min(w.y0 for w in words),
        max(w.x1 for w in words),
        max(w.y1 for w in words),
    )


def _read(ctx: _PageCtx, a: _Anchor, w: Widget) -> FieldValue:
    if w.kind == "text":
        return _read_text(ctx, a, w)
    if w.kind == "textarea":
        return _read_textarea(ctx, a, w)
    if w.kind == "radio":
        return _read_radio(ctx, a, w)
    return _read_checkbox(ctx, a, w)


def _blank(w: Widget) -> FieldValue:
    return FieldValue(
        name=w.name, page=0, value="Off" if w.kind == "checkbox" else "", kind=_kind(w)
    )


def _recover_missing(ctx: _PageCtx, w: Widget, y_from: int, y_to: int) -> FieldValue | None:
    """A label the page pass dropped: OCR the strip between its neighbours' anchors on its
    own and, if the label is there, read the answer relative to it."""
    if y_to - y_from < ctx.pt(8):
        return None
    strip = _zone(ctx, (0, y_from, ctx.image.width, y_to), h=ctx.pt(9), mode="sparse")
    if not strip:
        return None
    lines = group_lines([x for x in strip if not _is_noise(x)])
    if not lines:
        return None
    local = _PageCtx(image=ctx.image, lines=lines, engine=ctx.engine, dpi=ctx.dpi)
    hits = align_labels([w.label], lines)
    if 0 not in hits:
        return None
    idx, n_label = hits[0]
    return _read(local, _Anchor(idx, n_label, len(lines)), w)


def recover_page(
    image: Image, widgets: list[Widget], engine: OcrEngine, dpi: int = 200
) -> dict[str, FieldValue]:
    """Recover every widget on one page from its image."""
    ctx = _PageCtx(image=image, lines=group_lines(engine.words(image)), engine=engine, dpi=dpi)
    anchors = align_labels([w.label for w in widgets], ctx.lines)
    out: dict[str, FieldValue] = {}
    for k, w in enumerate(widgets):
        if w.kind in ("heading", "note"):
            continue
        later = [anchors[i][0] for i in range(k + 1, len(widgets)) if i in anchors]
        stop = min(later) if later else len(ctx.lines)
        if k in anchors:
            idx, n_label = anchors[k]
            out[w.name] = _read(ctx, _Anchor(idx, n_label, stop), w)
            continue
        earlier = [anchors[i][0] for i in range(k) if i in anchors]
        y_from = ctx.lines[max(earlier)].y1 + 2 if earlier else 0
        y_to = ctx.lines[stop].y0 - 2 if stop < len(ctx.lines) else image.height
        out[w.name] = _recover_missing(ctx, w, y_from, y_to) or _blank(w)
    return out


def _kind(w: Widget) -> FieldKind:
    return "radio" if w.kind == "radio" else ("checkbox" if w.kind == "checkbox" else "text")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def rasterize(path: Path, dpi: int = 200) -> list[Image]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    scale: float = dpi / 72
    # pypdfium2 is untyped; `scale` is documented as a float (pyright infers int from the default)
    return [
        pdf[i].render(scale=scale).to_pil().convert("L")  # pyright: ignore[reportArgumentType]
        for i in range(len(pdf))
    ]


def ingest_scanned_pdf(
    path: Path,
    case_id: str | None = None,
    engine: OcrEngine | None = None,
    dpi: int = 200,
) -> IngestedDoc:
    engine = engine or TesseractEngine()
    images = rasterize(path, dpi)
    fields: dict[str, FieldValue] = {}
    for page_no, widgets in enumerate(PAGES, start=1):
        if page_no > len(images):
            break
        for name, fv in recover_page(images[page_no - 1], widgets, engine, dpi).items():
            fields[name] = fv.model_copy(update={"page": page_no})
    cid = case_id or re.sub(r"\.pdf$", "", path.name, flags=re.IGNORECASE)
    return IngestedDoc(
        case_id=cid,
        source_path=str(path),
        page_count=len(images),
        fields=fields,
        source_kind="ocr",
        dpi=dpi,
    )
