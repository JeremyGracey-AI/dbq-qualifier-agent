"""PHI boundary.

Two mechanisms, both enforced by tests:

1. Allowlist. Only `FREE_TEXT_FIELDS` may ever be serialized toward a model, and only through
   `Deidentifier.deidentify()`. `IDENTITY_FIELDS` are read into `Identity` and stay in-process.
2. Redaction of the free text itself, because a remark can say "Mr. Smith, DOB 3/14/75, ...".
   Three detectors are merged into one span set and replaced with numbered tokens:
   - known values: the identity fields we already parsed (name variants, DOB in several formats)
   - patterns: SSN, phone, e-mail, dates
   - phi-scrub (Rust engine, optional extra): its own detections, when the package is installed

The token map (token -> original) never leaves the boundary; `reidentify()` restores tokens in
the final report for the human reviewer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from dbq_agent.models import DeidPacket, Identity, IngestedDoc
from dbq_agent.synth.form_spec import FREE_TEXT_FIELDS, IDENTITY_FIELDS

__all__ = [
    "FREE_TEXT_FIELDS",
    "IDENTITY_FIELDS",
    "Deidentifier",
    "Detector",
    "Finding",
    "KnownValueDetector",
    "PatternDetector",
    "PhiScrubDetector",
    "read_identity",
    "reidentify",
]


@dataclass(frozen=True)
class Finding:
    start: int
    end: int
    category: str  # NAME, DOB, SSN, PHONE, EMAIL, DATE, ...


class Detector(Protocol):
    name: str

    def detect(self, text: str) -> list[Finding]: ...


# --------------------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------------------

_MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]


def _parse_date(s: str) -> datetime | None:
    for fmt in ("%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%m/%d/%y"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def _date_variants(s: str) -> list[str]:
    d = _parse_date(s)
    if d is None:
        return [s]
    out = {
        s,
        d.strftime("%m/%d/%Y"),
        d.strftime("%-m/%-d/%Y"),
        d.strftime("%m-%d-%Y"),
        d.strftime("%Y-%m-%d"),
        d.strftime("%B %d, %Y"),
        d.strftime("%B %-d, %Y"),
        d.strftime("%b %d, %Y"),
        d.strftime("%m/%d/%y"),
        d.strftime("%-m/%-d/%y"),
    }
    return sorted(out, key=len, reverse=True)


_HONORIFICS = {"dr", "mr", "ms", "mrs", "md", "do", "phd", "np", "pa", "jr", "sr", "ii", "iii"}


def _name_variants(full: str) -> list[str]:
    parts = [p.strip(".,;:") for p in re.split(r"\s+", full.strip()) if p]
    parts = [
        p for p in parts if p and not re.fullmatch(r"[A-Z]", p) and p.lower() not in _HONORIFICS
    ]
    out = {full.strip()}
    if not parts:
        return sorted((v for v in out if len(v) >= 3), key=len, reverse=True)
    last = parts[-1]
    honorific_forms = ("Mr.", "Ms.", "Mrs.", "Dr.", "Mr", "Ms", "Mrs", "Dr")
    out.update({last, *(f"{h} {last}" for h in honorific_forms)})
    if len(parts) >= 2:
        first = parts[0]
        out.update({f"{first} {last}", first, " ".join(parts)})
    return sorted((v for v in out if len(v) >= 3), key=len, reverse=True)


@dataclass
class KnownValueDetector:
    """Finds the identity values we already know, in any common variant."""

    identity: Identity
    name: str = "known-values"

    def detect(self, text: str) -> list[Finding]:
        targets: list[tuple[str, str]] = []
        if self.identity.name:
            targets += [(v, "NAME") for v in _name_variants(self.identity.name)]
        if self.identity.examiner:
            targets += [(v, "NAME") for v in _name_variants(self.identity.examiner)]
        if self.identity.dob:
            targets += [(v, "DOB") for v in _date_variants(self.identity.dob)]
        if self.identity.ssn:
            targets += [(self.identity.ssn, "SSN"), (self.identity.ssn.replace("-", ""), "SSN")]
        found: list[Finding] = []
        for value, cat in targets:
            for m in re.finditer(rf"(?<!\w){re.escape(value)}(?!\w)", text):
                found.append(Finding(m.start(), m.end(), cat))
        return found


_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"(?:\(\d{3}\)\s?|\b\d{3}[-.\s])\d{3}[-.\s]\d{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("DATE", re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")),
    (
        "DATE",
        re.compile(
            r"\b(?:" + "|".join(m.capitalize() for m in _MONTHS) + r")\s+\d{1,2},\s+\d{4}\b"
        ),
    ),
]


@dataclass
class PatternDetector:
    name: str = "patterns"

    def detect(self, text: str) -> list[Finding]:
        return [
            Finding(m.start(), m.end(), cat) for cat, pat in _PATTERNS for m in pat.finditer(text)
        ]


@dataclass
class PhiScrubDetector:
    """Adapter for Jeremy's `phi-scrub` Rust engine (optional extra `phi`)."""

    name: str = "phi-scrub"
    _scrubber: object | None = field(default=None, repr=False)

    @classmethod
    def available(cls) -> bool:
        try:
            import phi_scrub  # noqa: F401
        except ImportError:
            return False
        return True

    def detect(self, text: str) -> list[Finding]:
        if self._scrubber is None:
            import phi_scrub

            self._scrubber = phi_scrub.Scrubber()  # type: ignore[attr-defined]
        scrubber = self._scrubber
        return [
            Finding(int(f.start), int(f.end), str(f.category).upper())
            for f in scrubber.detect(text)  # type: ignore[attr-defined]
        ]


# --------------------------------------------------------------------------------------
# Deidentifier
# --------------------------------------------------------------------------------------


def read_identity(doc: IngestedDoc) -> Identity:
    return Identity(
        name=doc.get("vet_name") or None,
        dob=doc.get("vet_dob") or None,
        ssn=doc.get("vet_ssn") or None,
        exam_date=doc.get("exam_date") or None,
        examiner=doc.get("examiner_name") or None,
    )


def _merge(findings: list[Finding]) -> list[Finding]:
    """Merge overlapping findings; the longer (earlier-starting) span wins."""
    ordered = sorted(findings, key=lambda f: (f.start, -(f.end - f.start)))
    merged: list[Finding] = []
    for f in ordered:
        if merged and f.start < merged[-1].end:
            if f.end > merged[-1].end:
                merged[-1] = Finding(merged[-1].start, f.end, merged[-1].category)
            continue
        merged.append(f)
    return merged


@dataclass
class Deidentifier:
    detectors: list[Detector]

    @classmethod
    def default(cls, identity: Identity) -> Deidentifier:
        dets: list[Detector] = [KnownValueDetector(identity), PatternDetector()]
        if PhiScrubDetector.available():
            dets.append(PhiScrubDetector())
        return cls(dets)

    def deidentify(self, doc: IngestedDoc) -> DeidPacket:
        token_map: dict[str, str] = {}
        value_to_token: dict[tuple[str, str], str] = {}
        counters: dict[str, int] = {}
        out: dict[str, str] = {}
        for name in FREE_TEXT_FIELDS:
            text = doc.get(name)
            if text is None:
                continue
            findings = _merge([f for d in self.detectors for f in d.detect(text)])
            pieces: list[str] = []
            cursor = 0
            for f in findings:
                original = text[f.start : f.end]
                key = (f.category, original)
                token = value_to_token.get(key)
                if token is None:
                    counters[f.category] = counters.get(f.category, 0) + 1
                    token = f"[{f.category}_{counters[f.category]}]"
                    value_to_token[key] = token
                    token_map[token] = original
                pieces.append(text[cursor : f.start])
                pieces.append(token)
                cursor = f.end
            pieces.append(text[cursor:])
            out[name] = "".join(pieces)
        return DeidPacket(
            free_text=out, token_map=token_map, redactors=[d.name for d in self.detectors]
        )


_TOKEN = re.compile(r"\[[A-Z]+_\d+\]")


def reidentify(text: str, token_map: dict[str, str]) -> str:
    return _TOKEN.sub(lambda m: token_map.get(m.group(0), m.group(0)), text)


def assert_no_phi(payload: str, identity: Identity) -> None:
    """Raise if any known identity value (or its variants) appears in `payload`."""
    leaks: list[str] = []
    checks: list[str] = []
    if identity.name:
        checks += [v for v in _name_variants(identity.name) if len(v) > 3]
    if identity.dob:
        checks += _date_variants(identity.dob)
    if identity.ssn:
        checks += [identity.ssn, identity.ssn.replace("-", "")]
    for v in checks:
        if re.search(rf"(?<!\w){re.escape(v)}(?!\w)", payload):
            leaks.append(v)
    if leaks:
        raise AssertionError(f"PHI leaked past the boundary: {sorted(set(leaks))}")
