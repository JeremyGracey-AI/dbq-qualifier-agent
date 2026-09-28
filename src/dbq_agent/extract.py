"""Extract typed findings from an ingested knee DBQ.

Two layers:

- structured: checkboxes, radios and numbers read straight from fields (deterministic, spans
  point at the field on its page). This is where the rating numbers come from.
- free text: remarks, flare descriptions, opinion rationale — read only in de-identified form,
  by either the heuristic extractor (regex, offline, default) or an LLM extractor. Each item
  carries a verbatim quote so the verify step can check it against the de-identified text.

Structured values always win over free-text values for the same fact.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Protocol

from dbq_agent.llm import LLMClient
from dbq_agent.models import (
    DeidPacket,
    FreeTextFindings,
    IngestedDoc,
    KneeFindings,
    OpinionFindings,
    RomSet,
    Span,
    TextClaim,
    TextClaimKind,
)
from dbq_agent.synth.form_spec import FUNCTIONAL_LOSS_FACTORS

# --------------------------------------------------------------------------------------
# Structured layer
# --------------------------------------------------------------------------------------


def _yn(doc: IngestedDoc, name: str) -> bool | None:
    v = doc.get(name)
    if v == "Yes":
        return True
    if v == "No":
        return False
    return None


def _int(doc: IngestedDoc, name: str) -> int | None:
    v = doc.get(name)
    if v is None or v.strip() == "":
        return None
    m = re.search(r"-?\d+", v)
    return int(m.group(0)) if m else None


def _text(doc: IngestedDoc, name: str) -> str | None:
    v = doc.get(name)
    return v.strip() if v and v.strip() else None


def _rom(doc: IngestedDoc, source: str, flex_field: str, ext_field: str) -> RomSet | None:
    flex, ext = _int(doc, flex_field), _int(doc, ext_field)
    if flex is None and ext is None:
        return None
    spans = [doc.span(f) for f, val in ((flex_field, flex), (ext_field, ext)) if val is not None]
    return RomSet(source=source, flexion=flex, extension=ext, spans=spans)  # type: ignore[arg-type]


def extract_structured(doc: IngestedDoc) -> KneeFindings:
    side_raw = (doc.get("claimed_side") or "").lower()
    side = side_raw if side_raw in ("right", "left") else None
    initial = _rom(doc, "initial", "rom_flexion_initial", "rom_extension_initial") or RomSet(
        source="initial"
    )
    factors = [label.lower() for name, label in FUNCTIONAL_LOSS_FACTORS if doc.get(name) == "Yes"]
    opinion: OpinionFindings | None = None
    if doc.has("opinion_requested"):
        requested = _yn(doc, "opinion_requested") or False
        opinion = OpinionFindings(
            requested=requested,
            conclusion=_text(doc, "opinion_conclusion"),
            rationale_text=_text(doc, "opinion_rationale"),
            spans=[doc.span("opinion_requested")]
            + ([doc.span("opinion_conclusion")] if _text(doc, "opinion_conclusion") else []),
        )
    bool_fields = [
        "flare_ups",
        "exam_during_flare",
        "pain_noted_on_exam",
        "pain_causes_functional_loss",
        "pain_weight_bearing",
        "pain_non_weight_bearing",
        "pain_passive_rom",
        "opposite_joint_undamaged",
        "rep_use_performed",
        "ankylosis",
        "imaging_arthritis",
        "flare_functional_loss",
    ]
    spans = {f: doc.span(f) for f in bool_fields if doc.has(f)}
    spans.update(
        {name: doc.span(name) for name, _ in FUNCTIONAL_LOSS_FACTORS if doc.get(name) == "Yes"}
    )
    spans.update(
        {
            k: doc.span(k)
            for k in ("test_lachman", "test_posterior_drawer", "test_medial", "test_lateral")
            if doc.get(k)
        }
    )
    for f in (
        "passive_not_tested_reason",
        "rep_use_not_performed_reason",
        "flare_no_estimate_reason",
    ):
        if _text(doc, f):
            spans[f] = doc.span(f)

    return KneeFindings(
        side=side,  # type: ignore[arg-type]
        diagnosis=_text(doc, "dx_1"),
        initial=initial,
        rep_use_performed=_yn(doc, "rep_use_performed"),
        rep_use_not_performed_reason=_text(doc, "rep_use_not_performed_reason"),
        post_rep=_rom(doc, "post_rep", "rom_flexion_post_rep", "rom_extension_post_rep"),
        flare_ups_reported=_yn(doc, "flare_ups"),
        exam_during_flare=_yn(doc, "exam_during_flare"),
        flare_estimate=_rom(doc, "flare_estimate", "flare_flexion_est", "flare_extension_est"),
        flare_no_estimate_reason=_text(doc, "flare_no_estimate_reason"),
        pain_on_motion=_yn(doc, "pain_noted_on_exam"),
        pain_causes_functional_loss=_yn(doc, "pain_causes_functional_loss"),
        pain_weight_bearing=_yn(doc, "pain_weight_bearing"),
        pain_non_weight_bearing=_yn(doc, "pain_non_weight_bearing"),
        pain_passive=_yn(doc, "pain_passive_rom"),
        passive_not_tested_reason=_text(doc, "passive_not_tested_reason"),
        opposite_joint_undamaged=_yn(doc, "opposite_joint_undamaged"),
        opposite_joint_measured=_int(doc, "opposite_flexion") is not None,
        functional_loss_factors=factors,
        ankylosis=_yn(doc, "ankylosis"),
        arthritis_on_imaging=_yn(doc, "imaging_arthritis"),
        instability={
            k: v
            for k in ("test_lachman", "test_posterior_drawer", "test_medial", "test_lateral")
            if (v := doc.get(k))
        },
        opinion=opinion,
        spans=spans,
    )


# --------------------------------------------------------------------------------------
# Free-text layer
# --------------------------------------------------------------------------------------

REASONING_CUES = (
    "because",
    "due to",
    "consistent with",
    "based on",
    "records",
    "str",
    "service treatment",
    "documented",
    "in-service",
    "onset",
    "continuity",
    "mechanism",
    "literature",
    "history of",
    "as a result",
    "secondary to",
    "reason",
    "unable to measure",
    "could not be",
    "refused",
    "declined",
    "safety",
    "contraindicated",
)

_DEG = r"(\d{1,3})\s*(?:degrees|deg|°)"


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;])\s+", text) if s.strip()]


def _has_cue(text: str) -> bool:
    t = text.lower()
    return any(c in t for c in REASONING_CUES)


def _page_of(doc: IngestedDoc, fld: str) -> int:
    return doc.fields[fld].page if fld in doc.fields else 1


class TextExtractor(Protocol):
    name: str

    def extract(self, deid: DeidPacket, doc: IngestedDoc) -> FreeTextFindings: ...


@dataclass
class HeuristicTextExtractor:
    """Regex-based, offline. Good enough for the slice; the LLM extractor is a drop-in."""

    name: str = "heuristic"

    def extract(self, deid: DeidPacket, doc: IngestedDoc) -> FreeTextFindings:
        items: list[TextClaim] = []
        ft = deid.free_text

        def span(fld: str, quote: str) -> Span:
            return Span(field=fld, page=_page_of(doc, fld), text=quote, deid=True)

        # flare-up estimates expressed in prose
        for fld in ("remarks", "flare_ups_desc", "functional_loss_desc", "functional_impact"):
            for sent in _sentences(ft.get(fld, "")):
                low = sent.lower()
                if "flare" not in low:
                    continue
                m = re.search(r"flexion[^.;]*?(?:limited to|to|of)\s*" + _DEG, low)
                if m:
                    items.append(
                        TextClaim(
                            kind="flare_flexion_estimate",
                            value=int(m.group(1)),
                            span=span(fld, sent),
                        )
                    )
                m = re.search(r"extension[^.;]*?(?:limited to|to|of)\s*" + _DEG, low)
                if m:
                    items.append(
                        TextClaim(
                            kind="flare_extension_estimate",
                            value=int(m.group(1)),
                            span=span(fld, sent),
                        )
                    )

        # reasons and speculation
        for fld, reason_kind in (
            ("flare_no_estimate_reason", "flare_no_estimate_reason"),
            ("rep_use_not_performed_reason", "rep_use_not_performed_reason"),
        ):
            text = ft.get(fld, "").strip()
            if not text:
                continue
            if "speculat" in text.lower() and not _has_cue(text):
                items.append(
                    TextClaim(kind="speculation_unexplained", value=True, span=span(fld, text))
                )
            elif _has_cue(text):
                items.append(TextClaim(kind=reason_kind, value=text, span=span(fld, text)))  # type: ignore[arg-type]

        rationale = ft.get("opinion_rationale", "").strip()
        if rationale:
            words = len(rationale.split())
            present = words >= 20 and _has_cue(rationale)
            quote = rationale if len(rationale) <= 160 else rationale[:160]
            items.append(
                TextClaim(
                    kind="rationale_present", value=present, span=span("opinion_rationale", quote)
                )
            )
            if "speculat" in rationale.lower() and not _has_cue(rationale):
                items.append(
                    TextClaim(
                        kind="speculation_unexplained",
                        value=True,
                        span=span("opinion_rationale", quote),
                    )
                )

        # functional-loss factors mentioned in prose
        remarks = ft.get("remarks", "")
        for sent in _sentences(remarks):
            low = sent.lower()
            for factor in ("weakness", "fatigability", "incoordination"):
                if factor in low:
                    items.append(
                        TextClaim(
                            kind="functional_loss_factor", value=factor, span=span("remarks", sent)
                        )
                    )
        return FreeTextFindings(items=items, extractor=self.name)


_LLM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": [
                            "flare_flexion_estimate",
                            "flare_extension_estimate",
                            "flare_no_estimate_reason",
                            "rep_use_not_performed_reason",
                            "rationale_present",
                            "speculation_unexplained",
                            "functional_loss_factor",
                        ],
                    },
                    "value": {"type": ["string", "integer", "boolean", "null"]},
                    "field": {"type": "string"},
                    "quote": {"type": "string", "description": "verbatim from the field text"},
                },
                "required": ["kind", "field", "quote"],
            },
        }
    },
    "required": ["items"],
}

_LLM_SYSTEM = """You read de-identified free-text fields from a VA knee DBQ and record findings.
Rules: quote verbatim; never infer numbers that are not written; tokens like [NAME_1] are
placeholders and are not findings. Record:
- flare_flexion_estimate / flare_extension_estimate: degrees the examiner estimates during flare-ups
- flare_no_estimate_reason / rep_use_not_performed_reason: a stated reason (value = the reason)
- rationale_present: true if the opinion rationale contains actual reasoning, false if conclusory
- speculation_unexplained: true if the examiner declines to estimate/opine as speculative without saying why.
  "Cannot be determined without resorting to speculation" on its own is NOT a reason; record it
  here, not as a *_reason item, unless the examiner also explains why (missing records, refusal, safety...)
- functional_loss_factor: weakness, fatigability or incoordination described in prose
Do not rate, do not cite law, do not add anything not in the text."""


@dataclass
class LLMTextExtractor:
    client: LLMClient
    fallback: HeuristicTextExtractor | None = None
    name: str = field(init=False, default="llm")

    def __post_init__(self) -> None:
        self.name = f"llm:{self.client.name}"

    def extract(self, deid: DeidPacket, doc: IngestedDoc) -> FreeTextFindings:
        user = "\n\n".join(
            f"[{fld}]\n{text}" for fld, text in deid.free_text.items() if text.strip()
        )
        try:
            raw = self.client.extract_json(_LLM_SYSTEM, user, _LLM_SCHEMA)
        except Exception as exc:  # network/model failure -> stay offline, but say so
            if self.fallback is None:
                raise
            reason = type(exc).__name__
            warnings.warn(
                f"{self.name} failed ({reason}: {exc}); falling back to {self.fallback.name}",
                RuntimeWarning,
                stacklevel=2,
            )
            out = self.fallback.extract(deid, doc)
            out.extractor = f"{self.name}->fallback:{self.fallback.name} ({reason})"
            return out
        items: list[TextClaim] = []
        valid_kinds = set(TextClaimKind.__args__)  # type: ignore[attr-defined]
        for it in raw.get("items", []) or []:
            kind, fld, quote = it.get("kind"), it.get("field"), it.get("quote", "")
            if kind not in valid_kinds or fld not in deid.free_text:
                continue
            if not quote or quote not in deid.free_text[fld]:
                continue  # missing or hallucinated quote; verify would drop it anyway
            value = normalize_value(kind, it.get("value"), quote)
            if value is None and kind not in (
                "flare_no_estimate_reason",
                "rep_use_not_performed_reason",
            ):
                continue  # e.g. "extension is not additionally limited" carries no number
            if kind == "speculation_unexplained" and value is False:
                continue  # a negative finding is not evidence of anything
            if (
                kind in ("flare_no_estimate_reason", "rep_use_not_performed_reason")
                and isinstance(value, str)
                and "speculat" in value.lower()
                and not _has_cue(value)
            ):
                # Jones v. Shinseki: "cannot say without speculation" is not a reason unless
                # the examiner explains why. Rules decide this, not the model.
                kind, value = "speculation_unexplained", True
            items.append(
                TextClaim(
                    kind=kind,
                    value=value,
                    span=Span(field=fld, page=_page_of(doc, fld), text=quote, deid=True),
                )
            )
        return FreeTextFindings(items=items, extractor=self.name)


_INT = re.compile(r"-?\d{1,3}")


def normalize_value(kind: str, value: Any, quote: str) -> str | int | bool | None:
    """Coerce a model-supplied value into the type the evaluator expects for `kind`.

    Models return "30 degrees" where the rules need 30, "true" where they need True, and
    prose where a factor name is expected. Anything that cannot be coerced becomes None so the
    caller drops the item instead of carrying an unusable value into evaluation.
    """
    if kind in ("flare_flexion_estimate", "flare_extension_estimate"):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str) and (m := _INT.search(value)):
            return int(m.group(0))
        # no usable value: take the number written with a degree unit in the quote, if any
        if m := re.search(_DEG, quote.lower()):
            return int(m.group(1))
        return None
    if kind in ("rationale_present", "speculation_unexplained"):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "yes", "1")
        return None
    if kind == "functional_loss_factor":
        text = f"{value or ''} {quote}".lower()
        for factor in ("weakness", "fatigability", "incoordination"):
            if factor in text:
                return factor
        return None
    if kind in ("flare_no_estimate_reason", "rep_use_not_performed_reason"):
        return str(value) if value else quote
    return value if isinstance(value, str | int | bool) else None


def extract(
    doc: IngestedDoc, deid: DeidPacket, text_extractor: TextExtractor | None = None
) -> KneeFindings:
    findings = extract_structured(doc)
    extractor = text_extractor or HeuristicTextExtractor()
    findings.free_text = extractor.extract(deid, doc)
    return findings
