"""Typed state shared by every pipeline step.

Design rule: nothing moves between steps as a bare dict. Every value that can end up in the
report carries a `Span` (where it came from) so the verify step can check it against the source.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------

FieldKind = Literal["text", "checkbox", "radio"]


class Span(BaseModel):
    """Where a value came from: a form field on a page, and the verbatim text relied on."""

    field: str
    page: int = Field(ge=1, description="1-based page number")
    text: str = Field(description="verbatim text from the field value (or its de-identified view)")
    deid: bool = Field(
        default=False,
        description="True when `text` quotes the de-identified view of a free-text field",
    )


class FieldValue(BaseModel):
    name: str
    page: int
    value: str
    kind: FieldKind


class IngestedDoc(BaseModel):
    case_id: str
    source_path: str
    page_count: int
    fields: dict[str, FieldValue]

    def get(self, name: str) -> str | None:
        fv = self.fields.get(name)
        if fv is None:
            return None
        return fv.value

    def has(self, name: str) -> bool:
        return name in self.fields

    def span(self, name: str, text: str | None = None) -> Span:
        fv = self.fields[name]
        return Span(field=name, page=fv.page, text=fv.value if text is None else text)


# --------------------------------------------------------------------------------------
# PHI boundary
# --------------------------------------------------------------------------------------


class Identity(BaseModel):
    """Identity fields. Stays inside the PHI boundary; never serialized into LLM payloads."""

    name: str | None = None
    dob: str | None = None
    ssn: str | None = None
    exam_date: str | None = None
    examiner: str | None = None


class DeidPacket(BaseModel):
    """The only view of free text that may leave the boundary."""

    free_text: dict[str, str] = Field(description="field name -> redacted text")
    token_map: dict[str, str] = Field(
        default_factory=dict, description="token -> original value (inside boundary only)"
    )
    redactors: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# Knowledge base
# --------------------------------------------------------------------------------------

Measure = Literal["flexion", "extension"]
Op = Literal["<=", ">="]


CriterionKind = Literal["rom_threshold", "predicate"]


class Criterion(BaseModel):
    """One rating level of one diagnostic code, in one version of the schedule.

    - `rom_threshold`: a degree threshold on a measure (DC 5260/5261 style).
    - `predicate`: a list of alternatives (OR); each alternative maps an instability finding
      to the values that satisfy it (AND). Used for the 2021 DC 5257 text, which keys on tear
      status, persistent instability and prescriptions rather than degrees.
    """

    id: str
    dc: str
    dc_title: str
    pct: int
    kind: CriterionKind = "rom_threshold"
    measure: Measure | None = None
    op: Op | None = None
    threshold_deg: int | None = None
    requires: list[dict[str, list[str]]] | None = None
    subtable: str | None = Field(default=None, description="e.g. 'ligament' or 'patellar'")
    text: str = Field(description="criterion text as printed in the schedule")
    cite: str
    effective_from: date
    effective_to: date | None = None

    @model_validator(mode="after")
    def _shape(self) -> Criterion:
        if self.kind == "rom_threshold" and (
            self.measure is None or self.op is None or self.threshold_deg is None
        ):
            raise ValueError(f"{self.id}: rom_threshold needs measure, op and threshold_deg")
        if self.kind == "predicate" and not self.requires:
            raise ValueError(f"{self.id}: predicate needs at least one alternative in requires")
        return self


class Authority(BaseModel):
    id: str
    kind: Literal["cfr", "rule", "case", "gc_opinion", "manual"]
    cite: str
    title: str
    summary: str
    tags: list[str] = Field(default_factory=list)
    verify: bool = Field(default=False, description="citation still needs manual verification")


Severity = Literal["blocking", "advisory"]


class AdequacyRule(BaseModel):
    id: str
    severity: Severity
    description: str
    authorities: list[str]
    tags: list[str] = Field(default_factory=list)


class RetrievedContext(BaseModel):
    criteria: list[Criterion] = Field(default_factory=list)
    authorities: list[Authority] = Field(default_factory=list)
    rules: list[AdequacyRule] = Field(default_factory=list)
    query: dict[str, str | list[str]] = Field(default_factory=dict)
    iteration: int = 0

    def authority_ids(self) -> set[str]:
        return {a.id for a in self.authorities}


# --------------------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------------------

RomSource = Literal["initial", "post_rep", "flare_estimate"]


class RomSet(BaseModel):
    source: RomSource
    flexion: int | None = None
    extension: int | None = None
    spans: list[Span] = Field(default_factory=list)


class OpinionFindings(BaseModel):
    requested: bool
    conclusion: str | None = None
    rationale_text: str | None = None
    rationale_present: bool | None = None
    speculation_unexplained: bool | None = None
    spans: list[Span] = Field(default_factory=list)


TextClaimKind = Literal[
    "flare_flexion_estimate",
    "flare_extension_estimate",
    "flare_no_estimate_reason",
    "rep_use_not_performed_reason",
    "rationale_present",
    "speculation_unexplained",
    "functional_loss_factor",
]


class TextClaim(BaseModel):
    """One thing an extractor (heuristic or LLM) found in de-identified free text."""

    kind: TextClaimKind
    value: str | int | bool | None = None
    span: Span


class FreeTextFindings(BaseModel):
    items: list[TextClaim] = Field(default_factory=list)
    extractor: str = "none"


class InstabilityFindings(BaseModel):
    """What the 2021 DC 5257 criteria turn on, as the examiner documented it.

    Values are the form's radio tokens; `facts()` renders them as the strings the KB
    predicates compare against. `None` means the examiner left the question blank.
    """

    ligament_injury: str | None = None  # None | Sprain | IncompleteTear | CompleteTear
    ligament_repair_status: str | None = None  # NA | Repaired | Unrepaired | FailedRepair
    persistent_instability: bool | None = None
    rx_bracing: bool | None = None
    rx_assistive_device: str | None = None  # None | Cane | Crutches | Walker
    patellar_instability: bool | None = None
    patellar_surgical_repair: bool | None = None
    subluxation_history: str | None = None  # None | Slight | Moderate | Severe
    lateral_instability_history: str | None = None  # None | Slight | Moderate | Severe
    spans: dict[str, Span] = Field(default_factory=dict)

    def any_instability(self) -> bool:
        return bool(
            (self.ligament_injury and self.ligament_injury != "None")
            or self.patellar_instability
            or (self.subluxation_history and self.subluxation_history != "None")
            or (self.lateral_instability_history and self.lateral_instability_history != "None")
        )

    def severity(self) -> str | None:
        """Pre-2021 characterisation: the worse of the two history fields."""
        order = ["None", "Slight", "Moderate", "Severe"]
        vals = [
            v for v in (self.subluxation_history, self.lateral_instability_history) if v in order
        ]
        if not vals:
            return None
        worst = max(vals, key=order.index)
        return None if worst == "None" else worst.lower()

    def facts(self) -> dict[str, str]:
        """Flat string facts for predicate criteria. Blank answers are simply absent."""
        out: dict[str, str] = {}
        if self.ligament_injury:
            out["ligament_injury"] = self.ligament_injury
        if self.ligament_repair_status:
            out["ligament_repair_status"] = self.ligament_repair_status
        if self.persistent_instability is not None:
            out["persistent_instability"] = "yes" if self.persistent_instability else "no"
        if self.rx_bracing is not None:
            out["rx_bracing"] = "yes" if self.rx_bracing else "no"
        if self.rx_assistive_device:
            out["rx_assistive_device"] = self.rx_assistive_device
        if self.patellar_instability is not None:
            out["patellar_instability"] = "yes" if self.patellar_instability else "no"
        if self.patellar_surgical_repair is not None:
            out["patellar_surgical_repair"] = "yes" if self.patellar_surgical_repair else "no"
        sev = self.severity()
        if sev:
            out["severity"] = sev
        return out


class KneeFindings(BaseModel):
    side: Literal["right", "left"] | None = None
    diagnosis: str | None = None
    initial: RomSet
    rep_use_performed: bool | None = None
    rep_use_not_performed_reason: str | None = None
    post_rep: RomSet | None = None
    flare_ups_reported: bool | None = None
    exam_during_flare: bool | None = None
    flare_estimate: RomSet | None = None
    flare_no_estimate_reason: str | None = None
    pain_on_motion: bool | None = None
    pain_causes_functional_loss: bool | None = None
    pain_weight_bearing: bool | None = None
    pain_non_weight_bearing: bool | None = None
    pain_passive: bool | None = None
    passive_not_tested_reason: str | None = None
    opposite_joint_undamaged: bool | None = None
    opposite_joint_measured: bool | None = None
    functional_loss_factors: list[str] = Field(default_factory=list)
    ankylosis: bool | None = None
    arthritis_on_imaging: bool | None = None
    instability: dict[str, str] = Field(default_factory=dict)
    instability_findings: InstabilityFindings | None = None
    opinion: OpinionFindings | None = None
    free_text: FreeTextFindings = Field(default_factory=FreeTextFindings)
    spans: dict[str, Span] = Field(default_factory=dict, description="field -> span for booleans")


# --------------------------------------------------------------------------------------
# Claims, gaps, evaluation
# --------------------------------------------------------------------------------------

ClaimKind = Literal["rating_tier", "painful_motion_minimum", "consider_higher", "gap", "note"]
ClaimStatus = Literal["pending", "verified", "dropped"]


class Claim(BaseModel):
    id: str
    kind: ClaimKind
    statement: str
    dc: str | None = None
    pct: int | None = None
    severity: Severity | None = None
    rule_id: str | None = None
    evidence: list[Span] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    status: ClaimStatus = "pending"
    drop_reason: str | None = None


class Classification(BaseModel):
    form_type: str
    form_number: str | None = None
    body_system: str
    candidate_dcs: list[str]
    not_evaluated: dict[str, str] = Field(
        default_factory=dict, description="dc -> reason it is out of scope for this slice"
    )
    evidence: list[Span] = Field(default_factory=list)


class Evaluation(BaseModel):
    claims: list[Claim] = Field(default_factory=list)
    ratings: dict[str, int] = Field(default_factory=dict, description="dc -> pct (provisional)")


class VerificationResult(BaseModel):
    verified: int = 0
    dropped: list[Claim] = Field(default_factory=list)
    unresolved_citations: list[str] = Field(default_factory=list)
    iterations: int = 0


# --------------------------------------------------------------------------------------
# Pipeline state
# --------------------------------------------------------------------------------------


class PipelineMeta(BaseModel):
    steps: list[str] = Field(default_factory=list)
    text_extractor: str = "none"
    redactors: list[str] = Field(default_factory=list)
    retrieval_iterations: int = 0
    kb_version: str = ""


class State(BaseModel):
    doc: IngestedDoc
    claim_date: date
    identity: Identity | None = None
    deid: DeidPacket | None = None
    classification: Classification | None = None
    findings: KneeFindings | None = None
    retrieved: RetrievedContext | None = None
    evaluation: Evaluation | None = None
    gaps: list[Claim] = Field(default_factory=list)
    notes: list[Claim] = Field(default_factory=list)
    verification: VerificationResult | None = None
    meta: PipelineMeta = Field(default_factory=PipelineMeta)

    def all_claims(self) -> list[Claim]:
        out: list[Claim] = []
        if self.evaluation:
            out.extend(self.evaluation.claims)
        out.extend(self.gaps)
        out.extend(self.notes)
        return out


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


class RatingLine(BaseModel):
    dc: str
    dc_title: str
    pct: int
    basis: str
    evidence: list[Span]
    citations: list[str]


class QualifierReport(BaseModel):
    case_id: str
    form_type: str
    form_number: str | None
    claim_date: date
    adequate: bool
    rating: dict[str, RatingLine] | None = Field(
        description="None when a blocking adequacy gap exists; see `provisional`"
    )
    provisional: dict[str, RatingLine]
    gaps: list[Claim]
    notes: list[Claim]
    consider: list[Claim]
    dropped_claims: list[Claim]
    veteran: Identity | None = Field(default=None, description="re-identified; inside boundary")
    pipeline: PipelineMeta
    disclaimer: str = (
        "Decision support only. Ratings are provisional mappings of DBQ findings to 38 CFR "
        "Part 4 criteria for a human reviewer (VSO or attorney). Not a VA rating decision."
    )
