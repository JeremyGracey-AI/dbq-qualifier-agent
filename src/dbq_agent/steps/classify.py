"""Classify the form and route to candidate diagnostic codes.

Routing happens before retrieval so the KB is queried by DC rather than by free text. Codes the
form's findings point at but this slice cannot evaluate are reported, not silently ignored.
"""

from __future__ import annotations

from dbq_agent.models import Classification, IngestedDoc, KneeFindings, Span

KNEE_SIGNATURE_FIELDS = ("rom_flexion_initial", "rom_extension_initial", "test_lachman")
KNEE_ROM_DCS = ("5260", "5261")


def classify(doc: IngestedDoc, findings: KneeFindings, covered_dcs: set[str]) -> Classification:
    form_number = doc.get("form_number") or None
    is_knee = all(doc.has(f) for f in KNEE_SIGNATURE_FIELDS) or (form_number or "").startswith(
        "21-0960M-9"
    )
    if not is_knee:
        raise ValueError("unsupported form: only the knee DBQ structure is modeled in this slice")

    evidence: list[Span] = []
    if form_number:
        evidence.append(doc.span("form_number"))
    if doc.has("claimed_side"):
        evidence.append(doc.span("claimed_side"))

    candidates = [dc for dc in KNEE_ROM_DCS if dc in covered_dcs]
    not_evaluated: dict[str, str] = {}
    for dc in KNEE_ROM_DCS:
        if dc not in covered_dcs:
            not_evaluated[dc] = "no criteria rows in the KB"
    abnormal_stability = any(
        v not in ("Normal", "NotTested", "") for v in findings.instability.values()
    )
    if abnormal_stability or (doc.get("subluxation_history") or "None") != "None":
        not_evaluated["5257"] = (
            "instability/subluxation findings present; DC 5257 (revised 2021) not modeled"
        )
    if findings.arthritis_on_imaging:
        not_evaluated["5003"] = "arthritis on imaging; DC 5003 interaction noted but not rated"
    if findings.ankylosis:
        not_evaluated["5256"] = "ankylosis reported; DC 5256 not modeled"

    return Classification(
        form_type="knee_lower_leg",
        form_number=form_number,
        body_system="musculoskeletal",
        candidate_dcs=candidates,
        not_evaluated=not_evaluated,
        evidence=evidence,
    )
