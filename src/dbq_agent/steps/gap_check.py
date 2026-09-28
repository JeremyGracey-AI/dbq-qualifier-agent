"""Exam-adequacy gap check.

Each predicate is keyed by the rule id in data/kb/adequacy.json. A rule is only evaluated when
retrieval returned it, and the resulting gap claim cites the rule's authorities, so the verify
step can check them like any other claim. Blocking gaps withhold the final rating; advisory gaps
are surfaced for the reviewer.
"""

from __future__ import annotations

from collections.abc import Callable

from dbq_agent.models import AdequacyRule, Claim, KneeFindings, RetrievedContext, Span


def _spans(findings: KneeFindings, *keys: str) -> list[Span]:
    return [findings.spans[k] for k in keys if k in findings.spans]


def _text_claim(findings: KneeFindings, kind: str) -> Span | None:
    for i in findings.free_text.items:
        if i.kind == kind:
            return i.span
    return None


def _has_text_estimate(findings: KneeFindings) -> bool:
    return any(
        i.kind in ("flare_flexion_estimate", "flare_extension_estimate")
        and isinstance(i.value, int)
        for i in findings.free_text.items
    )


# Each predicate returns the evidence spans when the gap applies, else None.
Predicate = Callable[[KneeFindings], list[Span] | None]


def correia_passive(f: KneeFindings) -> list[Span] | None:
    if f.pain_passive is None and not f.passive_not_tested_reason:
        return _spans(f, "pain_passive_rom") or list(f.initial.spans)
    return None


def correia_weight_bearing(f: KneeFindings) -> list[Span] | None:
    if f.pain_weight_bearing is None or f.pain_non_weight_bearing is None:
        return _spans(f, "pain_weight_bearing", "pain_non_weight_bearing") or list(f.initial.spans)
    return None


def correia_opposite_joint(f: KneeFindings) -> list[Span] | None:
    if f.opposite_joint_undamaged and not f.opposite_joint_measured:
        return _spans(f, "opposite_joint_undamaged")
    return None


def deluca_rep_use(f: KneeFindings) -> list[Span] | None:
    if f.rep_use_performed:
        return None
    reason = f.rep_use_not_performed_reason or _text_claim(f, "rep_use_not_performed_reason")
    if reason:
        return None
    return _spans(f, "rep_use_performed") or list(f.initial.spans)


def sharp_flare_estimate(f: KneeFindings) -> list[Span] | None:
    if not f.flare_ups_reported or f.exam_during_flare:
        return None
    if f.flare_estimate is not None or _has_text_estimate(f):
        return None
    explained = _text_claim(f, "flare_no_estimate_reason") is not None
    if explained:
        return None
    spans = _spans(f, "flare_ups", "flare_functional_loss", "flare_no_estimate_reason")
    spec = _text_claim(f, "speculation_unexplained")
    if spec is not None and spec.field == "flare_no_estimate_reason":
        spans.append(spec)
    return spans


def mitchell_pain_function(f: KneeFindings) -> list[Span] | None:
    if f.pain_on_motion and f.pain_causes_functional_loss is None:
        return _spans(f, "pain_noted_on_exam", "pain_causes_functional_loss")
    return None


def opinion_rationale(f: KneeFindings) -> list[Span] | None:
    op = f.opinion
    if op is None or not op.requested or not op.conclusion:
        return None
    rationale_ok = any(i.kind == "rationale_present" and i.value is True for i in f.free_text.items)
    spec = next(
        (
            i.span
            for i in f.free_text.items
            if i.kind == "speculation_unexplained" and i.span.field == "opinion_rationale"
        ),
        None,
    )
    if rationale_ok and spec is None:
        return None
    spans = list(op.spans)
    present = next((i.span for i in f.free_text.items if i.kind == "rationale_present"), None)
    if present is not None:
        spans.append(present)
    if spec is not None:
        spans.append(spec)
    return spans


PREDICATES: dict[str, Predicate] = {
    "correia_passive": correia_passive,
    "correia_weight_bearing": correia_weight_bearing,
    "correia_opposite_joint": correia_opposite_joint,
    "deluca_rep_use": deluca_rep_use,
    "sharp_flare_estimate": sharp_flare_estimate,
    "mitchell_pain_function": mitchell_pain_function,
    "opinion_rationale": opinion_rationale,
}


def gap_check(findings: KneeFindings, retrieved: RetrievedContext) -> list[Claim]:
    gaps: list[Claim] = []
    for rule in retrieved.rules:
        pred = PREDICATES.get(rule.id)
        if pred is None:
            continue  # rule in KB without an implemented predicate: nothing to assert
        evidence = pred(findings)
        if evidence is None:
            continue
        gaps.append(_gap_claim(rule, evidence))
    return gaps


def _gap_claim(rule: AdequacyRule, evidence: list[Span]) -> Claim:
    return Claim(
        id=f"gap-{rule.id}",
        kind="gap",
        rule_id=rule.id,
        severity=rule.severity,
        statement=rule.description,
        evidence=evidence,
        citations=list(rule.authorities),
    )


def notes(findings: KneeFindings) -> list[Claim]:
    """Things that are not gaps but a reviewer should see."""
    out: list[Claim] = []
    if findings.arthritis_on_imaging:
        out.append(
            Claim(
                id="note-dc5003_consider",
                kind="note",
                statement=(
                    "Arthritis documented on imaging: consider DC 5003 and whether separate "
                    "ratings for limitation of motion and instability apply."
                ),
                evidence=_spans(findings, "imaging_arthritis"),
                citations=["cfr-4.71a-5003", "vaopgcprec-23-97", "vaopgcprec-9-98"],
            )
        )
    if any(v not in ("Normal", "NotTested") for v in findings.instability.values()):
        out.append(
            Claim(
                id="note-dc5257_consider",
                kind="note",
                statement=(
                    "Instability findings on examination: a separate rating under DC 5257 may "
                    "apply (criteria revised 2021; not modeled here)."
                ),
                evidence=_spans(
                    findings, "test_lachman", "test_posterior_drawer", "test_medial", "test_lateral"
                ),
                citations=["cfr-4.71a-5257", "vaopgcprec-23-97"],
            )
        )
    return out
