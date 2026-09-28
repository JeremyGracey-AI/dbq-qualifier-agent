"""The critic. Every claim must be grounded and cited, or it does not reach the report.

A claim is verified when:
- it has at least one evidence span, and every span resolves: the field exists, the page matches
  the field's page, and the quoted text occurs verbatim in the field value (or in the
  de-identified view for `deid` spans, which is what a model actually read);
- it has at least one citation, and every citation is a real KB authority that was retrieved
  for this case.

A citation that exists in the KB but was not retrieved is reported as `unresolved` so the
pipeline can re-retrieve (bounded) instead of dropping a legitimate claim. Anything else is
dropped with a reason and listed in the report.
"""

from __future__ import annotations

from dbq_agent.kb import KnowledgeBase
from dbq_agent.models import (
    Claim,
    DeidPacket,
    IngestedDoc,
    RetrievedContext,
    Span,
    VerificationResult,
)


def resolve_span(span: Span, doc: IngestedDoc, deid: DeidPacket | None) -> str | None:
    """Return a failure reason, or None when the span checks out."""
    fv = doc.fields.get(span.field)
    if fv is None:
        return f"unknown field {span.field!r}"
    if fv.page != span.page:
        return f"page mismatch for {span.field!r}: claimed {span.page}, actual {fv.page}"
    haystack = fv.value
    if span.deid:
        if deid is None or span.field not in deid.free_text:
            return f"de-identified view missing for {span.field!r}"
        haystack = deid.free_text[span.field]
    if span.text == "":
        return f"empty quote for {span.field!r}"
    if span.text not in haystack:
        return f"quote not found verbatim in {span.field!r}"
    return None


def verify_claims(
    claims: list[Claim],
    doc: IngestedDoc,
    deid: DeidPacket | None,
    retrieved: RetrievedContext,
    kb: KnowledgeBase,
) -> tuple[list[Claim], list[str]]:
    """Mark each claim verified/dropped/pending(unresolved). Returns (claims, unresolved ids)."""
    retrieved_ids = retrieved.authority_ids()
    unresolved: set[str] = set()
    out: list[Claim] = []
    for claim in claims:
        c = claim.model_copy(deep=True)
        reasons: list[str] = []
        if not c.evidence:
            reasons.append("no evidence spans")
        for s in c.evidence:
            r = resolve_span(s, doc, deid)
            if r:
                reasons.append(r)
        if not c.citations:
            reasons.append("no citations")
        pending_ids: list[str] = []
        for cid in c.citations:
            if not kb.has_authority(cid):
                reasons.append(f"unknown authority {cid!r}")
            elif cid not in retrieved_ids:
                pending_ids.append(cid)
        if reasons:
            c.status = "dropped"
            c.drop_reason = "; ".join(reasons)
        elif pending_ids:
            c.status = "pending"
            unresolved.update(pending_ids)
        else:
            c.status = "verified"
            c.drop_reason = None
        out.append(c)
    return out, sorted(unresolved)


def finalize(claims: list[Claim]) -> list[Claim]:
    """After the last retrieval iteration, anything still pending is dropped."""
    out: list[Claim] = []
    for c in claims:
        if c.status == "pending":
            c = c.model_copy(
                update={"status": "dropped", "drop_reason": "citation not retrievable"}
            )
        out.append(c)
    return out


def summarize(claims: list[Claim], iterations: int, unresolved: list[str]) -> VerificationResult:
    return VerificationResult(
        verified=sum(1 for c in claims if c.status == "verified"),
        dropped=[c for c in claims if c.status == "dropped"],
        unresolved_citations=unresolved,
        iterations=iterations,
    )
