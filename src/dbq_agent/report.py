"""Build the qualifier report (JSON + markdown) from a finished pipeline state.

Only verified claims contribute. A blocking adequacy gap sets `rating` to None and keeps the
computed tiers under `provisional`, so the reviewer sees both what the exam supports and what
is missing before a rating could rest on it. Evidence quotes are re-identified here, inside the
boundary, for the human reader.
"""

from __future__ import annotations

import json
from datetime import date

from dbq_agent.kb import KnowledgeBase
from dbq_agent.models import Claim, QualifierReport, RatingLine, Span, State
from dbq_agent.phi import reidentify


def _reid_span(span: Span, token_map: dict[str, str]) -> Span:
    if not span.deid:
        return span
    return span.model_copy(update={"text": reidentify(span.text, token_map), "deid": False})


def _reid_claim(claim: Claim, token_map: dict[str, str]) -> Claim:
    return claim.model_copy(update={"evidence": [_reid_span(s, token_map) for s in claim.evidence]})


def build_report(state: State, kb: KnowledgeBase) -> QualifierReport:
    if state.evaluation is None or state.classification is None or state.verification is None:
        raise ValueError("pipeline did not finish")
    token_map = state.deid.token_map if state.deid else {}
    verified = [c for c in state.all_claims() if c.status == "verified"]
    dropped = [c for c in state.all_claims() if c.status == "dropped"]

    titles = {c.dc: c.dc_title for c in (state.retrieved.criteria if state.retrieved else [])}
    lines: dict[str, RatingLine] = {}
    for c in verified:
        if c.kind == "rating_tier" and c.dc is not None and c.pct is not None:
            lines[c.dc] = RatingLine(
                dc=c.dc,
                dc_title=titles.get(c.dc, ""),
                pct=c.pct,
                basis=c.statement,
                evidence=[_reid_span(s, token_map) for s in c.evidence],
                citations=list(c.citations),
            )
    for c in verified:
        if c.kind == "painful_motion_minimum" and c.dc is not None and c.pct is not None:
            base = lines.get(c.dc)
            lines[c.dc] = RatingLine(
                dc=c.dc,
                dc_title=titles.get(c.dc, ""),
                pct=max(c.pct, base.pct if base else 0),
                basis=c.statement,
                evidence=[_reid_span(s, token_map) for s in c.evidence]
                + (base.evidence if base else []),
                citations=list(c.citations) + (base.citations if base else []),
            )

    gaps = [_reid_claim(c, token_map) for c in verified if c.kind == "gap"]
    blocking = any(g.severity == "blocking" for g in gaps)
    return QualifierReport(
        case_id=state.doc.case_id,
        form_type=state.classification.form_type,
        form_number=state.classification.form_number,
        claim_date=state.claim_date,
        adequate=not blocking,
        rating=None if blocking else dict(lines),
        provisional=dict(lines),
        gaps=gaps,
        notes=[_reid_claim(c, token_map) for c in verified if c.kind == "note"],
        consider=[_reid_claim(c, token_map) for c in verified if c.kind == "consider_higher"],
        dropped_claims=[_reid_claim(c, token_map) for c in dropped],
        veteran=state.identity,
        pipeline=state.meta,
    )


def to_json(report: QualifierReport) -> str:
    return json.dumps(report.model_dump(mode="json"), indent=2)


def _cite(kb: KnowledgeBase, cid: str) -> str:
    a = kb.authority(cid)
    if a is None:
        return cid
    flag = " (verify)" if a.verify else ""
    return f"{a.cite}{flag}"


def _evidence_lines(spans: list[Span]) -> list[str]:
    out: list[str] = []
    for s in spans:
        q = s.text if len(s.text) <= 90 else s.text[:87] + "..."
        out.append(f'    - p.{s.page} `{s.field}`: "{q}"')
    return out


def to_markdown(report: QualifierReport, kb: KnowledgeBase, claim_date: date | None = None) -> str:
    md: list[str] = []
    vet = report.veteran.name if report.veteran and report.veteran.name else "(unknown)"
    md.append(f"# DBQ qualifier report — {report.case_id}")
    md.append("")
    md.append(
        f"Veteran: {vet} · Form: {report.form_number or report.form_type} · "
        f"Claim date: {report.claim_date.isoformat()} · KB {report.pipeline.kb_version}"
    )
    md.append("")
    status = "ADEQUATE" if report.adequate else "INADEQUATE — rating withheld (see gaps)"
    md.append(f"**Exam adequacy:** {status}")
    md.append("")
    title = (
        "Rating" if report.adequate else "Provisional rating (not assignable until gaps are cured)"
    )
    md.append(f"## {title}")
    md.append("")
    if not report.provisional:
        md.append("_No rating criteria could be evaluated._")
    for dc, line in sorted(report.provisional.items()):
        md.append(f"- **DC {dc} — {line.dc_title}: {line.pct}%**")
        md.append(f"  - Basis: {line.basis}")
        md.append("  - Evidence:")
        md.extend(_evidence_lines(line.evidence))
        md.append(
            "  - Authority: " + "; ".join(_cite(kb, c) for c in dict.fromkeys(line.citations))
        )
    if report.consider:
        md.append("")
        md.append("## Consider (reviewer judgment)")
        md.append("")
        for c in report.consider:
            md.append(f"- {c.statement}")
            md.extend(_evidence_lines(c.evidence))
            md.append("  - Authority: " + "; ".join(_cite(kb, x) for x in c.citations))
    md.append("")
    md.append("## Exam adequacy gaps")
    md.append("")
    if not report.gaps:
        md.append("_None found by the modeled rules._")
    for g in report.gaps:
        md.append(f"- **[{g.severity}] {g.rule_id}** — {g.statement}")
        md.extend(_evidence_lines(g.evidence))
        md.append("  - Authority: " + "; ".join(_cite(kb, x) for x in g.citations))
    if report.notes:
        md.append("")
        md.append("## Notes")
        md.append("")
        for n in report.notes:
            md.append(f"- {n.statement}")
            md.append("  - Authority: " + "; ".join(_cite(kb, x) for x in n.citations))
    if report.dropped_claims:
        md.append("")
        md.append("## Dropped by the verifier")
        md.append("")
        for d in report.dropped_claims:
            md.append(f"- `{d.id}`: {d.drop_reason}")
    md.append("")
    md.append("## Pipeline")
    md.append("")
    md.append(
        f"steps: {' → '.join(report.pipeline.steps)} · text extractor: {report.pipeline.text_extractor} · "
        f"redactors: {', '.join(report.pipeline.redactors)} · re-retrievals: {report.pipeline.retrieval_iterations}"
    )
    md.append("")
    md.append(f"_{report.disclaimer}_")
    md.append("")
    return "\n".join(md)
