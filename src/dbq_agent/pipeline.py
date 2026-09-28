"""Pipeline runner.

    ingest → identity → deidentify → extract → classify → retrieve → evaluate → gap check → notes
           → verify ──(unresolved citations, ≤ max iterations)──▶ retrieve again → verify

Everything after `deidentify` sees identity only through `state.identity`, which no step
serializes. The re-retrieval loop is the "agentic" part of the slice: the critic can send the
pipeline back to the KB, but only a bounded number of times, and only for citations that exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from dbq_agent.extract import HeuristicTextExtractor, TextExtractor, extract
from dbq_agent.ingest import ingest_any
from dbq_agent.kb import KnowledgeBase
from dbq_agent.models import Evaluation, IngestedDoc, State
from dbq_agent.phi import Deidentifier, read_identity
from dbq_agent.steps.classify import classify
from dbq_agent.steps.evaluate import evaluate
from dbq_agent.steps.gap_check import gap_check, notes
from dbq_agent.steps.retrieve import retrieve
from dbq_agent.steps.verify import finalize, summarize, verify_claims


@dataclass
class Context:
    kb: KnowledgeBase = field(default_factory=KnowledgeBase.load)
    text_extractor: TextExtractor = field(default_factory=HeuristicTextExtractor)
    max_retrieval_iterations: int = 2
    deidentifier: Deidentifier | None = None  # default: Deidentifier.default(identity)


def run_doc(doc: IngestedDoc, claim_date: date, ctx: Context | None = None) -> State:
    ctx = ctx or Context()
    state = State(doc=doc, claim_date=claim_date)
    state.meta.kb_version = ctx.kb.version
    state.meta.source_kind = doc.source_kind
    state.meta.dpi = doc.dpi

    # PHI boundary -------------------------------------------------------------------
    state.identity = read_identity(doc)
    deidentifier = ctx.deidentifier or Deidentifier.default(state.identity)
    state.deid = deidentifier.deidentify(doc)
    state.meta.redactors = list(state.deid.redactors)
    state.meta.steps.append("deidentify")

    # Findings -----------------------------------------------------------------------
    state.findings = extract(doc, state.deid, ctx.text_extractor)
    state.meta.text_extractor = state.findings.free_text.extractor
    state.meta.steps.append("extract")

    state.classification = classify(doc, state.findings, ctx.kb.covered_dcs())
    state.meta.steps.append("classify")

    # Retrieve → evaluate → gaps -------------------------------------------------------
    state.retrieved = retrieve(ctx.kb, state.findings, state.classification, claim_date)
    state.meta.steps.append("retrieve")

    state.evaluation = evaluate(state.findings, state.retrieved)
    state.meta.steps.append("evaluate")

    state.gaps = gap_check(state.findings, state.retrieved)
    state.notes = notes(state.findings)
    state.meta.steps.append("gap_check")

    # Verify with bounded re-retrieval ---------------------------------------------------
    claims = state.all_claims()
    iteration = 0
    requested: list[str] = []  # cumulative, so a later iteration never forgets an earlier ask
    claims, unresolved = verify_claims(claims, doc, state.deid, state.retrieved, ctx.kb)
    while unresolved and iteration < ctx.max_retrieval_iterations:
        iteration += 1
        requested = sorted(set(requested) | set(unresolved))
        state.retrieved = retrieve(
            ctx.kb,
            state.findings,
            state.classification,
            claim_date,
            extra_ids=requested,
            iteration=iteration,
        )
        claims, unresolved = verify_claims(claims, doc, state.deid, state.retrieved, ctx.kb)
    claims = finalize(claims)
    state.meta.retrieval_iterations = iteration
    state.meta.steps.append("verify")

    # Write verified claims back where they came from --------------------------------------
    by_id = {c.id: c for c in claims}
    state.evaluation = Evaluation(
        claims=[by_id[c.id] for c in state.evaluation.claims],
        ratings=dict(state.evaluation.ratings),
    )
    state.gaps = [by_id[c.id] for c in state.gaps]
    state.notes = [by_id[c.id] for c in state.notes]
    state.verification = summarize(claims, iteration, unresolved)
    return state


def run(pdf_path: Path, claim_date: date, ctx: Context | None = None) -> State:
    """Fillable or scanned PDF; the ingest path is chosen by whether the file has fields."""
    return run_doc(ingest_any(pdf_path), claim_date, ctx)
