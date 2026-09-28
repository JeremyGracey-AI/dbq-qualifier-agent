"""Retrieve the criteria, adequacy rules and authorities this case needs.

Tags are derived from the findings so the rules checked downstream are the ones the exam
actually implicates (an opinion rule is only retrieved when an opinion was requested, etc.).
`extra_ids` lets the verify step ask for authorities a claim cited but retrieval missed.
"""

from __future__ import annotations

from datetime import date

from dbq_agent.kb import KnowledgeBase
from dbq_agent.models import Classification, KneeFindings, RetrievedContext


def tags_for(findings: KneeFindings, classification: Classification) -> list[str]:
    tags = ["knee", "rating", "rom", "normal", "zero", "4.7", *classification.candidate_dcs]
    tags += ["correia", "passive", "weight-bearing", "opposite-joint", "4.59"]
    tags += ["deluca", "repetitive-use", "functional-loss", "flare-ups"]
    if len(classification.candidate_dcs) > 1:
        tags += ["separate-ratings", "combined"]
    if findings.flare_ups_reported:
        tags += ["sharp", "flare-ups"]
    if findings.pain_on_motion:
        tags += ["mitchell", "pain", "painful-motion"]
    if findings.opinion is not None and findings.opinion.requested:
        tags += ["opinion", "nexus", "rationale", "speculation"]
    if findings.arthritis_on_imaging:
        tags += ["5003", "arthritis", "separate-ratings", "5257"]
    inst = findings.instability_findings
    if (inst is not None and inst.any_instability()) or any(
        v not in ("Normal", "NotTested") for v in findings.instability.values()
    ):
        tags += [
            "5257",
            "instability",
            "subluxation",
            "patellar",
            "effective-date",
            "separate-ratings",
        ]
    return sorted(set(tags))


def retrieve(
    kb: KnowledgeBase,
    findings: KneeFindings,
    classification: Classification,
    claim_date: date,
    extra_ids: list[str] | None = None,
    iteration: int = 0,
) -> RetrievedContext:
    query_text = " ".join(
        [
            findings.diagnosis or "",
            " ".join(findings.functional_loss_factors),
            "knee flexion extension",
        ]
    )
    return kb.retrieve(
        dcs=classification.candidate_dcs,
        as_of=claim_date,
        tags=tags_for(findings, classification),
        query_text=query_text,
        extra_ids=extra_ids,
        iteration=iteration,
    )
