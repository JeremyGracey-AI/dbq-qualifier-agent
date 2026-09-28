from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dbq_agent.extract import LLMTextExtractor
from dbq_agent.ingest import ingest_pdf
from dbq_agent.llm import CapturingClient
from dbq_agent.models import Claim, Span
from dbq_agent.pipeline import Context, run, run_doc
from dbq_agent.report import build_report, to_json, to_markdown
from dbq_agent.steps.verify import verify_claims
from tests.conftest import CLAIM_DATE


def test_golden_outcomes_for_every_case(
    cases_dir: Path, truth: dict[str, Any], ctx: Context
) -> None:
    for cid, entry in truth.items():
        exp = entry["expected"]
        state = run(cases_dir / f"{cid}.pdf", CLAIM_DATE, ctx)
        report = build_report(state, ctx.kb)
        assert report.adequate is exp["adequate"], cid
        assert {dc: line.pct for dc, line in report.provisional.items()} == exp["ratings"], cid
        assert (report.rating is None) is (not exp["adequate"]), cid
        assert sorted(g.rule_id or "" for g in report.gaps) == sorted(exp["gaps"]), cid
        assert sorted(c.dc or "" for c in report.consider) == sorted(exp["consider_higher"]), cid
        for note_id in exp["notes"]:
            assert any(n.id == f"note-{note_id}" for n in report.notes), cid
        assert report.dropped_claims == [], cid
        if exp["painful_motion_minimum"]:
            assert any(
                c.kind == "painful_motion_minimum" and c.status == "verified"
                for c in state.all_claims()
            ), cid
        for dc, source in exp["limiting_source"].items():
            claim = next(c for c in state.all_claims() if c.id == f"rating-{dc}")
            expected_note = {
                "post_rep": "after repetitive-use testing",
                "flare_estimate": "examiner's flare-up estimate",
                "initial": "initial measurement",
            }[source]
            assert expected_note in claim.statement, cid


def test_every_verified_claim_is_grounded_and_cited(
    cases_dir: Path, truth: dict[str, Any], ctx: Context
) -> None:
    """Citation faithfulness: every quote is verbatim in the document, every cite is a KB id."""
    for cid in truth:
        state = run(cases_dir / f"{cid}.pdf", CLAIM_DATE, ctx)
        assert state.deid is not None
        for c in state.all_claims():
            assert c.status == "verified", (cid, c.id, c.drop_reason)
            assert c.evidence and c.citations
            for s in c.evidence:
                fv = state.doc.fields[s.field]
                assert fv.page == s.page
                haystack = state.deid.free_text[s.field] if s.deid else fv.value
                assert s.text in haystack
            for cite in c.citations:
                assert ctx.kb.has_authority(cite)


def test_inadequate_case_withholds_rating_but_keeps_provisional(
    cases_dir: Path, ctx: Context
) -> None:
    report = build_report(run(cases_dir / "knee_07.pdf", CLAIM_DATE, ctx), ctx.kb)
    assert report.rating is None
    assert report.provisional["5260"].pct == 10
    assert [g.rule_id for g in report.gaps] == ["correia_passive"]
    assert report.gaps[0].severity == "blocking"
    assert "correia-2016" in report.gaps[0].citations


def test_verifier_drops_ungrounded_and_uncited_claims(cases_dir: Path, ctx: Context) -> None:
    state = run(cases_dir / "knee_01.pdf", CLAIM_DATE, ctx)
    assert state.retrieved is not None
    bad_span = Claim(
        id="x-span",
        kind="note",
        statement="quote not in doc",
        evidence=[Span(field="remarks", page=4, text="this text does not appear")],
        citations=["cfr-4.59"],
    )
    bad_page = Claim(
        id="x-page",
        kind="note",
        statement="wrong page",
        evidence=[Span(field="remarks", page=1, text="flares")],
        citations=["cfr-4.59"],
    )
    bad_cite = Claim(
        id="x-cite",
        kind="note",
        statement="made-up authority",
        evidence=[state.doc.span("rom_flexion_initial")],
        citations=["smith-v-nobody-2030"],
    )
    no_evidence = Claim(id="x-none", kind="note", statement="nothing", citations=["cfr-4.59"])
    out, unresolved = verify_claims(
        [bad_span, bad_page, bad_cite, no_evidence], state.doc, state.deid, state.retrieved, ctx.kb
    )
    assert [c.status for c in out] == ["dropped"] * 4
    assert unresolved == []
    assert "quote not found" in (out[0].drop_reason or "")
    assert "page mismatch" in (out[1].drop_reason or "")
    assert "unknown authority" in (out[2].drop_reason or "")
    assert "no evidence" in (out[3].drop_reason or "")


def test_re_retrieval_loop_resolves_a_real_but_unretrieved_citation(
    cases_dir: Path, ctx: Context
) -> None:
    """A claim cites a KB authority that first-pass retrieval did not return."""
    doc = ingest_pdf(cases_dir / "knee_01.pdf")
    from dbq_agent.steps import gap_check as gc

    original = gc.notes

    def notes_with_extra(findings):  # type: ignore[no-untyped-def]
        extra = Claim(
            id="note-extra",
            kind="note",
            statement="meniscal symptoms may warrant a separate rating",
            evidence=[doc.span("dx_1")],
            citations=["lyles-2017"],
        )
        return [*original(findings), extra]

    import dbq_agent.pipeline as pipeline_mod

    pipeline_mod.notes = notes_with_extra  # type: ignore[assignment]
    try:
        first_pass = ctx.kb.retrieve(
            ["5260", "5261"], CLAIM_DATE, tags=["knee"], top_k_authorities=3
        )
        assert "lyles-2017" not in first_pass.authority_ids()
        small_ctx = Context(kb=ctx.kb, max_retrieval_iterations=2)
        state = run_doc(doc, CLAIM_DATE, small_ctx)
    finally:
        pipeline_mod.notes = original  # type: ignore[assignment]
    extra = next(c for c in state.all_claims() if c.id == "note-extra")
    assert extra.status == "verified"
    assert state.verification is not None
    assert state.verification.unresolved_citations == []


def test_loop_is_bounded(cases_dir: Path, ctx: Context) -> None:
    doc = ingest_pdf(cases_dir / "knee_01.pdf")
    state = run_doc(doc, CLAIM_DATE, Context(kb=ctx.kb, max_retrieval_iterations=0))
    assert state.meta.retrieval_iterations == 0
    assert all(c.status in ("verified", "dropped") for c in state.all_claims())


def test_llm_extractor_rejects_hallucinated_quotes(cases_dir: Path, ctx: Context) -> None:
    client = CapturingClient(
        canned={
            "items": [
                {
                    "kind": "flare_flexion_estimate",
                    "value": 15,
                    "field": "remarks",
                    "quote": "flexion to 15 degrees",
                },
                {
                    "kind": "rationale_present",
                    "value": True,
                    "field": "opinion_rationale",
                    "quote": "at least as likely as not",
                },
            ]
        }
    )
    state = run(
        cases_dir / "knee_01.pdf",
        CLAIM_DATE,
        Context(kb=ctx.kb, text_extractor=LLMTextExtractor(client)),
    )
    assert state.findings is not None
    kinds = [i.kind for i in state.findings.free_text.items]
    assert kinds == [
        "rationale_present"
    ]  # the fabricated flare estimate never entered the findings
    assert {dc: pct for dc, pct in state.evaluation.ratings.items()} == {"5260": 10, "5261": 0}  # type: ignore[union-attr]


def test_report_serializes(cases_dir: Path, ctx: Context) -> None:
    state = run(cases_dir / "knee_02.pdf", CLAIM_DATE, ctx)
    report = build_report(state, ctx.kb)
    js = to_json(report)
    md = to_markdown(report, ctx.kb)
    assert '"case_id": "knee_02"' in js
    assert "DC 5260" in md and "DC 5261" in md and "VAOPGCPREC 9-2004" in md
    assert "Dana R. Whitfield" in md  # re-identified for the reviewer

    # knee_05's rating rests on a remarks quote the model saw de-identified; the reviewer sees it whole
    state = run(cases_dir / "knee_05.pdf", CLAIM_DATE, ctx)
    md = to_markdown(build_report(state, ctx.kb), ctx.kb)
    assert "[NAME_1]" not in md and "Robert J. Okafor, during flare-ups" in md


@pytest.mark.parametrize("bad_date", ["1900-01-01"])
def test_claim_date_before_criteria_yields_no_rating(
    cases_dir: Path, ctx: Context, bad_date: str
) -> None:
    from datetime import date

    state = run(cases_dir / "knee_01.pdf", date.fromisoformat(bad_date), ctx)
    assert state.evaluation is not None and state.evaluation.ratings == {}


class _RaisingClient:
    name = "raising"

    def extract_json(self, system: str, user: str, schema: dict) -> dict:  # type: ignore[type-arg]
        raise ConnectionError("simulated 401")


def test_llm_fallback_is_loud_and_named(cases_dir: Path, ctx: Context) -> None:
    from dbq_agent.extract import HeuristicTextExtractor

    extractor = LLMTextExtractor(_RaisingClient(), fallback=HeuristicTextExtractor())
    with pytest.warns(RuntimeWarning, match="ConnectionError"):
        state = run(
            cases_dir / "knee_05.pdf", CLAIM_DATE, Context(kb=ctx.kb, text_extractor=extractor)
        )
    assert state.meta.text_extractor == "llm:raising->fallback:heuristic (ConnectionError)"
    assert state.evaluation is not None and state.evaluation.ratings["5260"] == 20


def test_llm_strict_mode_raises(cases_dir: Path, ctx: Context) -> None:
    extractor = LLMTextExtractor(_RaisingClient(), fallback=None)
    with pytest.raises(ConnectionError):
        run(cases_dir / "knee_05.pdf", CLAIM_DATE, Context(kb=ctx.kb, text_extractor=extractor))


def test_llm_values_are_normalized_to_what_the_rules_expect(cases_dir: Path, ctx: Context) -> None:
    """Exactly what claude-sonnet-5 returned for knee_05 on 2026-09-27: strings, not ints."""
    client = CapturingClient(
        canned={
            "items": [
                {
                    "kind": "flare_flexion_estimate",
                    "field": "remarks",
                    "quote": "during flare-ups flexion is estimated to be limited to 30 degrees",
                    "value": "30 degrees",
                },
                {
                    "kind": "flare_extension_estimate",
                    "field": "remarks",
                    "quote": "extension is not additionally limited",
                    "value": "not additionally limited",
                },
                {
                    "kind": "rationale_present",
                    "field": "opinion_rationale",
                    "quote": "There is no evidence of an intercurrent injury.",
                    "value": "true",
                },
                {
                    "kind": "functional_loss_factor",
                    "field": "functional_loss_desc",
                    "quote": "Difficulty with prolonged standing, squatting, and stairs.",
                    "value": None,
                },
            ]
        }
    )
    state = run(
        cases_dir / "knee_05.pdf",
        CLAIM_DATE,
        Context(kb=ctx.kb, text_extractor=LLMTextExtractor(client)),
    )
    assert state.findings is not None
    kinds = {(i.kind, i.value) for i in state.findings.free_text.items}
    assert ("flare_flexion_estimate", 30) in kinds
    assert ("rationale_present", True) in kinds
    assert not any(k == "flare_extension_estimate" for k, _ in kinds)  # no number -> dropped
    assert not any(k == "functional_loss_factor" for k, _ in kinds)  # no factor named -> dropped
    assert state.evaluation is not None and state.evaluation.ratings["5260"] == 20


def test_bare_speculation_is_never_accepted_as_a_reason(cases_dir: Path, ctx: Context) -> None:
    """What claude-sonnet-5 returned for knee_08: the speculation sentence filed as a 'reason'."""
    client = CapturingClient(
        canned={
            "items": [
                {
                    "kind": "flare_no_estimate_reason",
                    "field": "flare_no_estimate_reason",
                    "quote": "Unable to say without resorting to mere speculation.",
                    "value": "Unable to say without resorting to mere speculation.",
                },
                {
                    "kind": "speculation_unexplained",
                    "field": "flare_no_estimate_reason",
                    "quote": "Unable to say without resorting to mere speculation.",
                    "value": False,
                },
                {
                    "kind": "rationale_present",
                    "field": "opinion_rationale",
                    "quote": "There is no evidence of an intercurrent injury.",
                    "value": True,
                },
            ]
        }
    )
    state = run(
        cases_dir / "knee_08.pdf",
        CLAIM_DATE,
        Context(kb=ctx.kb, text_extractor=LLMTextExtractor(client)),
    )
    report = build_report(state, ctx.kb)
    assert report.adequate is False
    assert [g.rule_id for g in report.gaps] == ["sharp_flare_estimate"]
    assert state.findings is not None
    assert {(i.kind, i.value) for i in state.findings.free_text.items} == {
        ("speculation_unexplained", True),
        ("rationale_present", True),
    }
