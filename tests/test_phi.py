"""The PHI boundary, enforced.

If any of these fail, PHI can reach a model. Do not weaken them; fix the gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dbq_agent.extract import LLMTextExtractor
from dbq_agent.ingest import ingest_pdf
from dbq_agent.llm import CapturingClient
from dbq_agent.models import Identity, IngestedDoc
from dbq_agent.phi import (
    FREE_TEXT_FIELDS,
    IDENTITY_FIELDS,
    Deidentifier,
    KnownValueDetector,
    PatternDetector,
    assert_no_phi,
    read_identity,
    reidentify,
)
from dbq_agent.pipeline import Context, run_doc
from tests.conftest import CLAIM_DATE


def test_identity_and_free_text_fields_are_disjoint() -> None:
    assert not set(IDENTITY_FIELDS) & set(FREE_TEXT_FIELDS)


def test_known_values_and_patterns_are_tokenized() -> None:
    ident = Identity(
        name="Kekoa M. Naeole", dob="03/14/1975", ssn="900-12-3456", examiner="Dr. A. Fictional, MD"
    )
    doc = IngestedDoc(
        case_id="t",
        source_path="t",
        page_count=1,
        fields={
            "remarks": {  # type: ignore[dict-item]
                "name": "remarks",
                "page": 1,
                "value": (
                    "Mr. Naeole (born March 14, 1975; SSN 900123456) called (808) 555-0142. "
                    "Seen by Dr. Fictional on 2026-08-12. Kekoa reports pain."
                ),
                "kind": "text",
            }
        },
    )
    deid = Deidentifier([KnownValueDetector(ident), PatternDetector()]).deidentify(doc)
    text = deid.free_text["remarks"]
    for leak in ("Naeole", "Kekoa", "1975", "900123456", "555-0142", "Fictional", "2026-08-12"):
        assert leak not in text, text
    assert "[NAME_1]" in text and "[DOB_1]" in text and "[SSN_1]" in text and "[PHONE_1]" in text
    assert reidentify(text, deid.token_map) == doc.fields["remarks"].value


def test_deidentified_view_never_contains_identity(cases_dir: Path, truth: dict[str, Any]) -> None:
    for cid in truth:
        doc = ingest_pdf(cases_dir / f"{cid}.pdf")
        ident = read_identity(doc)
        deid = Deidentifier.default(ident).deidentify(doc)
        assert set(deid.free_text) <= set(FREE_TEXT_FIELDS)
        assert_no_phi("\n".join(deid.free_text.values()), ident)


def test_llm_payloads_never_contain_phi(
    cases_dir: Path, truth: dict[str, Any], ctx: Context
) -> None:
    """Run the whole pipeline with a capturing fake model and inspect every payload."""
    client = CapturingClient(canned={"items": []})
    llm_ctx = Context(kb=ctx.kb, text_extractor=LLMTextExtractor(client))
    for cid in truth:
        doc = ingest_pdf(cases_dir / f"{cid}.pdf")
        state = run_doc(doc, CLAIM_DATE, llm_ctx)
        assert state.meta.text_extractor == "llm:capturing"
    assert client.calls, "the fake model was never called"
    payload = client.payload_text()
    for cid in truth:
        doc = ingest_pdf(cases_dir / f"{cid}.pdf")
        assert_no_phi(payload, read_identity(doc))
    for name in IDENTITY_FIELDS:
        assert f"[{name}]" not in payload  # identity fields are never even listed


def test_assert_no_phi_catches_a_leak() -> None:
    with pytest.raises(AssertionError):
        assert_no_phi("note about Naeole", Identity(name="Kekoa Naeole"))
