from pathlib import Path
from typing import Any

from dbq_agent.ingest import ingest_pdf
from dbq_agent.synth.form_spec import ALL_WIDGETS, PAGE_OF


def test_every_case_round_trips_through_ingest(cases_dir: Path, truth: dict[str, Any]) -> None:
    for cid, entry in truth.items():
        doc = ingest_pdf(cases_dir / f"{cid}.pdf")
        assert doc.case_id == cid
        assert doc.page_count == 5
        for name, expected in entry["values"].items():
            assert doc.has(name), f"{cid}: field {name} missing"
            assert doc.get(name) == expected, f"{cid}: {name}"
            assert doc.fields[name].page == PAGE_OF[name], f"{cid}: page for {name}"


def test_field_kinds_are_detected(cases_dir: Path) -> None:
    doc = ingest_pdf(cases_dir / "knee_01.pdf")
    assert doc.fields["pain_noted_on_exam"].kind == "radio"
    assert doc.fields["factor_pain"].kind == "checkbox"
    assert doc.fields["remarks"].kind == "text"
    assert set(doc.fields) == set(ALL_WIDGETS)


def test_structured_extraction_matches_truth_exactly(
    cases_dir: Path, truth: dict[str, Any]
) -> None:
    """Structured-field F1 = 1.0: every numeric/boolean finding equals the truth value."""
    from dbq_agent.extract import extract_structured

    for cid, entry in truth.items():
        doc = ingest_pdf(cases_dir / f"{cid}.pdf")
        f = extract_structured(doc)
        v = entry["values"]
        assert f.initial.flexion == int(v["rom_flexion_initial"])
        assert f.initial.extension == int(v["rom_extension_initial"])
        assert f.pain_on_motion is (v["pain_noted_on_exam"] == "Yes")
        assert f.flare_ups_reported is (v["flare_ups"] == "Yes")
        assert f.rep_use_performed is (v["rep_use_performed"] == "Yes")
        if v["rom_flexion_post_rep"]:
            assert f.post_rep is not None and f.post_rep.flexion == int(v["rom_flexion_post_rep"])
        else:
            assert f.post_rep is None
        assert ("weakened movement" in f.functional_loss_factors) == (
            v.get("factor_weakness") == "Yes"
        )
