from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from dbq_agent.kb import KnowledgeBase
from dbq_agent.pipeline import Context
from dbq_agent.synth.cases import all_cases
from dbq_agent.synth.knee import KneeTruth, build_pdf

CLAIM_DATE = date(2026, 9, 1)


@pytest.fixture(scope="session")
def cases_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Fresh synthetic cases in a temp dir, so tests never depend on committed PDFs."""
    out = tmp_path_factory.mktemp("cases")
    truth: dict[str, Any] = {}
    for case in all_cases():
        cid = str(case["case_id"])
        build_pdf(KneeTruth(case_id=cid, values=case["values"]), out / f"{cid}.pdf")
        truth[cid] = {
            "values": case["values"],
            "claim_date": case.get("claim_date"),
            "expected": case["expected"],
        }
    (out / "truth.json").write_text(json.dumps(truth))
    return out


@pytest.fixture(scope="session")
def truth(cases_dir: Path) -> dict[str, Any]:
    return json.loads((cases_dir / "truth.json").read_text())


@pytest.fixture(scope="session")
def kb() -> KnowledgeBase:
    return KnowledgeBase.load()


@pytest.fixture(scope="session")
def ctx(kb: KnowledgeBase) -> Context:
    return Context(kb=kb)
