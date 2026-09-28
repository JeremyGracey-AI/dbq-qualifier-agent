from datetime import date

from dbq_agent.kb import KnowledgeBase
from dbq_agent.models import Criterion
from dbq_agent.steps.evaluate import select_tier

AS_OF = date(2026, 9, 1)


def test_kb_loads_and_validates() -> None:
    kb = KnowledgeBase.load()
    assert kb.covered_dcs() == {"5260", "5261"}
    assert kb.has_authority("correia-2016")
    assert all(kb.has_authority(a) for r in kb.rules for a in r.authorities)


def test_5260_tiers_match_schedule() -> None:
    kb = KnowledgeBase.load()
    rows = kb.criteria_for("5260", AS_OF)
    assert [(c.threshold_deg, c.pct) for c in rows] == [(15, 30), (30, 20), (45, 10), (60, 0)]
    assert select_tier(rows, 45)[1] == 10
    assert select_tier(rows, 40)[1] == 10  # still "limited to 45", not yet 30
    assert select_tier(rows, 30)[1] == 20
    assert select_tier(rows, 15)[1] == 30
    assert select_tier(rows, 60)[1] == 0
    assert select_tier(rows, 90) == (None, 0)  # no listed criterion met -> 0% via 4.31


def test_5261_tiers_match_schedule() -> None:
    kb = KnowledgeBase.load()
    rows = kb.criteria_for("5261", AS_OF)
    assert [(c.threshold_deg, c.pct) for c in rows] == [
        (45, 50),
        (30, 40),
        (20, 30),
        (15, 20),
        (10, 10),
        (5, 0),
    ]
    assert select_tier(rows, 20)[1] == 30
    assert select_tier(rows, 25)[1] == 30
    assert select_tier(rows, 10)[1] == 10
    assert select_tier(rows, 0) == (None, 0)


def test_vaopgcprec_9_2004_examples() -> None:
    """Worked examples from VAOPGCPREC 9-2004 as regression vectors."""
    kb = KnowledgeBase.load()
    flex = kb.criteria_for("5260", AS_OF)
    ext = kb.criteria_for("5261", AS_OF)
    # ROM 0-30: 20% for flexion limited to 30; extension beyond 5 -> not compensable
    assert select_tier(flex, 30)[1] == 20
    assert select_tier(ext, 0) == (None, 0)
    # ROM 30-90: extension limited to 30 -> 40%; flexion to 90 -> nothing
    assert select_tier(ext, 30)[1] == 40
    assert select_tier(flex, 90) == (None, 0)
    # ROM 15-45: 10% flexion + 20% extension (combined 30 under 4.25)
    assert select_tier(flex, 45)[1] == 10
    assert select_tier(ext, 15)[1] == 20


def test_effective_date_filter() -> None:
    kb = KnowledgeBase.load()
    old = Criterion(
        id="9999-old",
        dc="9999",
        dc_title="fixture",
        pct=10,
        measure="flexion",
        op="<=",
        threshold_deg=45,
        text="old",
        cite="fixture",
        effective_from=date(2000, 1, 1),
        effective_to=date(2021, 2, 7),
    )
    new = old.model_copy(
        update={
            "id": "9999-new",
            "pct": 20,
            "effective_from": date(2021, 2, 7),
            "effective_to": None,
        }
    )
    kb.criteria.extend([old, new])
    assert [c.id for c in kb.criteria_for("9999", date(2020, 6, 1))] == ["9999-old"]
    assert [c.id for c in kb.criteria_for("9999", date(2021, 2, 7))] == ["9999-new"]


def test_retrieve_includes_rule_authorities_and_extra_ids() -> None:
    kb = KnowledgeBase.load()
    ctx = kb.retrieve(["5260", "5261"], AS_OF, tags=["knee", "correia"], extra_ids=["lyles-2017"])
    ids = ctx.authority_ids()
    assert {"cfr-4.71a-5260", "cfr-4.71a-5261", "correia-2016", "cfr-4.59", "lyles-2017"} <= ids
    assert [c.dc for c in ctx.criteria] == ["5260"] * 4 + ["5261"] * 6
    assert any(r.id == "correia_passive" for r in ctx.rules)
