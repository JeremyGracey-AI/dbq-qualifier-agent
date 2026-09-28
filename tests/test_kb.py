from datetime import date

from dbq_agent.kb import KnowledgeBase
from dbq_agent.models import Criterion
from dbq_agent.steps.evaluate import select_tier

AS_OF = date(2026, 9, 1)


def test_kb_loads_and_validates() -> None:
    kb = KnowledgeBase.load()
    assert kb.covered_dcs() == {"5257", "5260", "5261"}
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


def test_5257_has_two_versions_split_at_2021_02_07() -> None:
    kb = KnowledgeBase.load()
    old = kb.criteria_for("5257", date(2021, 2, 6))
    new = kb.criteria_for("5257", date(2021, 2, 7))
    assert [c.pct for c in old] == [30, 20, 10]
    assert all(c.kind == "predicate" and c.effective_to == date(2021, 2, 7) for c in old)
    assert sorted({c.subtable for c in new if c.subtable}) == ["ligament", "patellar"]
    assert len(new) == 6 and all(c.effective_from == date(2021, 2, 7) for c in new)


def _pct(rows: list[Criterion], facts: dict[str, str]) -> int:
    from dbq_agent.steps.evaluate import select_predicate_tier

    row, _ = select_predicate_tier(rows, facts)
    return row.pct if row is not None else -1


def test_5257_2021_predicates_follow_the_schedule_text() -> None:
    kb = KnowledgeBase.load()
    rows = kb.criteria_for("5257", AS_OF)
    lig = {"ligament_injury": "CompleteTear", "persistent_instability": "yes"}
    # unrepaired complete tear: both brace and device -> 30; either alone -> 20; neither -> 10
    both = lig | {
        "ligament_repair_status": "Unrepaired",
        "rx_bracing": "yes",
        "rx_assistive_device": "Cane",
    }
    assert _pct(rows, both) == 30
    assert _pct(rows, both | {"rx_assistive_device": "None"}) == 20
    assert _pct(rows, both | {"rx_bracing": "no", "rx_assistive_device": "None"}) == 10
    # a repaired complete tear with both prescriptions is 20, not 30 (30 needs unrepaired/failed)
    assert _pct(rows, both | {"ligament_repair_status": "Repaired"}) == 20
    # sprain with persistent instability: a brace alone reaches 20; nothing prescribed is 10
    sprain = {"ligament_injury": "Sprain", "persistent_instability": "yes"}
    assert _pct(rows, sprain | {"rx_bracing": "yes", "rx_assistive_device": "None"}) == 20
    assert _pct(rows, sprain | {"rx_bracing": "no", "rx_assistive_device": "None"}) == 10
    # patellar: after surgical repair, brace + walker -> 30; brace alone -> 20; nothing -> 10;
    # crutches do not count for the patellar sub-table
    pat = {"patellar_instability": "yes", "patellar_surgical_repair": "yes"}
    assert _pct(rows, pat | {"rx_bracing": "yes", "rx_assistive_device": "Walker"}) == 30
    assert _pct(rows, pat | {"rx_bracing": "yes", "rx_assistive_device": "None"}) == 20
    assert _pct(rows, pat | {"rx_bracing": "no", "rx_assistive_device": "Crutches"}) == 10
    # instability answered but persistence not documented -> nothing rated (gap check speaks)
    assert _pct(rows, {"ligament_injury": "Sprain"}) == -1


def test_5257_pre_2021_uses_examiner_severity() -> None:
    kb = KnowledgeBase.load()
    rows = kb.criteria_for("5257", date(2020, 6, 1))
    assert _pct(rows, {"severity": "moderate"}) == 20
    assert _pct(rows, {"severity": "severe"}) == 30
    assert _pct(rows, {"persistent_instability": "yes"}) == -1
