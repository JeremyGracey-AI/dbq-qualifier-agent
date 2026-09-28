"""Ten synthetic knee cases with ground truth. All names, dates and numbers are fictional.

Seven adequate exams and three inadequate ones (Correia, Sharp, DeLuca+opinion). Expected
outcomes are what the deterministic evaluate/gap-check steps should produce; the free-text
extractor only matters for knee_05 (flare estimate lives in the remarks, not the ROM field)
and for the rationale/speculation checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dbq_agent.synth.knee import KneeTruth, blank_values, build_pdf

DEFAULT_CASES_DIR = Path(__file__).resolve().parents[3] / "cases"

# Fictional identities. SSNs use the 900 series, which SSA does not issue.
IDENTITIES = [
    ("Kekoa M. Naeole", "03/14/1975", "900-12-3456"),
    ("Dana R. Whitfield", "11/02/1981", "900-23-4567"),
    ("Miguel A. Santos", "07/29/1968", "900-34-5678"),
    ("Leilani K. Ahuna", "01/16/1990", "900-45-6789"),
    ("Robert J. Okafor", "09/08/1972", "900-56-7890"),
    ("Priya N. Raman", "05/21/1985", "900-67-8901"),
    ("Thomas E. Grady", "12/30/1963", "900-78-9012"),
    ("Aolani P. Kahale", "04/03/1993", "900-89-0123"),
    ("Samuel T. Vance", "08/17/1979", "900-90-1234"),
    ("Chloe B. Marsh", "02/25/1988", "900-01-2345"),
]


def _adequate_base(idx: int, flexion: int, extension: int) -> dict[str, str]:
    name, dob, ssn = IDENTITIES[idx]
    v = blank_values()
    v.update(
        {
            "form_number": "21-0960M-9",
            "vet_name": name,
            "vet_dob": dob,
            "vet_ssn": ssn,
            "exam_date": "08/12/2026",
            "examiner_name": "Dr. A. Fictional, MD",
            "claimed_side": "Right",
            "dx_1": "Right knee strain with chondromalacia patella",
            "dx_icd_1": "M22.41",
            "history": (
                "Veteran reports right knee pain beginning in service after a fall during a field "
                "exercise in 2011, treated conservatively. Symptoms have persisted with daily pain "
                "and stiffness. Uses ibuprofen as needed."
            ),
            "flare_ups": "No",
            "functional_loss": "Yes",
            "functional_loss_desc": "Difficulty with prolonged standing, squatting, and stairs.",
            "rom_flexion_initial": str(flexion),
            "rom_extension_initial": str(extension),
            "rom_abnormal": "Yes",
            "rom_contributes_functional_loss": "Yes",
            "pain_noted_on_exam": "Yes",
            "pain_causes_functional_loss": "Yes",
            "pain_weight_bearing": "Yes",
            "pain_non_weight_bearing": "No",
            "pain_passive_rom": "Yes",
            "tenderness": "Yes",
            "crepitus": "No",
            "opposite_joint_undamaged": "Yes",
            "opposite_flexion": "140",
            "opposite_extension": "0",
            "rep_use_performed": "Yes",
            "rom_flexion_post_rep": str(flexion),
            "rom_extension_post_rep": str(extension),
            "rep_use_additional_loss": "No",
            "exam_during_flare": "No",
            "flare_functional_loss": "No",
            "factor_pain": "Yes",
            "factor_less_movement": "Yes",
            "strength_flexion": "5/5",
            "strength_extension": "5/5",
            "atrophy": "No",
            "ankylosis": "No",
            "test_lachman": "Normal",
            "test_posterior_drawer": "Normal",
            "test_medial": "Normal",
            "test_lateral": "Normal",
            "subluxation_history": "None",
            "imaging_performed": "Yes",
            "imaging_arthritis": "No",
            "functional_impact": (
                "The knee condition limits prolonged standing and walking; the Veteran can perform "
                "sedentary work with position changes."
            ),
            "remarks": "",
            "opinion_requested": "Yes",
            "opinion_conclusion": "AtLeastAsLikely",
            "opinion_rationale": (
                "The current right knee condition is at least as likely as not related to the "
                "documented in-service fall because the service treatment records show a knee "
                "injury with continued complaints at separation, and the current findings are "
                "consistent with that mechanism. There is no evidence of an intercurrent injury."
            ),
        }
    )
    return v


def _case(
    case_id: str,
    values: dict[str, str],
    *,
    adequate: bool,
    ratings: dict[str, int],
    gaps: list[str] | None = None,
    consider_higher: list[str] | None = None,
    notes: list[str] | None = None,
    painful_motion_minimum: bool = False,
    limiting_source: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "values": values,
        "expected": {
            "adequate": adequate,
            "ratings": ratings,
            "gaps": gaps or [],
            "consider_higher": consider_higher or [],
            "notes": notes or [],
            "painful_motion_minimum": painful_motion_minimum,
            "limiting_source": limiting_source or {},
        },
    }


def all_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    # 01 — plain flexion 45 with a flare estimate that stays in the same tier
    v = _adequate_base(0, 45, 0)
    v.update(
        {
            "flare_ups": "Yes",
            "flare_ups_desc": "Weekly flares lasting 1-2 days after prolonged walking; severity moderate.",
            "flare_functional_loss": "Yes",
            "flare_flexion_est": "40",
            "flare_extension_est": "0",
            "remarks": f"Mr. {IDENTITIES[0][0].split()[-1]} (DOB {IDENTITIES[0][1]}) reports flares worsen with cold weather.",
        }
    )
    cases.append(
        _case(
            "knee_01",
            v,
            adequate=True,
            ratings={"5260": 10, "5261": 0},
            limiting_source={"5260": "flare_estimate"},
        )
    )

    # 02 — flexion 30 / extension 10; opposite joint not measured (advisory)
    v = _adequate_base(1, 30, 10)
    v.update(
        {
            "flare_ups": "Yes",
            "flare_ups_desc": "Flares 2-3 times per month, each lasting a day; unable to climb stairs.",
            "flare_functional_loss": "Yes",
            "flare_flexion_est": "30",
            "flare_extension_est": "10",
            "opposite_flexion": "",
            "opposite_extension": "",
            "remarks": f"Veteran SSN {IDENTITIES[1][2]} confirmed at check-in. Wears a soft brace for walking.",
        }
    )
    cases.append(
        _case(
            "knee_02",
            v,
            adequate=True,
            ratings={"5260": 20, "5261": 10},
            gaps=["correia_opposite_joint"],
        )
    )

    # 03 — noncompensable ROM but painful motion -> 4.59 minimum
    v = _adequate_base(2, 90, 0)
    cases.append(
        _case(
            "knee_03",
            v,
            adequate=True,
            ratings={"5260": 10, "5261": 0},
            painful_motion_minimum=True,
        )
    )

    # 04 — repetitive use is the limiting measurement (DeLuca)
    v = _adequate_base(3, 60, 5)
    v.update(
        {
            "rom_flexion_post_rep": "45",
            "rom_extension_post_rep": "5",
            "rep_use_additional_loss": "Yes",
            "factor_fatigability": "Yes",
            "remarks": "Additional loss of 15 degrees of flexion after three repetitions due to pain and fatigability.",
        }
    )
    cases.append(
        _case(
            "knee_04",
            v,
            adequate=True,
            ratings={"5260": 10, "5261": 0},
            limiting_source={"5260": "post_rep"},
            consider_higher=["5261"],  # extension 5° is 5° from the 10% tier, with fatigability
        )
    )

    # 05 — flare estimate only in the remarks (free-text path feeds the rules)
    v = _adequate_base(4, 120, 0)
    v.update(
        {
            "flare_ups": "Yes",
            "flare_ups_desc": "Daily flares after standing more than an hour, lasting several hours; severe.",
            "flare_functional_loss": "Yes",
            "flare_flexion_est": "",
            "flare_extension_est": "",
            "remarks": (
                f"Per the history provided by {IDENTITIES[4][0]}, during flare-ups flexion is "
                "estimated to be limited to 30 degrees; extension is not additionally limited."
            ),
        }
    )
    cases.append(
        _case(
            "knee_05",
            v,
            adequate=True,
            ratings={"5260": 20, "5261": 0},
            limiting_source={"5260": "flare_estimate"},
        )
    )

    # 06 — extension-limited knee with arthritis on imaging; pain/function not addressed (advisory)
    v = _adequate_base(5, 140, 20)
    v.update(
        {
            "pain_causes_functional_loss": "NotAddressed",
            "imaging_arthritis": "Yes",
            "remarks": "Radiographs show moderate medial compartment degenerative changes.",
        }
    )
    cases.append(
        _case(
            "knee_06",
            v,
            adequate=True,
            ratings={"5260": 0, "5261": 30},
            gaps=["mitchell_pain_function"],
            notes=["dc5003_consider"],
        )
    )

    # 07 — INADEQUATE: passive ROM pain not tested, no reason (Correia)
    v = _adequate_base(6, 45, 0)
    v.update({"pain_passive_rom": "NotTested", "passive_not_tested_reason": ""})
    cases.append(
        _case(
            "knee_07",
            v,
            adequate=False,
            ratings={"5260": 10, "5261": 0},
            gaps=["correia_passive"],
        )
    )

    # 08 — INADEQUATE: flare-ups reported, no estimate, unexplained speculation (Sharp / Jones)
    v = _adequate_base(7, 60, 10)
    v.update(
        {
            "flare_ups": "Yes",
            "flare_ups_desc": "Flares several times a week with swelling; cannot kneel or squat during flares.",
            "flare_functional_loss": "Unable",
            "flare_flexion_est": "",
            "flare_extension_est": "",
            "flare_no_estimate_reason": "Unable to say without resorting to mere speculation.",
            "remarks": f"Ms. {IDENTITIES[7][0].split()[-1]} was not examined during a flare-up.",
        }
    )
    cases.append(
        _case(
            "knee_08",
            v,
            adequate=False,
            ratings={"5260": 0, "5261": 10},
            gaps=["sharp_flare_estimate"],
        )
    )

    # 09 — INADEQUATE: repetitive use not performed without reason (DeLuca) + bare opinion
    v = _adequate_base(8, 45, 0)
    v.update(
        {
            "rep_use_performed": "No",
            "rep_use_not_performed_reason": "",
            "rom_flexion_post_rep": "",
            "rom_extension_post_rep": "",
            "rep_use_additional_loss": "",
            "opinion_rationale": "Yes.",
        }
    )
    cases.append(
        _case(
            "knee_09",
            v,
            adequate=False,
            ratings={"5260": 10, "5261": 0},
            gaps=["deluca_rep_use", "opinion_rationale"],
        )
    )

    # 10 — near a tier boundary with functional-loss factors beyond pain (4.7 consider flag)
    v = _adequate_base(9, 33, 0)
    v.update(
        {
            "flare_ups": "Yes",
            "flare_ups_desc": "Flares weekly, one day each; marked weakness and fatigue with use.",
            "flare_functional_loss": "Yes",
            "flare_flexion_est": "33",
            "flare_extension_est": "0",
            "factor_weakness": "Yes",
            "factor_fatigability": "Yes",
            "strength_flexion": "4/5",
            "remarks": "Weakened movement and excess fatigability observed on examination.",
        }
    )
    cases.append(
        _case(
            "knee_10",
            v,
            adequate=True,
            ratings={"5260": 10, "5261": 0},
            consider_higher=["5260"],
        )
    )
    return cases


def write_cases(cases_dir: Path | None = None) -> list[Path]:
    """Build every case PDF and cases/truth.json. Returns the PDF paths."""
    cases_dir = cases_dir or DEFAULT_CASES_DIR
    cases_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    truth: dict[str, Any] = {}
    for case in all_cases():
        cid = str(case["case_id"])
        pdf = build_pdf(KneeTruth(case_id=cid, values=case["values"]), cases_dir / f"{cid}.pdf")
        paths.append(pdf)
        truth[cid] = {"values": case["values"], "expected": case["expected"]}
    (cases_dir / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    return paths


def load_truth(cases_dir: Path | None = None) -> dict[str, Any]:
    cases_dir = cases_dir or DEFAULT_CASES_DIR
    return json.loads((cases_dir / "truth.json").read_text())
