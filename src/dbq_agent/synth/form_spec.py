"""Field specification for the synthetic knee DBQ.

Mirrors the section structure of VA Form 21-0960M-9 (Knee and Lower Leg Conditions DBQ).
Field names are this project's own; when the real fillable form is available, add a mapping
from its AcroForm names to these and nothing downstream changes.

Radio option values are single tokens (no spaces) so they round-trip cleanly as PDF names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

WidgetKind = Literal["text", "textarea", "radio", "checkbox", "heading", "note"]


@dataclass(frozen=True)
class Widget:
    name: str
    kind: WidgetKind
    label: str = ""
    options: tuple[tuple[str, str], ...] = field(default_factory=tuple)  # (value, label)
    width: int = 220


YN = (("Yes", "Yes"), ("No", "No"))
YN_NT = (("Yes", "Yes"), ("No", "No"), ("NotTested", "Not tested"))
YN_NA = (("Yes", "Yes"), ("No", "No"), ("NotAddressed", "Not addressed"))
STAB = (
    ("Normal", "Normal"),
    ("1plus", "1+"),
    ("2plus", "2+"),
    ("3plus", "3+"),
    ("NotTested", "Not tested"),
)


def H(text: str) -> Widget:
    return Widget(name=f"_h_{abs(hash(text))}", kind="heading", label=text)


def N(text: str) -> Widget:
    return Widget(name=f"_n_{abs(hash(text))}", kind="note", label=text)


IDENTITY_FIELDS: tuple[str, ...] = ("vet_name", "vet_dob", "vet_ssn", "exam_date", "examiner_name")

FREE_TEXT_FIELDS: tuple[str, ...] = (
    "dx_1",
    "history",
    "flare_ups_desc",
    "functional_loss_desc",
    "passive_not_tested_reason",
    "rep_use_not_performed_reason",
    "flare_no_estimate_reason",
    "functional_impact",
    "remarks",
    "opinion_rationale",
)

FUNCTIONAL_LOSS_FACTORS: tuple[tuple[str, str], ...] = (
    ("factor_pain", "Pain on movement"),
    ("factor_weakness", "Weakened movement"),
    ("factor_fatigability", "Excess fatigability"),
    ("factor_incoordination", "Incoordination"),
    ("factor_less_movement", "Less movement than normal"),
    ("factor_swelling", "Swelling"),
    ("factor_atrophy", "Atrophy of disuse"),
    ("factor_instability", "Instability of station"),
)

PAGES: list[list[Widget]] = [
    # ---------------------------------------------------------------- page 1
    [
        H("Knee and Lower Leg Conditions Disability Benefits Questionnaire"),
        N(
            "SYNTHETIC REPLICA for software testing. Structure follows VA Form 21-0960M-9. All data fictional."
        ),
        Widget("form_number", "text", "Form number", width=120),
        Widget("vet_name", "text", "Veteran name"),
        Widget("vet_dob", "text", "Date of birth", width=120),
        Widget("vet_ssn", "text", "SSN", width=120),
        Widget("exam_date", "text", "Date of examination", width=120),
        Widget("examiner_name", "text", "Examiner"),
        Widget("claimed_side", "radio", "Knee examined", (("Right", "Right"), ("Left", "Left"))),
        H("Section 1. Diagnosis"),
        Widget("dx_1", "text", "Diagnosis #1", width=300),
        Widget("dx_icd_1", "text", "ICD code", width=100),
        H("Section 2. Medical history"),
        Widget("history", "textarea", "History (onset, course, treatment)"),
        H("Section 3. Flare-ups"),
        Widget("flare_ups", "radio", "Does the Veteran report flare-ups?", YN),
        Widget(
            "flare_ups_desc",
            "textarea",
            "Describe (frequency, duration, severity, functional loss)",
        ),
        H("Section 4. Functional loss or impairment"),
        Widget("functional_loss", "radio", "Functional loss or impairment reported?", YN),
        Widget("functional_loss_desc", "textarea", "Describe"),
    ],
    # ---------------------------------------------------------------- page 2
    [
        H("Section 5. Initial range of motion (ROM) measurements"),
        N(
            "Normal knee ROM: flexion 0 to 140 degrees; extension 140 to 0 degrees (38 CFR 4.71, Plate II)."
        ),
        Widget("rom_flexion_initial", "text", "Flexion (0-140), degrees", width=80),
        Widget("rom_extension_initial", "text", "Extension (140-0), degrees short of 0", width=80),
        Widget("rom_abnormal", "radio", "Is ROM abnormal or outside normal range?", YN),
        Widget(
            "rom_contributes_functional_loss",
            "radio",
            "If abnormal, does ROM contribute to functional loss?",
            YN_NA,
        ),
        Widget("pain_noted_on_exam", "radio", "Is pain noted on examination (on ROM)?", YN),
        Widget(
            "pain_causes_functional_loss", "radio", "Does the pain cause functional loss?", YN_NA
        ),
        Widget("pain_weight_bearing", "radio", "Evidence of pain with weight-bearing?", YN_NT),
        Widget("pain_non_weight_bearing", "radio", "Evidence of pain in nonweight-bearing?", YN_NT),
        Widget("pain_passive_rom", "radio", "Evidence of pain on passive ROM testing?", YN_NT),
        Widget("passive_not_tested_reason", "text", "If not tested, explain", width=300),
        Widget(
            "tenderness",
            "radio",
            "Objective evidence of localized tenderness or pain on palpation?",
            YN,
        ),
        Widget("crepitus", "radio", "Objective evidence of crepitus?", YN),
        Widget("opposite_joint_undamaged", "radio", "Is the opposing joint undamaged?", YN),
        Widget("opposite_flexion", "text", "Opposing joint flexion, degrees", width=80),
        Widget("opposite_extension", "text", "Opposing joint extension, degrees", width=80),
        H("Section 6. Repetitive-use testing"),
        Widget(
            "rep_use_performed",
            "radio",
            "Was repetitive-use testing (3 repetitions) performed?",
            YN,
        ),
        Widget("rep_use_not_performed_reason", "text", "If not performed, explain", width=300),
        Widget("rom_flexion_post_rep", "text", "Flexion after 3 repetitions, degrees", width=80),
        Widget(
            "rom_extension_post_rep", "text", "Extension after 3 repetitions, degrees", width=80
        ),
        Widget(
            "rep_use_additional_loss",
            "radio",
            "Additional loss of function or ROM after repetitions?",
            YN,
        ),
    ],
    # ---------------------------------------------------------------- page 3
    [
        H("Section 7. Flare-ups and repeated use over time"),
        Widget(
            "exam_during_flare",
            "radio",
            "Is the examination being conducted during a flare-up?",
            YN,
        ),
        Widget(
            "flare_functional_loss",
            "radio",
            "Do pain, weakness, fatigability or incoordination significantly limit functional ability during flare-ups?",
            (("Yes", "Yes"), ("No", "No"), ("Unable", "Unable to say without mere speculation")),
        ),
        Widget(
            "flare_flexion_est", "text", "Estimated flexion during flare-ups, degrees", width=80
        ),
        Widget(
            "flare_extension_est", "text", "Estimated extension during flare-ups, degrees", width=80
        ),
        Widget("flare_no_estimate_reason", "text", "If unable to estimate, explain", width=300),
        N("Contributing factors of disability (check all that apply):"),
        *[Widget(name, "checkbox", label) for name, label in FUNCTIONAL_LOSS_FACTORS],
        H("Section 9. Muscle strength testing"),
        Widget("strength_flexion", "text", "Knee flexion strength (0/5 - 5/5)", width=60),
        Widget("strength_extension", "text", "Knee extension strength (0/5 - 5/5)", width=60),
        Widget("atrophy", "radio", "Muscle atrophy?", YN),
        H("Section 10. Ankylosis"),
        Widget("ankylosis", "radio", "Ankylosis of the knee?", YN),
    ],
    # ---------------------------------------------------------------- page 4
    [
        H("Section 11. Joint stability tests"),
        Widget("test_lachman", "radio", "Anterior instability (Lachman test)", STAB),
        Widget(
            "test_posterior_drawer", "radio", "Posterior instability (posterior drawer test)", STAB
        ),
        Widget("test_medial", "radio", "Medial instability (valgus pressure)", STAB),
        Widget("test_lateral", "radio", "Lateral instability (varus pressure)", STAB),
        Widget(
            "subluxation_history",
            "radio",
            "History of recurrent subluxation",
            (
                ("None", "None"),
                ("Slight", "Slight"),
                ("Moderate", "Moderate"),
                ("Severe", "Severe"),
            ),
        ),
        H("Section 18. Diagnostic testing"),
        Widget("imaging_performed", "radio", "Imaging studies performed and reviewed?", YN),
        Widget("imaging_arthritis", "radio", "Degenerative or traumatic arthritis documented?", YN),
        H("Section 19. Functional impact"),
        Widget("functional_impact", "textarea", "Impact on ability to work"),
        H("Section 20. Remarks"),
        Widget("remarks", "textarea", "Remarks, if any"),
    ],
    # ---------------------------------------------------------------- page 5
    [
        H("Medical opinion (if requested)"),
        Widget("opinion_requested", "radio", "Was a medical opinion requested?", YN),
        Widget(
            "opinion_conclusion",
            "radio",
            "Opinion",
            (
                ("AtLeastAsLikely", "At least as likely as not (50% or greater probability)"),
                ("LessLikely", "Less likely than not (less than 50% probability)"),
                ("CannotDetermine", "Cannot be determined without resorting to mere speculation"),
            ),
        ),
        Widget("opinion_rationale", "textarea", "Rationale"),
    ],
]

ALL_WIDGETS: dict[str, Widget] = {
    w.name: w for page in PAGES for w in page if w.kind not in ("heading", "note")
}
PAGE_OF: dict[str, int] = {
    w.name: i + 1 for i, page in enumerate(PAGES) for w in page if w.kind not in ("heading", "note")
}
