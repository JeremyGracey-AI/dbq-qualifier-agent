"""Map findings to rating criteria. Deterministic; no model involved.

Policy choices (documented so a reviewer can disagree with them):

- Effective ROM is the most limiting of initial, post-repetitive-use and flare-up estimate
  (§§ 4.40, 4.45; DeLuca). A flare estimate written only in prose is used when the structured
  field is blank, and the claim's evidence then points at that prose.
- A measurement that meets no listed criterion is 0% under § 4.31.
- Painful motion (§ 4.59; Burton; Southall-Norman) yields the minimum compensable rating for
  the joint once, under DC 5260, and only when neither ROM code is already compensable.
- § 4.7 is never applied automatically. When the effective ROM is within `NEAR_TIER_DEG` of the
  next tier and the exam documents functional-loss factors beyond pain (weakness, fatigability,
  incoordination), a `consider_higher` claim asks the reviewer to weigh it.
"""

from __future__ import annotations

from dbq_agent.models import (
    Claim,
    Criterion,
    Evaluation,
    KneeFindings,
    Measure,
    RetrievedContext,
    RomSet,
    RomSource,
    Span,
)

NEAR_TIER_DEG = 5
FACTORS_BEYOND_PAIN = ("weakness", "fatigability", "incoordination")


def select_tier(rows: list[Criterion], measured: int) -> tuple[Criterion | None, int]:
    """Highest-percentage row whose threshold the measurement meets, else (None, 0)."""
    best: Criterion | None = None
    for c in sorted(rows, key=lambda r: r.pct, reverse=True):
        if c.kind != "rom_threshold" or c.threshold_deg is None:
            continue
        met = measured <= c.threshold_deg if c.op == "<=" else measured >= c.threshold_deg
        if met:
            best = c
            break
    return best, (best.pct if best else 0)


def _is_worse(measure: Measure, candidate: int, current: int) -> bool:
    return candidate < current if measure == "flexion" else candidate > current


def effective_rom(
    findings: KneeFindings, measure: Measure
) -> tuple[int, RomSource, list[Span]] | None:
    """Most limiting measurement across sources, with the spans that support it."""
    sets: list[RomSet] = [findings.initial]
    if findings.post_rep is not None:
        sets.append(findings.post_rep)
    if findings.flare_estimate is not None:
        sets.append(findings.flare_estimate)
    else:
        kind = "flare_flexion_estimate" if measure == "flexion" else "flare_extension_estimate"
        for item in findings.free_text.items:
            if item.kind == kind and isinstance(item.value, int):
                sets.append(
                    RomSet(source="flare_estimate", **{measure: item.value}, spans=[item.span])
                )
    best: tuple[int, RomSource, list[Span]] | None = None
    for s in sets:
        value = getattr(s, measure)
        if value is None:
            continue
        if best is None or _is_worse(measure, value, best[0]):
            own = [sp for sp in s.spans if measure in sp.field] or list(s.spans)
            best = (value, s.source, own)
    return best


def _factors_beyond_pain(findings: KneeFindings) -> list[str]:
    found = [
        f for f in findings.functional_loss_factors if any(k in f for k in FACTORS_BEYOND_PAIN)
    ]
    found += [
        str(i.value)
        for i in findings.free_text.items
        if i.kind == "functional_loss_factor" and i.value in FACTORS_BEYOND_PAIN
    ]
    return sorted(set(found))


def _factor_spans(findings: KneeFindings) -> list[Span]:
    spans = [
        s
        for k, s in findings.spans.items()
        if k.startswith("factor_") and any(f in k for f in FACTORS_BEYOND_PAIN)
    ]
    spans += [
        i.span
        for i in findings.free_text.items
        if i.kind == "functional_loss_factor" and i.value in FACTORS_BEYOND_PAIN
    ]
    return spans


def evaluate(findings: KneeFindings, retrieved: RetrievedContext) -> Evaluation:
    claims: list[Claim] = []
    factor_spans = _factor_spans(findings)
    ratings: dict[str, int] = {}
    by_dc: dict[str, list[Criterion]] = {}
    for c in retrieved.criteria:
        by_dc.setdefault(c.dc, []).append(c)

    for dc, rows in sorted(by_dc.items()):
        if rows[0].kind == "predicate":
            claims.extend(_evaluate_predicate_dc(dc, rows, findings, ratings))
            continue
        measure: Measure = rows[0].measure or "flexion"
        eff = effective_rom(findings, measure)
        if eff is None:
            continue
        value, source, spans = eff
        if not spans and findings.initial.spans:
            spans = list(findings.initial.spans)
        row, pct = select_tier(rows, value)
        ratings[dc] = pct
        source_note = {
            "initial": "initial measurement",
            "post_rep": "after repetitive-use testing",
            "flare_estimate": "examiner's flare-up estimate",
        }[source]
        cites = [f"cfr-4.71a-{dc}", "cfr-4.71-plate-ii"]
        if source != "initial":
            cites += ["cfr-4.40", "cfr-4.45", "deluca-1995"]
        if row is None:
            cites.append("cfr-4.31")
            statement = (
                f"DC {dc}: {measure} to {value}° ({source_note}) meets no listed criterion; "
                f"0% under § 4.31."
            )
        else:
            statement = (
                f"DC {dc}: {measure} to {value}° ({source_note}) meets '{row.text}' → {pct}%."
            )
        claims.append(
            Claim(
                id=f"rating-{dc}",
                kind="rating_tier",
                dc=dc,
                pct=pct,
                statement=statement,
                evidence=spans,
                citations=cites,
            )
        )

        # § 4.7 consideration: near the next tier with functional-loss factors beyond pain
        higher = [r for r in rows if r.pct > pct]
        if higher:
            nxt = min(higher, key=lambda r: r.pct)
            nxt_threshold = nxt.threshold_deg or 0
            gap_deg = (value - nxt_threshold) if measure == "flexion" else (nxt_threshold - value)
            factors = _factors_beyond_pain(findings)
            if 0 < gap_deg <= NEAR_TIER_DEG and factors:
                claims.append(
                    Claim(
                        id=f"consider-{dc}",
                        kind="consider_higher",
                        dc=dc,
                        pct=nxt.pct,
                        statement=(
                            f"DC {dc}: {measure} to {value}° is within {gap_deg}° of '{nxt.text}' "
                            f"({nxt.pct}%) and the exam documents {', '.join(factors)}; consider "
                            f"whether the disability picture more nearly approximates the higher tier."
                        ),
                        evidence=spans + factor_spans,
                        citations=["cfr-4.7", "cfr-4.40", "cfr-4.45", "deluca-1995"],
                    )
                )

    # § 4.59 painful motion minimum
    rom_ratings = {dc: p for dc, p in ratings.items() if dc in ("5260", "5261")}
    if findings.pain_on_motion and rom_ratings and all(p == 0 for p in rom_ratings.values()):
        dc = "5260" if "5260" in rom_ratings else sorted(rom_ratings)[0]
        ratings[dc] = max(ratings[dc], 10)
        evidence = [
            s
            for k, s in findings.spans.items()
            if k in ("pain_noted_on_exam", "pain_causes_functional_loss")
        ]
        claims.append(
            Claim(
                id=f"painful-motion-{dc}",
                kind="painful_motion_minimum",
                dc=dc,
                pct=10,
                statement=(
                    f"Painful motion documented with noncompensable limitation of motion; "
                    f"minimum compensable rating (10%) under § 4.59 applied to DC {dc}."
                ),
                evidence=evidence,
                citations=["cfr-4.59", "burton-2011", "southall-norman-2016"],
            )
        )

    if len([p for dc, p in ratings.items() if p > 0 and dc in ("5260", "5261")]) >= 2:
        claims.append(
            Claim(
                id="note-separate-ratings",
                kind="note",
                statement=(
                    "Limitation of flexion and limitation of extension of the same knee are rated "
                    "separately and combined under § 4.25."
                ),
                evidence=list(findings.initial.spans),
                citations=["vaopgcprec-9-2004", "cfr-4.25"],
            )
        )

    for dc, rows in sorted(by_dc.items()):
        superseded = [r for r in rows if r.kind == "predicate" and r.effective_to is not None]
        if dc in ratings and superseded:
            changed = max(r.effective_to for r in superseded if r.effective_to is not None)
            inst = findings.instability_findings
            claims.append(
                Claim(
                    id=f"note-version-{dc}",
                    kind="note",
                    statement=(
                        f"DC {dc} was rated under the criteria in effect on the claim date; the "
                        f"criteria changed on {changed.isoformat()}. For a claim pending across "
                        f"that date VA applies the earlier text before it and the more favorable "
                        f"text from it, so evaluate both versions."
                    ),
                    evidence=list(inst.spans.values())[:2]
                    if inst
                    else list(findings.initial.spans),
                    citations=["fr-2020-25450", f"cfr-4.71a-{dc}"],
                )
            )

    if "5257" in ratings and any(p > 0 for dc, p in ratings.items() if dc != "5257"):
        inst = findings.instability_findings
        claims.append(
            Claim(
                id="note-separate-instability",
                kind="note",
                statement=(
                    "Instability (DC 5257) and limitation of motion of the same knee are rated "
                    "separately when the limitation of motion is at least noncompensable or "
                    "painful; combine under § 4.25."
                ),
                evidence=list(inst.spans.values())[:2] if inst else list(findings.initial.spans),
                citations=["vaopgcprec-23-97", "vaopgcprec-9-98", "cfr-4.25"],
            )
        )

    return Evaluation(claims=claims, ratings=ratings)


def select_predicate_tier(
    rows: list[Criterion], facts: dict[str, str]
) -> tuple[Criterion | None, dict[str, list[str]] | None]:
    """Highest-percentage predicate row with an alternative every fact of which is satisfied."""
    for c in sorted(rows, key=lambda r: r.pct, reverse=True):
        if c.kind != "predicate" or not c.requires:
            continue
        for alternative in c.requires:
            if all(facts.get(fact) in allowed for fact, allowed in alternative.items()):
                return c, alternative
    return None, None


def _evaluate_predicate_dc(
    dc: str, rows: list[Criterion], findings: KneeFindings, ratings: dict[str, int]
) -> list[Claim]:
    inst = findings.instability_findings
    if inst is None or not inst.any_instability():
        return []
    facts = inst.facts()
    row, alternative = select_predicate_tier(rows, facts)
    version = rows[0].cite
    if row is None or alternative is None:
        # Instability is documented but no level's requirements are met as documented; the
        # gap check reports what is missing. Nothing is rated.
        return []
    used_fields = list(alternative)
    if "severity" in used_fields:
        used_fields = [f for f in used_fields if f != "severity"] + [
            "subluxation_history",
            "lateral_instability_history",
        ]
    evidence = [inst.spans[f] for f in used_fields if f in inst.spans]
    ratings[dc] = row.pct
    return [
        Claim(
            id=f"rating-{dc}",
            kind="rating_tier",
            dc=dc,
            pct=row.pct,
            statement=(
                f"DC {dc} ({row.subtable}): the exam documents "
                + ", ".join(f"{k}={facts[k]}" for k in alternative)
                + f", meeting '{row.text[:90]}…' → {row.pct}% [{version}]."
            ),
            evidence=evidence,
            citations=[f"cfr-4.71a-{dc}"]
            + (["fr-2020-25450"] if row.effective_from.year >= 2021 else []),
        )
    ]
