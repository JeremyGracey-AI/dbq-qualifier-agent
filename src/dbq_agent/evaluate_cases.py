"""Evaluate the pipeline against cases/truth.json.

Metrics (all computed from verified claims only):
- tier precision/recall over (case, dc, pct) tuples
- adequacy accuracy
- gap precision/recall over (case, rule_id)
- consider-flag precision/recall over (case, dc)
- citation faithfulness: share of claims whose every quote is verbatim in the document
  and every citation resolves in the KB (the verify step guarantees 1.0 for verified claims;
  we report it over all emitted claims, dropped included, so a regressing extractor shows up)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from dbq_agent.pipeline import Context, run
from dbq_agent.report import build_report
from dbq_agent.steps.verify import resolve_span


@dataclass
class PR:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, predicted: set[Any], expected: set[Any]) -> None:
        self.tp += len(predicted & expected)
        self.fp += len(predicted - expected)
        self.fn += len(expected - predicted)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 1.0


@dataclass
class EvalResult:
    tiers: PR = field(default_factory=PR)
    gaps: PR = field(default_factory=PR)
    consider: PR = field(default_factory=PR)
    adequacy_correct: int = 0
    cases: int = 0
    faithful_claims: int = 0
    total_claims: int = 0
    per_case: list[dict[str, Any]] = field(default_factory=list)

    def table(self) -> str:
        rows = [
            ("tier P / R", f"{self.tiers.precision:.2f} / {self.tiers.recall:.2f}"),
            ("adequacy accuracy", f"{self.adequacy_correct}/{self.cases}"),
            ("gap P / R", f"{self.gaps.precision:.2f} / {self.gaps.recall:.2f}"),
            ("consider-flag P / R", f"{self.consider.precision:.2f} / {self.consider.recall:.2f}"),
            (
                "citation faithfulness",
                f"{self.faithful_claims}/{self.total_claims} claims "
                f"({(self.faithful_claims / self.total_claims if self.total_claims else 1.0):.2f})",
            ),
        ]
        width = max(len(k) for k, _ in rows)
        lines = [f"{k.ljust(width)}  {v}" for k, v in rows]
        lines.append("")
        lines.append(
            f"{'case'.ljust(8)} {'ok'.ljust(4)} {'adequate'.ljust(9)} ratings            gaps"
        )
        for c in self.per_case:
            lines.append(
                f"{c['case_id'].ljust(8)} {('yes' if c['ok'] else 'NO').ljust(4)} "
                f"{str(c['adequate']).ljust(9)} {str(c['ratings']).ljust(18)} {','.join(c['gaps']) or '-'}"
            )
        return "\n".join(lines)


def evaluate_cases(
    cases_dir: Path, truth: dict[str, Any], claim_date: date, ctx: Context
) -> EvalResult:
    res = EvalResult()
    for cid, entry in sorted(truth.items()):
        exp = entry["expected"]
        state = run(cases_dir / f"{cid}.pdf", claim_date, ctx)
        report = build_report(state, ctx.kb)
        got_r = {dc: line.pct for dc, line in report.provisional.items()}
        got_gaps = {g.rule_id for g in report.gaps if g.rule_id}
        got_consider = {c.dc for c in report.consider if c.dc}
        res.tiers.add(
            {(cid, dc, p) for dc, p in got_r.items()},
            {(cid, dc, p) for dc, p in exp["ratings"].items()},
        )
        res.gaps.add({(cid, g) for g in got_gaps}, {(cid, g) for g in exp["gaps"]})
        res.consider.add(
            {(cid, d) for d in got_consider}, {(cid, d) for d in exp["consider_higher"]}
        )
        res.adequacy_correct += int(report.adequate is exp["adequate"])
        res.cases += 1
        for c in state.all_claims():
            res.total_claims += 1
            ok_spans = bool(c.evidence) and all(
                resolve_span(s, state.doc, state.deid) is None for s in c.evidence
            )
            ok_cites = bool(c.citations) and all(ctx.kb.has_authority(x) for x in c.citations)
            res.faithful_claims += int(ok_spans and ok_cites)
        ok = (
            got_r == exp["ratings"]
            and report.adequate is exp["adequate"]
            and got_gaps == set(exp["gaps"])
            and got_consider == set(exp["consider_higher"])
        )
        res.per_case.append(
            {
                "case_id": cid,
                "ok": ok,
                "adequate": report.adequate,
                "ratings": got_r,
                "gaps": sorted(got_gaps),
            }
        )
    return res
