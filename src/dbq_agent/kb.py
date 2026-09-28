"""Qualifiers knowledge base: rating criteria rows, adequacy rules, and citable authorities.

Retrieval is deliberately structured for the slice: criteria are filtered by diagnostic code and
the date the criteria were in effect (rating criteria change; the version in effect on the claim
date controls), and authorities/rules are ranked by tag overlap plus a small lexical score.
A dense index can replace `_lexical_score` later without touching the callers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from dbq_agent.models import AdequacyRule, Authority, Criterion, RetrievedContext

DEFAULT_KB_DIR = Path(__file__).resolve().parents[2] / "data" / "kb"

_WORD = re.compile(r"[a-z0-9.§]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


@dataclass
class KnowledgeBase:
    criteria: list[Criterion]
    authorities: list[Authority]
    rules: list[AdequacyRule]
    version: str = ""
    _auth_index: dict[str, Authority] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._auth_index = {a.id: a for a in self.authorities}

    # ---- loading -------------------------------------------------------------------

    @classmethod
    def load(cls, kb_dir: Path | None = None) -> KnowledgeBase:
        kb_dir = kb_dir or DEFAULT_KB_DIR
        crit_raw = json.loads((kb_dir / "criteria.json").read_text())
        auth_raw = json.loads((kb_dir / "authorities.json").read_text())
        rule_raw = json.loads((kb_dir / "adequacy.json").read_text())
        kb = cls(
            criteria=[Criterion.model_validate(c) for c in crit_raw["criteria"]],
            authorities=[Authority.model_validate(a) for a in auth_raw["authorities"]],
            rules=[AdequacyRule.model_validate(r) for r in rule_raw["rules"]],
            version=str(crit_raw.get("version", "")),
        )
        kb.validate()
        return kb

    def validate(self) -> None:
        """Every rule must cite authorities that exist; criteria ids must be unique."""
        ids = [c.id for c in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate criterion ids in criteria.json")
        for rule in self.rules:
            missing = [a for a in rule.authorities if a not in self._auth_index]
            if missing:
                raise ValueError(f"rule {rule.id} cites unknown authorities: {missing}")

    # ---- lookups -------------------------------------------------------------------

    def authority(self, auth_id: str) -> Authority | None:
        return self._auth_index.get(auth_id)

    def has_authority(self, auth_id: str) -> bool:
        return auth_id in self._auth_index

    def covered_dcs(self) -> set[str]:
        return {c.dc for c in self.criteria}

    def criteria_for(self, dc: str, as_of: date) -> list[Criterion]:
        """Criteria rows for `dc` in effect on `as_of`, highest percentage first."""
        rows = [
            c
            for c in self.criteria
            if c.dc == dc
            and c.effective_from <= as_of
            and (c.effective_to is None or as_of < c.effective_to)
        ]
        return sorted(rows, key=lambda c: c.pct, reverse=True)

    # ---- retrieval -----------------------------------------------------------------

    def retrieve(
        self,
        dcs: list[str],
        as_of: date,
        tags: list[str] | None = None,
        query_text: str = "",
        extra_ids: list[str] | None = None,
        top_k_authorities: int = 12,
        iteration: int = 0,
    ) -> RetrievedContext:
        """Structured + lexical retrieval.

        - criteria: exact filter on dc + effective date
        - rules: any tag overlap with `tags`
        - authorities: ranked by tag overlap and lexical overlap with `query_text`,
          plus everything cited by the selected rules and criteria, plus `extra_ids`
          (used by the verify step's bounded re-retrieval).
        """
        tags = tags or []
        tag_set = set(tags)
        criteria = [c for dc in dcs for c in self.criteria_for(dc, as_of)]

        rules = [r for r in self.rules if tag_set & set(r.tags)] if tag_set else list(self.rules)

        must_have: set[str] = set(extra_ids or [])
        for r in rules:
            must_have.update(r.authorities)
        for c in criteria:
            must_have.add(f"cfr-4.71a-{c.dc}")

        # must-have authorities are appended in full; the top-k cut applies only to the
        # organically ranked remainder, so adding an id never evicts another.
        q_tokens = _tokens(query_text) | tag_set
        scored: list[tuple[float, Authority]] = []
        for a in self.authorities:
            if a.id in must_have:
                continue
            score = float(len(tag_set & set(a.tags))) * 2.0 + self._lexical_score(a, q_tokens)
            if score > 0:
                scored.append((score, a))
        scored.sort(key=lambda t: (-t[0], t[1].id))
        picked = [self._auth_index[aid] for aid in sorted(must_have) if aid in self._auth_index]
        picked += [a for _, a in scored[:top_k_authorities]]

        return RetrievedContext(
            criteria=criteria,
            authorities=picked,
            rules=rules,
            query={"dcs": dcs, "as_of": as_of.isoformat(), "tags": tags},
            iteration=iteration,
        )

    @staticmethod
    def _lexical_score(a: Authority, q_tokens: set[str]) -> float:
        if not q_tokens:
            return 0.0
        doc_tokens = _tokens(f"{a.title} {a.summary} {a.cite}")
        overlap = len(q_tokens & doc_tokens)
        return overlap / (1.0 + len(doc_tokens) ** 0.5)
