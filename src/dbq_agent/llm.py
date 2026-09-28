"""LLM client boundary.

The pipeline only ever calls `LLMClient.extract_json(system, user, schema)`. Everything that
reaches `user` has already passed the PHI gate. Two implementations ship:

- `AnthropicClient` — real calls via forced tool use, so the reply is schema-shaped JSON.
- `CapturingClient` — test double that records every payload (so tests can assert no PHI
  crossed) and returns canned output.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_MODEL = os.environ.get("DBQ_AGENT_MODEL", "claude-sonnet-5")


class LLMClient(Protocol):
    name: str

    def extract_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class CapturingClient:
    """Records payloads; returns `canned` (or `{}`) for every call."""

    canned: dict[str, Any] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)
    name: str = "capturing"

    def extract_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"system": system, "user": user, "schema": schema})
        return json.loads(json.dumps(self.canned))

    def payload_text(self) -> str:
        return "\n".join(f"{c['system']}\n{c['user']}" for c in self.calls)


@dataclass
class AnthropicClient:
    model: str = DEFAULT_MODEL
    max_tokens: int = 2048
    name: str = "anthropic"

    def extract_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        import anthropic

        client = anthropic.Anthropic()
        tool = {
            "name": "record_findings",
            "description": "Record the extracted findings. Quote spans verbatim.",
            "input_schema": schema,
        }
        msg = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=[tool],  # type: ignore[arg-type]
            tool_choice={"type": "tool", "name": "record_findings"},
            messages=[{"role": "user", "content": user}],
        )
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use":
                data = getattr(block, "input", {})
                return dict(data) if isinstance(data, dict) else {}
        return {}

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
