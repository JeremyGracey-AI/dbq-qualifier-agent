# Copilot instructions

Follow `AGENTS.md` at the repo root. It is the single source of instructions for every coding
agent in this repo (Claude Code, Codex, Copilot, Cursor). Key points:

- PHI never reaches an LLM un-redacted; identity fields never leave the process.
- Rating numbers come from deterministic rules, never from model output.
- Every claim needs evidence spans and KB citations; the verify step drops the rest.
- Synthetic data only. Never add real DBQs or veteran data.
- Before saying a task is done: `uv run ruff check .`, `uv run pyright`, `uv run pytest`.
