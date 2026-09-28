# Current task

## Goal
Smallest working slice of the DBQ → 38 CFR qualifier agent: knee DBQ, DC 5260/5261, adequacy
gaps (Correia / DeLuca / Sharp / Mitchell / opinion rationale), verifying critic, report.

## Current state
- Done (2026-09-27): scaffold, KB (criteria + adequacy + authorities), synthetic form generator
  and 10 cases, ingest with page provenance, PHI gate (known values + patterns + phi-scrub),
  heuristic and Claude free-text extractors, classify → retrieve → evaluate → gap check → verify
  loop, JSON/markdown report, CLI (`run | eval | synth`), eval harness, CI workflow.
- Verified: ruff clean, pyright 0 errors, 23 tests green, `dbq-agent eval` 10/10 cases with
  citation faithfulness 31/31.
- Not yet exercised: `--llm anthropic` against the real API (no key in the build environment);
  the client uses forced tool use and validates every quote, but run it once before relying on it.

## Next step
Run `ANTHROPIC_API_KEY=... uv run dbq-agent run cases/knee_05.pdf --llm anthropic --claim-date 2026-09-01`
and confirm the flare estimate (30°) is extracted from the remarks with a verbatim quote.

## Verify
- `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest`
- `uv run dbq-agent eval --claim-date 2026-09-01` → every case `ok`
- `uv run dbq-agent run cases/knee_07.pdf --claim-date 2026-09-01` → INADEQUATE, `correia_passive`

## Blockers
- None. Open decisions: DC 5257 post-2021 criteria text (KB row marked verify), M21-1 section
  numbering (verify), real 21-0960M-9 field-name mapping when a blank fillable form is available.

## Parking lot
- OCR path for scanned DBQs (Docling) → second `IngestedDoc` producer.
- Hybrid BM25 + dense retrieval (Qdrant) once the KB covers more than a few DCs.
- PTSD DBQ: general rating formula tiers are qualitative → LLM-judged with a critic.
- Combined ratings tool (§ 4.25) and bilateral factor.
- HF Space demo; GitLab mirror + CI.
