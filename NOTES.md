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
- 2026-09-27 late: `--llm anthropic --strict` verified against claude-sonnet-5 on the Mac.
  Two real-model findings fixed the same night: (1) the model returns "30 degrees" as a string,
  so values are now normalized per kind before evaluation; (2) it filed a bare "cannot say
  without speculation" as a *reason*, so that reclassification is now a rule (Jones), not a
  model judgment. `dbq-agent eval` is 10/10 with both extractors, citation faithfulness 31/31.
- Silent LLM fallback is gone: failures warn with the exception name, the report records it,
  and `--strict` disables the fallback.

- 2026-09-28: DC 5257 added. Criteria as `predicate` rows (2021 ligament + patellar sub-tables
  from 85 FR 76453, verified verbatim; pre-2021 slight/moderate/severe rows with effective_to),
  eight new form fields (Section 11b), `InstabilityFindings`, `select_predicate_tier`, adequacy
  rule `instability_rx_undocumented`, version-change note, 4 new cases (knee_11–14, knee_12 with
  its own claim date). eval 14/14, 30 tests.

## Next step
Scanned-DBQ ingest path (chunk B): a second `IngestedDoc` producer for flattened PDFs —
rasterize → OCR → recover fields by the form's question labels; provenance = page + bbox.

## Verify
- `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest`
- `uv run dbq-agent eval --claim-date 2026-09-01` → every case `ok`
- `uv run dbq-agent run cases/knee_07.pdf --claim-date 2026-09-01` → INADEQUATE, `correia_passive`

## Blockers
- None. Open decisions: M21-1 section numbering (verify), real 21-0960M-9 field-name mapping
  when a blank fillable form is available, DC 5257 pending-claim dual-version evaluation
  (currently a note, not two ratings).

## Parking lot
- OCR path for scanned DBQs (Docling) → second `IngestedDoc` producer.
- Hybrid BM25 + dense retrieval (Qdrant) once the KB covers more than a few DCs.
- PTSD DBQ: general rating formula tiers are qualitative → LLM-judged with a critic.
- Combined ratings tool (§ 4.25) and bilateral factor.
- HF Space demo; GitLab mirror + CI.
