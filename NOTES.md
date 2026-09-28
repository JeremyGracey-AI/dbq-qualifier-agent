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

- 2026-09-28: scanned-DBQ ingest (`ingest_ocr.py`). Sparse-mode tesseract page pass →
  monotone DP alignment of form labels to OCR lines → per-field zone reads (text: re-OCR of
  the strip right of the label with box borders erased; radios/checkboxes: ink fill inside the
  mark beside the option label; text areas: lines under the label). `ingest_any()` routes by
  presence of form fields; `synth --scanned` / `eval --scanned` build and score image-only
  renditions. 14/14 cases identical through OCR; field recovery exact on all 14 (text areas
  ≥ 0.9 similarity). 41 tests. Things that bit: psm 6 drops whole lines and lone digits in
  boxes; tesseract's char whitelist silently drops characters (don't use it); a greedy
  first-match anchor cascades after one dropped line (hence the DP); zone crops must be
  clamped to the line's own band or the row above bleeds in as garbage.

## Next step
Pick one: (a) real 21-0960M-9 field-name mapping once a blank fillable form is in hand;
(b) DC 5003 / 5258 / 5259 criteria; (c) pending-claim dual-version DC 5257 evaluation (two
ratings, not a note).

## Verify
- `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest`
- `uv run dbq-agent eval --claim-date 2026-09-01` → every case `ok`
- `uv run dbq-agent eval --scanned --claim-date 2026-09-01` → every case `ok` (needs tesseract; ~6 min)
- `uv run dbq-agent run cases/knee_07.pdf --claim-date 2026-09-01` → INADEQUATE, `correia_passive`

## Blockers
- None. Open decisions: M21-1 section numbering (verify), real 21-0960M-9 field-name mapping
  when a blank fillable form is available, DC 5257 pending-claim dual-version evaluation
  (currently a note, not two ratings).

## Parking lot
- OCR: handwriting / stamps / fax artefacts; a second engine behind `OcrEngine` (e.g. docTR) to compare.
- Hybrid BM25 + dense retrieval (Qdrant) once the KB covers more than a few DCs.
- PTSD DBQ: general rating formula tiers are qualitative → LLM-judged with a critic.
- Combined ratings tool (§ 4.25) and bilateral factor.
- HF Space demo; GitLab mirror + CI.
