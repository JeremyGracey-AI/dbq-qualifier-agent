"""dbq-agent command line.

dbq-agent run cases/knee_03.pdf [--claim-date 2026-09-01] [--llm heuristic|anthropic] [--json out.json]
dbq-agent eval [--cases cases] [--llm ...]
dbq-agent synth [--out cases]
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from dbq_agent.extract import HeuristicTextExtractor, LLMTextExtractor, TextExtractor
from dbq_agent.llm import AnthropicClient
from dbq_agent.pipeline import Context, run
from dbq_agent.report import build_report, to_json, to_markdown


def _extractor(name: str, strict: bool = False) -> TextExtractor:
    if name == "heuristic":
        return HeuristicTextExtractor()
    if name == "anthropic":
        if not AnthropicClient.available():
            sys.exit("ANTHROPIC_API_KEY is not set; use --llm heuristic or export the key")
        fallback = None if strict else HeuristicTextExtractor()
        return LLMTextExtractor(AnthropicClient(), fallback=fallback)
    sys.exit(f"unknown --llm {name!r}")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--claim-date",
        type=date.fromisoformat,
        default=date.today(),
        help="controls which criteria version applies",
    )
    p.add_argument(
        "--llm", choices=["heuristic", "anthropic"], default="heuristic", help="free-text extractor"
    )
    p.add_argument("--max-iter", type=int, default=2, help="bounded re-retrieval iterations")
    p.add_argument(
        "--strict",
        action="store_true",
        help="with --llm anthropic: fail instead of falling back to the heuristic extractor",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dbq-agent", description="DBQ → 38 CFR qualifier mapping (decision support)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the pipeline on one DBQ PDF")
    p_run.add_argument("pdf", type=Path)
    p_run.add_argument("--json", type=Path, default=None, help="also write the JSON report here")
    p_run.add_argument("--quiet", action="store_true", help="print only the JSON")
    _add_common(p_run)

    p_eval = sub.add_parser("eval", help="score the pipeline against cases/truth.json")
    p_eval.add_argument("--cases", type=Path, default=Path("cases"))
    _add_common(p_eval)

    p_synth = sub.add_parser("synth", help="regenerate the synthetic cases")
    p_synth.add_argument("--out", type=Path, default=Path("cases"))

    args = parser.parse_args(argv)

    if args.cmd == "synth":
        from dbq_agent.synth.cases import write_cases

        paths = write_cases(args.out)
        print(f"wrote {len(paths)} cases + truth.json to {args.out.resolve()}")
        return 0

    ctx = Context(
        text_extractor=_extractor(args.llm, strict=args.strict),
        max_retrieval_iterations=args.max_iter,
    )

    if args.cmd == "run":
        state = run(args.pdf, args.claim_date, ctx)
        report = build_report(state, ctx.kb)
        if args.json:
            args.json.write_text(to_json(report) + "\n")
        if args.quiet:
            print(to_json(report))
        else:
            print(to_markdown(report, ctx.kb))
            if args.json:
                print(f"(JSON written to {args.json})")
        return 0

    if args.cmd == "eval":
        from dbq_agent.evaluate_cases import evaluate_cases
        from dbq_agent.synth.cases import load_truth

        truth = load_truth(args.cases)
        result = evaluate_cases(args.cases, truth, args.claim_date, ctx)
        print(result.table())
        return 0 if all(c["ok"] for c in result.per_case) else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
