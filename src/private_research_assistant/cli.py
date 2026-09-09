from __future__ import annotations

import argparse
import sys

from .crawl import crawl_sources
from .evaluation import run_evaluation
from .exceptions import UserFacingError
from .extraction import extract_opportunities
from .inspection import inspect_chunks, inspect_opportunities, inspect_raw_documents, inspect_status
from .indexing import build_index
from .retrieval import answer_question
from .setup_check import run_setup_checks


def main(argv: list[str] | None = None) -> int:
    """Read command arguments, call one handler, and return an exit status.

    None means read the real command line; tests may pass a list of strings.
    For extraction the call path is main -> _cmd_extract -> extract_opportunities.
    This dispatcher does not itself crawl, embed, or validate advertisements.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "setup-check":
            return _cmd_setup_check(args)
        if args.command == "crawl":
            return _cmd_crawl(args)
        if args.command == "index":
            return _cmd_index(args)
        if args.command == "extract":
            return _cmd_extract(args)
        if args.command == "ask":
            return _cmd_ask(args)
        if args.command == "eval":
            return _cmd_eval(args)
        if args.command == "status":
            return _cmd_status(args)
        if args.command == "inspect":
            return _cmd_inspect(args)
        if args.command == "demo":
            from .demo import run_demo
            return run_demo()
    except UserFacingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130

    parser.print_help()
    return 1


def build_parser() -> argparse.ArgumentParser:
    """Define accepted commands/options; parsing them does not execute the pipeline."""
    parser = argparse.ArgumentParser(
        prog="pra",
        description="Private Research Assistant for citation-backed AI/ML PhD opportunity discovery.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("setup-check", help="Check local readiness.")
    subparsers.add_parser("status", help="Show the latest crawl/index/output state.")
    subparsers.add_parser("demo", help="Run a synthetic offline extraction demonstration; no models or API key.")

    crawl_parser = subparsers.add_parser("crawl", help="Collect public source pages with Firecrawl.")
    crawl_parser.add_argument("--source", action="append", dest="sources", help="Source id to crawl. Can be repeated.")
    crawl_parser.add_argument("--run-id", help="Optional run id for saved data.")
    crawl_parser.add_argument("--limit", type=int, help="Override per-source result limit.")

    index_parser = subparsers.add_parser("index", help="Chunk, embed, and index the latest crawl.")
    index_parser.add_argument("--run-id", help="Raw crawl run id to index. Defaults to latest.")

    extract_parser = subparsers.add_parser("extract", help="Extract citation-backed opportunity table.")
    extract_parser.add_argument("--run-id", help="Indexed run id to extract from. Defaults to latest.")
    extract_parser.add_argument("--max-documents", type=int, help="Limit documents for a small test run.")
    extract_parser.add_argument("--format", choices=("all", "json", "markdown", "csv"), default="all")
    extract_parser.add_argument("--llm-fallback", action="store_true", help="Use the local LLM for documents the structured parser cannot read.")

    ask_parser = subparsers.add_parser("ask", help="Ask a cited question over indexed evidence.")
    ask_parser.add_argument("question", nargs="+")
    ask_parser.add_argument("--run-id", help="Indexed run id to query. Defaults to latest.")
    ask_parser.add_argument("--top-k", type=int, default=8)
    ask_parser.add_argument("--show-context", action="store_true", help="Print retrieved evidence text snippets.")

    eval_parser = subparsers.add_parser("eval", help="Run the ten-question evidence evaluation.")
    eval_parser.add_argument("--live", action="store_true", help="Also validate latest live extraction output.")

    inspect_parser = subparsers.add_parser("inspect", help="Peek inside saved pipeline artifacts.")
    inspect_parser.add_argument("stage", choices=("raw", "chunks", "opportunities"))
    inspect_parser.add_argument("--run-id", help="Run id to inspect. Defaults to latest where applicable.")
    inspect_parser.add_argument("--limit", type=int, default=5)
    inspect_parser.add_argument("--query", help="Optional text filter for raw documents or chunks.")

    return parser


def _cmd_setup_check(_: argparse.Namespace) -> int:
    results = run_setup_checks()
    required_failures = [result for result in results if result.required and not result.passing]
    for result in results:
        status = "OK" if result.passing else ("FAIL" if result.required else "WARN")
        print(f"{status:4} {result.name}: {result.detail}")
    return 1 if required_failures else 0


def _cmd_crawl(args: argparse.Namespace) -> int:
    manifest = crawl_sources(selected_source_ids=args.sources, run_id=args.run_id, limit_override=args.limit)
    print(f"Saved {manifest['document_count']} documents for run {manifest['run_id']}.")
    print(f"Manifest: data/raw/{manifest['run_id']}/manifest.json")
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    manifest = build_index(run_id=args.run_id)
    print(f"Indexed {manifest['chunk_count']} chunks from {manifest['document_count']} documents.")
    print(f"Collection: {manifest['collection_name']}")
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    """Pass parsed settings to extraction, then print its returned summary.

    args holds settings such as max_documents and llm_fallback. result holds
    rows, warnings, and file paths after extraction has written its outputs.
    Exit status 0 means the command completed, not that it found any rows.
    """
    result = extract_opportunities(
        run_id=args.run_id,
        max_documents=args.max_documents,
        llm_fallback=args.llm_fallback,
        output_format=args.format,
    )
    print(f"Extracted {len(result.opportunities)} opportunities with textual support, active as of crawl date {result.crawl_date} for run {result.run_id}.")
    if result.warnings:
        print(f"Dropped or warned on {len(result.warnings)} items. See warnings output.")
    for label, path in result.output_paths.items():
        print(f"{label}: {path}")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    question = " ".join(args.question)
    result = answer_question(question, top_k=args.top_k, run_id=args.run_id)
    print(result.answer)
    print("\nSources:")
    for item in result.evidence:
        score = "" if item.score is None else f" score={item.score:.3f}"
        print(f"- [{item.evidence_id}]{score} {item.title} {item.url}")
        if args.show_context:
            print(f"  {item.text[:500]}")
    return 0 if result.verified else 1


def _cmd_eval(args: argparse.Namespace) -> int:
    result = run_evaluation(include_live=args.live)
    passed = sum(1 for case in result.fixture_results if case.passing)
    total = len(result.fixture_results)
    print(f"Fixture consistency checks (saved answers; not RAG accuracy): {passed}/{total} passed")
    for case in result.fixture_results:
        status = "OK" if case.passing else "FAIL"
        print(f"{status:4} {case.case_id}: {case.question} ({case.detail})")
    if args.live:
        live_status = "OK" if result.live_passing else "FAIL"
        print(f"{live_status:4} live: {result.live_detail}")
    return 0 if result.passing else 1


def _cmd_status(_: argparse.Namespace) -> int:
    print(inspect_status())
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    if args.stage == "raw":
        print(inspect_raw_documents(run_id=args.run_id, limit=args.limit, query=args.query))
    elif args.stage == "chunks":
        print(inspect_chunks(run_id=args.run_id, limit=args.limit, query=args.query))
    elif args.stage == "opportunities":
        print(inspect_opportunities(limit=args.limit))
    return 0
