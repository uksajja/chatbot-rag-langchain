"""Command-line smoke test for the RAG pipeline (no UI).

Run from the repo root, with the package installed (pip install -e .).

Examples:
    python main.py --ingest                                  # build the index first
    python main.py -q "Sample Question?"    # one question
    python main.py -q "Why is there flash?" --stream         # stream tokens
    python main.py -q "PA66 drying" --retrieval-only         # no LLM call
    python main.py                                           # interactive prompt
    python main.py --override configs/experiments/local_ollama.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from dotenv import load_dotenv

from imrag.core.config import ConfigManager
from imrag.generation.context import source_label
from imrag.pipelines.ingestion import run_ingestion
from imrag.pipelines.factory import build_pipeline

PREVIEW_CHARS = 300


def print_sources(sources) -> None:
    if not sources:
        print("\n(no sources)")
        return
    print(f"\nSources ({len(sources)}):")
    for s in sources:
        text = " ".join(s.text.split())
        if len(text) > PREVIEW_CHARS:
            text = text[:PREVIEW_CHARS] + "..."
        print(f"  [{s.index}] {source_label(s)}\n      {text}")


def ask(pipe, question: str, stream: bool, retrieval_only: bool) -> None:
    print(f"\nQ: {question}\n")

    if retrieval_only:
        sources, timings = pipe.retrieve(question)
        print_sources(sources)
        print(f"\n[retrieve {timings.retrieve_ms:.0f} ms, rerank {timings.rerank_ms:.0f} ms]")
        return

    if stream:
        t0 = time.perf_counter()
        sources, tokens = pipe.stream(question)
        first = None
        for token in tokens:
            if first is None:
                first = (time.perf_counter() - t0) * 1000
            print(token, end="", flush=True)
        total = (time.perf_counter() - t0) * 1000
        print()
        print_sources(sources)
        print(f"\n[time to first token {first or 0:.0f} ms, total {total:.0f} ms]")
        return

    response = pipe.answer(question)
    print(response.answer)
    print_sources(response.sources)
    t = response.timings
    print(f"\n[retrieve {t.retrieve_ms:.0f} ms, rerank {t.rerank_ms:.0f} ms, "
          f"generate {t.generate_ms:.0f} ms, total {t.total_ms:.0f} ms]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the RAG pipeline without a UI")
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--override", nargs="*", default=[], help="override YAML files")
    parser.add_argument("--ingest", action="store_true", help="run ingestion before querying")
    parser.add_argument("-q", "--question", action="append", default=[],
                        help="question to ask (repeatable); omit for an interactive prompt")
    parser.add_argument("--stream", action="store_true", help="stream the answer")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="show retrieved chunks only, no LLM call (no API key needed)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = ConfigManager(args.config, args.override).config
    print(f"embedding={cfg.embedding.type}  vectorstore={cfg.vectorstore.type}  "
          f"retrieval={cfg.retrieval.type}  "
          f"reranker={cfg.reranker.type if cfg.reranker else 'none'}  "
          f"llm={cfg.llm.type}")

    if args.ingest:
        print("Ingesting...")
        summary = run_ingestion(cfg)
        print(f"Ingested {summary['docs']} docs -> {summary['chunks']} chunks")
        for key in ("skipped", "empty", "failed"):
            if summary.get(key):
                print(f"  {key}: {summary[key]}")

    pipe = build_pipeline(cfg)

    if args.question:
        for question in args.question:
            ask(pipe, question, args.stream, args.retrieval_only)
        return 0

    print("\nInteractive mode. Type a question, or 'exit' to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"", "exit", "quit"}:
            break
        try:
            ask(pipe, question, args.stream, args.retrieval_only)
        except Exception as exc:                  # keep the session alive on errors
            print(f"Error: {exc!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())