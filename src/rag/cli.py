"""Operator commands for safe hybrid-RAG index maintenance."""

from __future__ import annotations

import argparse
import json

from .service import ChemicalRAGService


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid chemical RAG maintenance")
    subparsers = parser.add_subparsers(dest="command", required=True)
    rebuild_parser = subparsers.add_parser(
        "rebuild",
        help="build a side-by-side index and activate it only after full ingestion",
    )
    rebuild_parser.add_argument(
        "sources",
        nargs="+",
        help="source documents to ingest into the replacement index",
    )
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="incrementally add or update documents in the active index",
    )
    ingest_parser.add_argument(
        "sources",
        nargs="+",
        help="source documents to add or update in the active index",
    )
    args = parser.parse_args()

    if args.command == "rebuild":
        result = ChemicalRAGService.rebuild(args.sources)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("success") else 1
    if args.command == "ingest":
        service = ChemicalRAGService()
        try:
            result = service.ingest(args.sources)
        finally:
            service.close()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("success") else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
