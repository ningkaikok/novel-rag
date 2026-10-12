"""One-shot retention for an offline tenant worker (used by production_ops)."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backend.retention import prune_documents, prune_history  # noqa: E402
from config import KNOWLEDGE_DIR, NOVELS_DIR  # noqa: E402
from postgres import connect  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--document-days", type=int)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    with connect() as conn:
        result = {"history": prune_history(conn, days=args.days, dry_run=not args.apply)}
        if args.document_days is not None:
            result["documents"] = prune_documents(
                conn,
                days=args.document_days,
                novels_dir=NOVELS_DIR,
                knowledge_dir=KNOWLEDGE_DIR,
                dry_run=not args.apply,
            )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
