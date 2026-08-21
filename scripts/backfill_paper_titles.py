"""One-off backfill: add "title" to already-processed data/references/*.json
files that predate citation_resolver.py's title field. Only calls GROBID's
lightweight header endpoint — no Docling reparse, no full-text validation
re-run, so this is cheap even across the whole corpus.

Usage:
    python scripts/backfill_paper_titles.py
"""

import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from pipeline import citation_resolver, grobid_client, tei_parser

MANIFEST_PATH = Path("data/manifest.csv")


def main():
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        papers = list(csv.DictReader(f))

    updated = 0
    for row in papers:
        paper_id, pdf_path = row["paper_id"], row["pdf_path"]
        ref_file = citation_resolver.REFERENCES_DIR / f"{paper_id}.json"
        if not ref_file.exists():
            continue
        with open(ref_file, encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("title"):
            continue  # already has one

        header_xml = grobid_client.fetch_header_xml(pdf_path)
        header = tei_parser.parse_header(header_xml)
        payload["title"] = header.title
        with open(ref_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"{paper_id}: {header.title!r}")
        updated += 1

    print(f"\nbackfilled {updated} papers")


if __name__ == "__main__":
    main()
