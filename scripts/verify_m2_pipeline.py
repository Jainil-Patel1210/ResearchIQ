"""Smoke test for the M2 parsing pipeline (grobid_client + tei_parser + docling_parser).

Run against one corpus paper and print results for manual inspection.

Usage:
    python scripts/verify_m2_pipeline.py [arxiv_id]
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from pipeline import docling_parser, grobid_client, tei_parser

DEFAULT_PDF = "data/raw/pdf/1706.03762v7.pdf"


def main():
    pdf_path = f"data/raw/pdf/{sys.argv[1]}.pdf" if len(sys.argv) > 1 else DEFAULT_PDF
    print(f"testing against: {pdf_path}\n")

    header_xml = grobid_client.fetch_header_xml(pdf_path)
    header = tei_parser.parse_header(header_xml)
    print("--- header ---")
    print("title:", header.title)
    print("authors:", header.authors)

    refs_xml = grobid_client.fetch_references_xml(pdf_path)
    refs = tei_parser.parse_references(refs_xml)
    print("\n--- references ---")
    print("count:", len(refs))
    print("ref[0]:", refs[0])
    missing_titles = [r.ref_id for r in refs if r.title is None]
    print("refs with no title even after fallback:", missing_titles)

    content_md = docling_parser.parse_pdf(pdf_path)
    print("\n--- docling content ---")
    print("length:", len(content_md))
    print("contains 'References' near the end:", "References" in content_md[-200:])
    print("last 200 chars:", repr(content_md[-200:]))


if __name__ == "__main__":
    main()
