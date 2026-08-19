"""One-off exploration: check how to render a single TableItem to markdown
(not the whole-document export), needed for pipeline/chunker.py.

Usage:
    python scripts/explore_docling_structure.py
"""

import os
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
sys.stdout.reconfigure(encoding="utf-8")

from docling.document_converter import DocumentConverter

PDF = "data/raw/pdf/1706.03762v7.pdf"


def main():
    converter = DocumentConverter()
    result = converter.convert(PDF)
    doc = result.document

    for item, level in doc.iterate_items():
        if getattr(item, "label", None) == "table":
            print("--- table item attrs ---")
            print([a for a in dir(item) if not a.startswith("_")])
            print("\n--- export_to_markdown(doc) ---")
            print(item.export_to_markdown(doc=doc)[:500])
            break


if __name__ == "__main__":
    main()
