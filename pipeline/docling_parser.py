import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from docling.document_converter import DocumentConverter

_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter


def convert(pdf_path):
    """Run Docling and return its structured DoclingDocument (not flattened
    markdown) — chunker.py needs the item-level structure to chunk by
    section boundary."""
    return _get_converter().convert(str(pdf_path)).document


def parse_pdf(pdf_path) -> str:
    """Full, unfiltered markdown export — useful for quick inspection.
    For chunking, use convert() + chunker.py instead: this function no
    longer drops the References section (that regex-on-markdown approach
    was fragile — see CLAUDE.md's M3 decision log — chunker.py now drops
    it structurally, per-section, instead)."""
    return convert(pdf_path).export_to_markdown().strip()
