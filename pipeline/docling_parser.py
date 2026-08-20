import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        # do_ocr defaults to True in Docling, but our corpus is entirely
        # digitally-native arXiv PDFs (never scanned) — the text layer is
        # always directly extractable, OCR is pure overhead. Worse: on
        # some diagram-heavy papers (e.g. DDPM) it got stuck in a loop of
        # "RapidOCR returned empty result!" that took ~55 minutes on one
        # paper before we disabled this. See CLAUDE.md's M6 corpus
        # validation entry.
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        _converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
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
