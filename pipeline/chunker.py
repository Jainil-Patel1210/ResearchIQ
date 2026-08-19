import re
from dataclasses import dataclass

from pipeline import docling_parser

# all-MiniLM-L6-v2 truncates at 256 tokens; word count is an approximation
# for token count (subword tokenization usually yields more tokens than
# words), so this stays well under the limit rather than right at it.
CHUNK_SIZE_WORDS = 150
CHUNK_OVERLAP_WORDS = 30

_REFERENCES_SECTION = re.compile(
    r"^\s*(\d+\.?\s*)?(references|bibliography|works cited)\s*$", re.IGNORECASE
)

# Approximation, not a real sentence tokenizer — doesn't handle abbreviations
# like "et al." or decimals like "3.5 days" perfectly, but good enough to
# avoid the much worse problem of cutting mid-sentence at an arbitrary word
# count.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_CONTENT_LABELS = {"text", "footnote", "caption"}
_FORMULA_PLACEHOLDER = "[formula omitted]"
_FRONT_MATTER = "Front Matter"


@dataclass
class Chunk:
    text: str
    section: str
    chunk_index: int
    paper_id: str
    chunk_type: str  # "text" or "table"
    page: int | None


def _page_of(item) -> int | None:
    prov = getattr(item, "prov", None)
    return getattr(prov[0], "page_no", None) if prov else None


def _group_by_section(doc) -> list[tuple[str, list[tuple[str, str, int | None]]]]:
    """Walk the document in reading order, grouping (kind, text, page)
    blocks under the nearest preceding section_header, where kind is
    "text" or "table". Skips the references section and any section with
    no real content (e.g. a heading misdetected from text inside a
    figure). Formulas keep a placeholder (not skipped outright) so the
    surrounding prose doesn't silently lose the anchor."""
    sections: list[tuple[str, list[tuple[str, str, int | None]]]] = [(_FRONT_MATTER, [])]

    for item, _level in doc.iterate_items():
        label = getattr(item, "label", None)

        if label == "section_header":
            sections.append((item.text, []))
            continue

        if label == "table":
            sections[-1][1].append(("table", item.export_to_markdown(doc=doc), _page_of(item)))
        elif label == "formula":
            sections[-1][1].append(("text", _FORMULA_PLACEHOLDER, _page_of(item)))
        elif label in _CONTENT_LABELS and item.text:
            sections[-1][1].append(("text", item.text, _page_of(item)))
        # picture items (no text) are still skipped — nothing to anchor.

    return [
        (heading, blocks)
        for heading, blocks in sections
        if blocks and not _REFERENCES_SECTION.match(heading)
    ]


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT.split(text) if s]


def _pack_sentences(sentences: list[str], size: int, overlap: int) -> list[str]:
    """Greedily pack whole sentences into ~size-word pieces, carrying the
    trailing ~overlap words of sentences into the next piece. Never cuts
    mid-sentence — a piece only exceeds size when a single sentence does."""
    if not sentences:
        return []

    pieces = []
    current: list[str] = []
    current_words = 0
    i = 0

    while i < len(sentences):
        sentence = sentences[i]
        sentence_words = len(sentence.split())

        if current and current_words + sentence_words > size:
            pieces.append(" ".join(current))
            overlap_sentences: list[str] = []
            overlap_words = 0
            for s in reversed(current):
                w = len(s.split())
                if overlap_words + w > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_words += w
            current, current_words = overlap_sentences, overlap_words
            continue

        current.append(sentence)
        current_words += sentence_words
        i += 1

    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_pdf(pdf_path, paper_id: str) -> list[Chunk]:
    doc = docling_parser.convert(pdf_path)
    chunks: list[Chunk] = []

    for section, blocks in _group_by_section(doc):
        prose_buffer: list[str] = []
        prose_page: int | None = None

        def flush_prose():
            nonlocal prose_page
            if not prose_buffer:
                return
            sentences = _split_sentences(" ".join(prose_buffer))
            for piece in _pack_sentences(sentences, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS):
                chunks.append(Chunk(
                    text=piece, section=section, chunk_index=len(chunks),
                    paper_id=paper_id, chunk_type="text", page=prose_page,
                ))
            prose_buffer.clear()
            prose_page = None

        for kind, content, page in blocks:
            if kind == "table":
                # Tables are kept atomic — never split mid-table, even past
                # CHUNK_SIZE_WORDS. A truncated-but-coherent table beats one
                # fragmented across two chunks with no header context.
                flush_prose()
                chunks.append(Chunk(
                    text=content, section=section, chunk_index=len(chunks),
                    paper_id=paper_id, chunk_type="table", page=page,
                ))
            else:
                if prose_page is None:
                    prose_page = page
                prose_buffer.append(content)

        flush_prose()

    return chunks
