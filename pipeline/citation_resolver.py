import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pipeline import grobid_client, tei_parser
from pipeline.tei_parser import Reference

REFERENCES_DIR = Path(__file__).resolve().parent.parent / "data" / "references"

_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# Words that suggest a "[N]" isn't actually a bibliographic citation (an
# equation/figure/table/section/algorithm number instead). Checked against
# the ~15 characters immediately before the bracket.
_NON_CITATION_CONTEXT = re.compile(
    r"(equation|eq\.|figure|fig\.|algorithm|alg\.|section|sec\.|table)\s*$",
    re.IGNORECASE,
)


@dataclass
class CitationValidation:
    citation_style: str  # "numbered" or "unknown"
    reference_count: int
    unique_markers_found: int
    max_marker: int | None
    unresolved_markers: list[int]
    suspicious_non_citation_brackets: int
    valid: bool


def build_reference_map(references: list[Reference]) -> dict[int, Reference]:
    """Marker number N (the "12" in "[12]") -> GROBID's (N-1)th reference.
    Relies on GROBID's reference-list order matching the paper's own
    numbering — true for numbered-citation-style papers, not author-year
    style. See validate_numbered_citations() to check this actually holds
    for a given paper before trusting resolve()'s output."""
    return {i: ref for i, ref in enumerate(references, start=1)}


def extract_citation_markers(text: str) -> list[int]:
    numbers = []
    for match in _MARKER.finditer(text):
        numbers.extend(int(n.strip()) for n in match.group(1).split(","))
    return numbers


def _is_suspicious_bracket(text: str, match_start: int) -> bool:
    preceding = text[max(0, match_start - 15):match_start]
    return bool(_NON_CITATION_CONTEXT.search(preceding))


def count_suspicious_brackets(text: str) -> int:
    """Count "[N]" occurrences that look like they're numbering an
    equation/figure/table/section/algorithm rather than citing a
    reference (e.g. "Equation [12]")."""
    return sum(1 for m in _MARKER.finditer(text) if _is_suspicious_bracket(text, m.start()))


def resolve(text: str, ref_map: dict[int, Reference]) -> list[Reference]:
    resolved = []
    seen = set()
    for n in extract_citation_markers(text):
        if n in ref_map and n not in seen:
            resolved.append(ref_map[n])
            seen.add(n)
    return resolved


def validate_numbered_citations(full_text: str, references: list[Reference]) -> CitationValidation:
    """Checks whether the [N] -> (N-1)th-reference assumption actually
    holds for this paper, rather than just assuming it does. Call this
    once per paper (at index time) and persist the result — don't trust
    resolve() output for a paper without checking this first."""
    unique_markers = sorted(set(extract_citation_markers(full_text)))
    ref_count = len(references)
    unresolved = [n for n in unique_markers if n < 1 or n > ref_count]
    suspicious = count_suspicious_brackets(full_text)

    resolved_fraction = 1 - (len(unresolved) / len(unique_markers)) if unique_markers else 0.0
    # A numbered-citation paper should have plenty of markers, almost all
    # of them resolving in range, and [1] should appear somewhere (nearly
    # universal — the first reference is cited early in any paper).
    is_numbered = bool(unique_markers) and resolved_fraction >= 0.9 and 1 in unique_markers

    return CitationValidation(
        citation_style="numbered" if is_numbered else "unknown",
        reference_count=ref_count,
        unique_markers_found=len(unique_markers),
        max_marker=unique_markers[-1] if unique_markers else None,
        unresolved_markers=unresolved,
        suspicious_non_citation_brackets=suspicious,
        valid=is_numbered and not unresolved,
    )


def fetch_and_save_references(
    pdf_path, paper_id: str, full_text: str
) -> tuple[list[Reference], CitationValidation]:
    xml = grobid_client.fetch_references_xml(pdf_path)
    references = tei_parser.parse_references(xml)
    validation = validate_numbered_citations(full_text, references)
    ref_map = build_reference_map(references)

    # Also grab the paper's own title (header endpoint is lightweight,
    # same GROBID service, no extra service call) — M7 needs this to show
    # "Attention Is All You Need, p.5" instead of "1706.03762v7, p.5".
    header_xml = grobid_client.fetch_header_xml(pdf_path)
    header = tei_parser.parse_header(header_xml)

    payload = {
        "paper_id": paper_id,
        "title": header.title,
        "citation_style": validation.citation_style,
        "validation": asdict(validation),
        "references": {str(n): asdict(ref) for n, ref in ref_map.items()},
    }
    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    with open(REFERENCES_DIR / f"{paper_id}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return references, validation


def load_paper_references(paper_id: str) -> tuple[dict[int, Reference], CitationValidation]:
    """Loads what fetch_and_save_references persisted — M7 (and
    retriever.py) never needs to call GROBID again to resolve a
    citation."""
    with open(REFERENCES_DIR / f"{paper_id}.json", encoding="utf-8") as f:
        payload = json.load(f)
    ref_map = {int(n): Reference(**d) for n, d in payload["references"].items()}
    validation = CitationValidation(**payload["validation"])
    return ref_map, validation


def load_paper_title(paper_id: str) -> str | None:
    """Falls back to None (caller should then fall back to paper_id) for
    papers indexed before this field was added — no need to re-run the
    whole corpus just to backfill titles."""
    try:
        with open(REFERENCES_DIR / f"{paper_id}.json", encoding="utf-8") as f:
            payload = json.load(f)
    except FileNotFoundError:
        return None
    return payload.get("title")
