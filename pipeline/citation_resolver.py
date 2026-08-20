import json
import re
from dataclasses import asdict
from pathlib import Path

from pipeline import grobid_client, tei_parser
from pipeline.tei_parser import Reference

REFERENCES_DIR = Path(__file__).resolve().parent.parent / "data" / "references"

_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def fetch_and_save_references(pdf_path, paper_id: str) -> list[Reference]:
    xml = grobid_client.fetch_references_xml(pdf_path)
    references = tei_parser.parse_references(xml)
    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    path = REFERENCES_DIR / f"{paper_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in references], f, indent=2)
    return references


def load_references(paper_id: str) -> list[Reference]:
    path = REFERENCES_DIR / f"{paper_id}.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [Reference(**d) for d in data]


def build_reference_map(references: list[Reference]) -> dict[int, Reference]:
    """Maps in-text citation numbers (the "12" in "[12]") to the matching
    Reference. Relies on GROBID's reference list order matching the
    paper's own numbering — true for numbered-citation-style papers (our
    whole corpus), not for author-year style papers."""
    return {i: ref for i, ref in enumerate(references, start=1)}


def find_marker_numbers(text: str) -> list[int]:
    numbers = []
    for match in _MARKER.finditer(text):
        numbers.extend(int(n.strip()) for n in match.group(1).split(","))
    return numbers


def resolve_chunk_citations(text: str, ref_map: dict[int, Reference]) -> list[Reference]:
    resolved = []
    seen = set()
    for n in find_marker_numbers(text):
        if n in ref_map and n not in seen:
            resolved.append(ref_map[n])
            seen.add(n)
    return resolved
