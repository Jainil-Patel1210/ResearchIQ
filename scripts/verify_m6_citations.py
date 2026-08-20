"""Smoke test for M6 — citation resolution.

Verifies the core assumption the whole scheme relies on: GROBID's
reference list order matches the paper's own in-text numbering (marker
[N] -> GROBID's (N-1)th reference), plus that markers actually resolve
to the correct paper when checked against known ground truth.

Usage:
    python scripts/verify_m6_citations.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from pipeline import citation_resolver, indexer, vector_store

PAPER_ID = "1706.03762v7"  # Attention Is All You Need
PDF_PATH = f"data/raw/pdf/{PAPER_ID}.pdf"

# Known ground truth from manually inspecting the paper's own reference
# list earlier in this project — [13] should be Hochreiter & Schmidhuber's
# "Long short-term memory", and [1] should be "Layer normalization".
EXPECTED = {
    1: "layer normalization",
    13: "long short-term memory",
}


def main():
    print(f"re-indexing {PAPER_ID} (proves indexer.py -> citation_resolver wiring)...")
    n = indexer.index_pdf(PDF_PATH, paper_id=PAPER_ID)
    print(f"indexed {n} chunks, references saved to data/references/{PAPER_ID}.json\n")

    references = citation_resolver.load_references(PAPER_ID)
    print(f"loaded {len(references)} references")
    ref_map = citation_resolver.build_reference_map(references)

    print("\n--- ground-truth check ---")
    all_ok = True
    for number, expected_substring in EXPECTED.items():
        ref = ref_map.get(number)
        title = (ref.title or "").lower() if ref else ""
        ok = expected_substring in title
        all_ok &= ok
        print(f"[{number}] expected {expected_substring!r} in title, got {ref.title!r} -> {'OK' if ok else 'MISMATCH'}")
    print(f"\nall ground-truth checks passed: {all_ok}")

    print("\n--- resolving citations in real retrieved chunks ---")
    collection = vector_store.get_collection()
    data = collection.get(where={"paper_id": PAPER_ID}, include=["documents"])
    chunks_with_markers = [
        (chunk_id, text) for chunk_id, text in zip(data["ids"], data["documents"])
        if citation_resolver.find_marker_numbers(text)
    ]
    print(f"chunks containing at least one citation marker: {len(chunks_with_markers)}")

    chunk_id, text = chunks_with_markers[0]
    numbers = citation_resolver.find_marker_numbers(text)
    resolved = citation_resolver.resolve_chunk_citations(text, ref_map)
    print(f"\nexample chunk [{chunk_id}]:")
    print(text[:200])
    print(f"\nmarker numbers found: {numbers}")
    print("resolved to:")
    for ref in resolved:
        print(f"  [{ref.ref_id}] {ref.title!r} — {ref.authors} ({ref.year})")

    unresolved_total = sum(
        1 for _id, t in chunks_with_markers
        for n in citation_resolver.find_marker_numbers(t)
        if n not in ref_map
    )
    print(f"\ntotal marker instances across all chunks that failed to resolve: {unresolved_total}")


if __name__ == "__main__":
    main()
