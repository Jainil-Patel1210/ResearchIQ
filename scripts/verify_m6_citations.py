"""Smoke test for M6 — citation resolution.

Verifies the core assumption the whole scheme relies on: GROBID's
reference list order matches the paper's own in-text numbering (marker
[N] -> GROBID's (N-1)th reference), plus that markers actually resolve
to the correct paper when checked against known ground truth.

For corpus-wide validation (not just this one paper), see
scripts/validate_citation_corpus.py instead.

Usage:
    python scripts/verify_m6_citations.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from pipeline import citation_resolver, indexer, retriever, vector_store

PAPER_ID = "1706.03762v7"  # Attention Is All You Need
PDF_PATH = f"data/raw/pdf/{PAPER_ID}.pdf"

# Known ground truth from manually inspecting the paper's own reference
# list earlier in this project.
EXPECTED = {
    1: "layer normalization",
    13: "long short-term memory",
}


def main():
    print(f"re-indexing {PAPER_ID} (proves indexer.py -> citation_resolver wiring)...")
    n = indexer.index_pdf(PDF_PATH, paper_id=PAPER_ID)
    print(f"indexed {n} chunks, references saved to data/references/{PAPER_ID}.json\n")

    ref_map, validation = citation_resolver.load_paper_references(PAPER_ID)
    print(f"loaded {len(ref_map)} references")
    print(f"validation: {validation}\n")

    print("--- ground-truth check ---")
    all_ok = True
    for number, expected_substring in EXPECTED.items():
        ref = ref_map.get(number)
        title = (ref.title or "").lower() if ref else ""
        ok = expected_substring in title
        all_ok &= ok
        print(f"[{number}] expected {expected_substring!r} in title, got {ref.title!r} -> {'OK' if ok else 'MISMATCH'}")
    print(f"\nall ground-truth checks passed: {all_ok}")

    print("\n--- resolving citations in real retrieved chunks (via retriever.py) ---")
    results = retriever.retrieve(
        "why is recurrence a limitation for sequence modeling",
        paper_ids=[PAPER_ID],
        top_k=5,
    )
    for r in results:
        if r.citations:
            print(f"\n[{r.chunk_id}] section={r.section!r}")
            print(r.text[:200])
            print("resolved citations:")
            for ref in r.citations:
                print(f"  [{ref.ref_id}] {ref.title!r} — {ref.authors} ({ref.year})")

    print("\n--- unresolved marker check across the whole paper ---")
    collection = vector_store.get_collection()
    data = collection.get(where={"paper_id": PAPER_ID}, include=["documents"])
    unresolved_total = sum(
        1 for text in data["documents"]
        for num in citation_resolver.extract_citation_markers(text)
        if num not in ref_map
    )
    print(f"total marker instances that failed to resolve: {unresolved_total}")


if __name__ == "__main__":
    main()
