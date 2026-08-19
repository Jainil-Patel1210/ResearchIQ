"""Smoke test for pipeline/chunker.py — section-aware chunking.

Usage:
    python scripts/verify_m3_chunking.py [arxiv_id]
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from pipeline import chunker

DEFAULT_PDF = "data/raw/pdf/1706.03762v7.pdf"
DEFAULT_ID = "1706.03762v7"


def main():
    arxiv_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ID
    pdf_path = f"data/raw/pdf/{arxiv_id}.pdf" if len(sys.argv) > 1 else DEFAULT_PDF
    print(f"testing against: {pdf_path}\n")

    chunks = chunker.chunk_pdf(pdf_path, paper_id=arxiv_id)
    print(f"total chunks: {len(chunks)}\n")

    print("--- section labels in order (with chunk counts) ---")
    seen = []
    for c in chunks:
        if not seen or seen[-1][0] != c.section:
            seen.append((c.section, 0))
        seen[-1] = (seen[-1][0], seen[-1][1] + 1)
    for section, count in seen:
        print(f"  {count:2d} chunk(s)  {section}")

    word_counts = [len(c.text.split()) for c in chunks]
    print(f"\nchunk word counts: min={min(word_counts)} max={max(word_counts)} "
          f"avg={sum(word_counts)/len(word_counts):.0f}")

    print(f"\nall chunks have paper_id set: {all(c.paper_id == arxiv_id for c in chunks)}")
    print(f"chunk_type values present: {sorted({c.chunk_type for c in chunks})}")
    print(f"chunks with a page number: {sum(1 for c in chunks if c.page is not None)}/{len(chunks)}")

    formula_chunks = [c for c in chunks if "[formula omitted]" in c.text]
    print(f"chunks containing the formula placeholder: {len(formula_chunks)}")

    # Rough mid-sentence-cut check: a text chunk shouldn't start with a
    # lowercase letter (a real sentence start is capitalized; a lowercase
    # start suggests we cut inside a sentence).
    suspect = [c for c in chunks if c.chunk_type == "text" and c.text and c.text[0].islower()]
    print(f"text chunks that look mid-sentence-cut (start lowercase): {len(suspect)}")
    if suspect:
        print("  example:", repr(suspect[0].text[:100]))

    print("\n--- first chunk ---")
    print(f"[section: {chunks[0].section} | type: {chunks[0].chunk_type} | page: {chunks[0].page}]")
    print(chunks[0].text[:400])

    table_chunk = next((c for c in chunks if c.chunk_type == "table"), None)
    print("\n--- a table chunk ---")
    if table_chunk:
        print(f"[section: {table_chunk.section} | page: {table_chunk.page}]")
        print(table_chunk.text[:400])
    else:
        print("none found")

    references_leaked = any("references" in c.section.lower() for c in chunks)
    print(f"\nany chunk still tagged as a References section: {references_leaked}")


if __name__ == "__main__":
    main()
