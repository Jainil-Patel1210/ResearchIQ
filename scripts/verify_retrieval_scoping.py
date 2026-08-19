"""Verify that retrieval (both dense and BM25) stays scoped to an explicit
set of paper_ids and never leaks results from other indexed papers.

Needs at least two different papers indexed to be a meaningful test —
indexes a second paper if it isn't already present.

Usage:
    python scripts/verify_retrieval_scoping.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from pipeline import bm25_index, embedder, indexer, vector_store

PAPER_A = "1706.03762v7"   # Attention Is All You Need
PAPER_B = "1810.04805v2"   # BERT


def main():
    collection = vector_store.get_collection()

    for arxiv_id in (PAPER_A, PAPER_B):
        n = indexer.index_pdf(f"data/raw/pdf/{arxiv_id}.pdf", paper_id=arxiv_id)
        print(f"indexed {arxiv_id}: {n} chunks")

    print(f"\ntotal collection count (both papers): {collection.count()}")

    print(f"\n--- BM25 scoped to only {PAPER_A} ---")
    bm25 = bm25_index.build_from_collection(collection, where={"paper_id": PAPER_A})
    print(f"BM25 index built from {len(bm25.ids)} chunks")
    leaked = [i for i in bm25.ids if not i.startswith(PAPER_A)]
    print(f"chunk ids belonging to a different paper (should be 0): {len(leaked)}")

    print(f"\n--- dense query scoped to only {PAPER_A}, asking about BERT-specific content ---")
    query_vec = embedder.embed(["bidirectional masked language model pre-training"])[0]
    results = vector_store.query(query_vec, top_k=3, where={"paper_id": PAPER_A})
    ids = results["ids"][0]
    leaked_dense = [i for i in ids if not i.startswith(PAPER_A)]
    print(f"results returned: {ids}")
    print(f"results belonging to a different paper (should be 0): {len(leaked_dense)}")


if __name__ == "__main__":
    main()
