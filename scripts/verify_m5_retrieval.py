"""Smoke test for M5 — hybrid (dense + BM25) retrieval via RRF fusion.

Usage:
    python scripts/verify_m5_retrieval.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from pipeline import bm25_index, embedder, retriever, vector_store

PAPER_A = "1706.03762v7"   # Attention Is All You Need
PAPER_B = "1810.04805v2"   # BERT


def show_component_lists(query: str, paper_ids: list[str]):
    """Print the raw dense-only and BM25-only rankings separately, so the
    fusion behavior is visible rather than a black box."""
    collection = vector_store.get_collection()
    where = {"paper_id": {"$in": paper_ids}}

    query_vec = embedder.embed([query])[0]
    dense = vector_store.query(query_vec, top_k=5, where=where)
    print("dense-only top 5:", dense["ids"][0])

    bm25 = bm25_index.build_from_collection(collection, where=where)
    print("bm25-only top 5: ", [i for i, _ in bm25.query(query, top_k=5)])


def main():
    query = "why is multi-head attention useful"

    print(f"--- scoped to just {PAPER_A} ---")
    print(f"query: {query!r}\n")
    show_component_lists(query, [PAPER_A])

    results = retriever.retrieve(query, paper_ids=[PAPER_A], top_k=5)
    print("\nfused (RRF) top 5:")
    for r in results:
        print(f"  {r.chunk_id}  score={r.score:.4f}  section={r.section!r}")
    leaked = [r for r in results if r.paper_id != PAPER_A]
    print(f"\nresults from a different paper (should be 0): {len(leaked)}")

    print(f"\n\n--- scoped to BOTH papers, same Transformer-specific query ---")
    results2 = retriever.retrieve(query, paper_ids=[PAPER_A, PAPER_B], top_k=5)
    for r in results2:
        print(f"  {r.chunk_id}  score={r.score:.4f}  paper={r.paper_id}  section={r.section!r}")
    other_papers = {r.paper_id for r in results2}
    print(f"\npapers represented in results: {other_papers}")


if __name__ == "__main__":
    main()
