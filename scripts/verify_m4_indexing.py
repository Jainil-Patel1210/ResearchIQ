"""Smoke test for M4 — embedding + indexing (embedder, vector_store, bm25_index).

Usage:
    python scripts/verify_m4_indexing.py [arxiv_id]
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from pipeline import bm25_index, embedder, indexer, vector_store

DEFAULT_PDF = "data/raw/pdf/1706.03762v7.pdf"
DEFAULT_ID = "1706.03762v7"


def main():
    arxiv_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ID
    pdf_path = f"data/raw/pdf/{arxiv_id}.pdf" if len(sys.argv) > 1 else DEFAULT_PDF
    print(f"testing against: {pdf_path}\n")

    n = indexer.index_pdf(pdf_path, paper_id=arxiv_id)
    print(f"indexed {n} chunks")

    collection = vector_store.get_collection()
    print(f"collection count: {collection.count()}\n")

    print("--- semantic (dense) query: 'why is multi-head attention useful' ---")
    query_vec = embedder.embed(["why is multi-head attention useful"])[0]
    print(f"embedding dimension: {len(query_vec)}")
    results = vector_store.query(query_vec, top_k=3, where={"paper_id": arxiv_id})
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    )):
        print(f"\n[{i}] distance={dist:.4f} section={meta['section']!r} type={meta['chunk_type']}")
        print(doc[:250])

    print("\n\n--- BM25 (keyword) query: 'BLEU score WMT' ---")
    bm25 = bm25_index.build_from_collection(collection)
    for chunk_id, score in bm25.query("BLEU score WMT", top_k=3):
        doc = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        text = doc["documents"][0]
        section = doc["metadatas"][0]["section"]
        print(f"\n[{chunk_id}] score={score:.2f} section={section!r}")
        print(text[:250])


if __name__ == "__main__":
    main()
