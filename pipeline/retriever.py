from dataclasses import dataclass

from pipeline import bm25_index, citation_resolver, embedder, vector_store
from pipeline.tei_parser import Reference

# Standard constant from the original Reciprocal Rank Fusion paper
# (Cormack et al., 2009) — not tuned, this is the well-known default.
RRF_K = 60


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    paper_id: str
    section: str
    chunk_type: str
    page: int | None
    score: float
    citations: list[Reference]


def _rrf_scores(*ranked_id_lists: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    return scores


def _citations_for(text: str, paper_id: str, ref_map_cache: dict) -> list[Reference]:
    if paper_id not in ref_map_cache:
        try:
            ref_map, _validation = citation_resolver.load_paper_references(paper_id)
        except FileNotFoundError:
            ref_map = {}
        ref_map_cache[paper_id] = ref_map
    return citation_resolver.resolve(text, ref_map_cache[paper_id])


def retrieve(query: str, paper_ids: list[str], top_k: int = 6, candidate_k: int = 20) -> list[RetrievedChunk]:
    """Hybrid retrieval scoped to an explicit set of papers: dense (Chroma)
    + BM25 candidates, fused by rank (RRF), never an unscoped search. Each
    result carries its resolved citations directly, so callers (M7) don't
    need to touch citation_resolver themselves."""
    collection = vector_store.get_collection()
    where = {"paper_id": {"$in": paper_ids}}

    query_vec = embedder.embed([query])[0]
    dense_results = vector_store.query(query_vec, top_k=candidate_k, where=where)
    dense_ids = dense_results["ids"][0]

    bm25 = bm25_index.build_from_collection(collection, where=where)
    bm25_ids = [chunk_id for chunk_id, _score in bm25.query(query, top_k=candidate_k)]

    fused = _rrf_scores(dense_ids, bm25_ids)
    top_ids = sorted(fused, key=fused.get, reverse=True)[:top_k]
    if not top_ids:
        return []

    data = collection.get(ids=top_ids, include=["documents", "metadatas"])
    by_id = {i: (doc, meta) for i, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])}

    ref_map_cache: dict = {}
    results = []
    for chunk_id in top_ids:
        text, meta = by_id[chunk_id]
        paper_id = meta["paper_id"]
        results.append(RetrievedChunk(
            chunk_id=chunk_id,
            text=text,
            paper_id=paper_id,
            section=meta["section"],
            chunk_type=meta["chunk_type"],
            page=meta.get("page"),
            score=fused[chunk_id],
            citations=_citations_for(text, paper_id, ref_map_cache),
        ))
    return results


def retrieve_cross_paper(
    query: str, paper_ids: list[str], top_k_per_paper: int = 3, candidate_k: int = 20
) -> dict[str, list[RetrievedChunk]]:
    """Per-paper scoped retrieval for cross-paper questions: runs retrieve()
    once per paper (top_k_per_paper each), never a single global top-k over
    the combined set — otherwise one paper with more/denser matching chunks
    could crowd another paper out entirely. Every paper_id is always
    included, even if it returns few/no strong matches for this particular
    query (M8 decision: the user explicitly chose these papers, so nothing
    gets silently dropped based on relevance)."""
    return {
        paper_id: retrieve(query, [paper_id], top_k=top_k_per_paper, candidate_k=candidate_k)
        for paper_id in paper_ids
    }
