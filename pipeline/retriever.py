from dataclasses import dataclass

from pipeline import bm25_index, embedder, vector_store

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


def _rrf_scores(*ranked_id_lists: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    return scores


def retrieve(query: str, paper_ids: list[str], top_k: int = 6, candidate_k: int = 20) -> list[RetrievedChunk]:
    """Hybrid retrieval scoped to an explicit set of papers: dense (Chroma)
    + BM25 candidates, fused by rank (RRF), never an unscoped search."""
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

    return [
        RetrievedChunk(
            chunk_id=chunk_id,
            text=by_id[chunk_id][0],
            paper_id=by_id[chunk_id][1]["paper_id"],
            section=by_id[chunk_id][1]["section"],
            chunk_type=by_id[chunk_id][1]["chunk_type"],
            page=by_id[chunk_id][1].get("page"),
            score=fused[chunk_id],
        )
        for chunk_id in top_ids
    ]
