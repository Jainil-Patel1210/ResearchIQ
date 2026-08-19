from pathlib import Path

import chromadb

CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
COLLECTION_NAME = "paper_chunks"

_client = None


def _get_client():
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def get_collection():
    return _get_client().get_or_create_collection(COLLECTION_NAME)


def _chunk_metadata(chunk) -> dict:
    meta = {
        "paper_id": chunk.paper_id,
        "section": chunk.section,
        "chunk_type": chunk.chunk_type,
        "chunk_index": chunk.chunk_index,
    }
    if chunk.page is not None:
        meta["page"] = chunk.page
    return meta


def add_chunks(chunks, embeddings) -> None:
    if not chunks:
        return
    collection = get_collection()
    # upsert, not add — re-running the indexer on a paper we've already
    # indexed should overwrite, not error on duplicate IDs.
    collection.upsert(
        ids=[f"{c.paper_id}_{c.chunk_index}" for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[_chunk_metadata(c) for c in chunks],
    )


def query(embedding, top_k: int = 6, where: dict | None = None):
    collection = get_collection()
    return collection.query(query_embeddings=[embedding], n_results=top_k, where=where)
