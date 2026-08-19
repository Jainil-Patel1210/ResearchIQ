import re

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, ids: list[str], texts: list[str]):
        self.ids = ids
        self._bm25 = BM25Okapi([_tokenize(t) for t in texts])

    def query(self, text: str, top_k: int = 6) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(_tokenize(text))
        ranked = sorted(zip(self.ids, scores), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]


def build_from_collection(collection, where: dict | None = None) -> BM25Index:
    # rank_bm25 has no real persistence format — rebuilding in memory from
    # Chroma (the single source of truth) is cheap at our corpus scale, so
    # there's no separate BM25 storage to keep in sync.
    #
    # `where` must be passed to scope this to the current session's
    # paper(s) — e.g. where={"paper_id": {"$in": [...]}}. Without it this
    # builds from every document ever indexed, which would let BM25
    # search across papers the user never asked about, even though dense
    # retrieval (vector_store.query) is correctly scoped. See CLAUDE.md's
    # "Retrieval scoping" decision.
    data = collection.get(where=where, include=["documents"])
    return BM25Index(ids=data["ids"], texts=data["documents"])
