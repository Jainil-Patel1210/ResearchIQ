"""Compare all-MiniLM-L6-v2 vs bge-base-en-v1.5 on this machine: real
memory footprint, load/embed time, and a small relevance sanity check.

Run each model in its own process invocation (not both in one script) so
memory measurements aren't skewed by the other model still being loaded.
Results are appended to scripts/benchmark_results.txt as well as printed.

Usage:
    python scripts/benchmark_embedding_models.py minilm
    python scripts/benchmark_embedding_models.py bge-base
"""

import datetime
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import psutil
from sentence_transformers import SentenceTransformer

from pipeline import vector_store

MODELS = {
    "minilm": "all-MiniLM-L6-v2",
    "bge-base": "BAAI/bge-base-en-v1.5",
}

# BGE recommends this instruction prefix on queries only, not on passages.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

RESULTS_PATH = Path(__file__).resolve().parent / "benchmark_results.txt"


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in MODELS:
        print(f"usage: python {sys.argv[0]} <{'|'.join(MODELS)}>")
        sys.exit(1)

    lines = []

    def out(line: str = ""):
        print(line)
        lines.append(line)

    key = sys.argv[1]
    model_name = MODELS[key]
    out(f"=== {datetime.datetime.now().isoformat(timespec='seconds')} ===")
    out(f"model: {model_name}\n")

    mem_before = rss_mb()
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name)
    load_s = time.perf_counter() - t0
    mem_after_load = rss_mb()
    out(f"load time: {load_s:.2f}s")
    out(f"RSS before load: {mem_before:.0f} MB")
    out(f"RSS after load:  {mem_after_load:.0f} MB  (+{mem_after_load - mem_before:.0f} MB)")

    # Real chunk texts already indexed (2 papers, 135 chunks) — avoids
    # re-running Docling/GROBID just for this benchmark.
    collection = vector_store.get_collection()
    data = collection.get(include=["documents"])
    texts = data["documents"]
    out(f"\nembedding {len(texts)} real chunks...")
    t0 = time.perf_counter()
    embeddings = model.encode(texts, show_progress_bar=False)
    embed_s = time.perf_counter() - t0
    mem_after_embed = rss_mb()
    out(f"embed time: {embed_s:.2f}s ({embed_s / len(texts) * 1000:.1f} ms/chunk)")
    out(f"embedding dimension: {embeddings.shape[1]}")
    out(f"RSS after embedding all chunks: {mem_after_embed:.0f} MB "
        f"(+{mem_after_embed - mem_after_load:.0f} MB over model load)")

    # Relevance sanity check: a query about multi-head attention should
    # score closer to a chunk that's actually about it than to an
    # unrelated BERT chunk about masked-language-model pretraining.
    query = "why is multi-head attention useful"
    if key == "bge-base":
        query = BGE_QUERY_PREFIX + query

    metadatas = collection.get(include=["metadatas"])["metadatas"]
    relevant_text = next(
        t for t, m in zip(data["documents"], metadatas)
        if m["paper_id"].startswith("1706") and "multi-head" in t.lower()
    )
    irrelevant_text = next(
        t for t, m in zip(data["documents"], metadatas)
        if m["paper_id"].startswith("1810") and "masked" in t.lower()
    )

    q_vec, rel_vec, irrel_vec = model.encode([query, relevant_text, irrelevant_text])

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    rel_score = cosine(q_vec, rel_vec)
    irrel_score = cosine(q_vec, irrel_vec)
    out(f"\nquery vs relevant (multi-head attention) chunk:   {rel_score:.4f}")
    out(f"query vs irrelevant (BERT masking) chunk:          {irrel_score:.4f}")
    out(f"margin (higher is better discrimination):          {rel_score - irrel_score:.4f}")
    out("")

    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"(results appended to {RESULTS_PATH})")


if __name__ == "__main__":
    main()
