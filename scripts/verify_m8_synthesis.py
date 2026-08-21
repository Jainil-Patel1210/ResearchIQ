"""Smoke test for M8 — cross-paper synthesis
(pipeline/retriever.retrieve_cross_paper + pipeline/generator.generate_cross_paper_answer).

Tests:
1. A genuine cross-paper comparison question -> answer that attributes
   claims to the correct paper, with citations resolving back to the right
   paper_id for each excerpt number.
2. Every paper_id passed in is retrieved from and represented in the
   prompt, even though BERT (author-year style) has no resolvable inline
   citations of its own (M8 decision: no relevance-based filtering).
3. A question neither paper can answer -> "not found", not hallucination.

Usage:
    python scripts/verify_m8_synthesis.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from pipeline import generator, retriever

ATTENTION = "1706.03762v7"  # Attention Is All You Need
BERT = "1810.04805v2"       # BERT
PAPER_IDS = [ATTENTION, BERT]


def ask(question: str, paper_ids: list[str]):
    print(f"\n{'=' * 70}\nQ: {question}\nPapers: {paper_ids}\n{'=' * 70}")
    chunks_by_paper = retriever.retrieve_cross_paper(question, paper_ids=paper_ids, top_k_per_paper=3)

    for paper_id, chunks in chunks_by_paper.items():
        print(f"\n[retrieved for {paper_id}]: {len(chunks)} chunk(s)")
        for c in chunks:
            print(f"    section={c.section!r} page={c.page}")

    result = generator.generate_cross_paper_answer(question, chunks_by_paper)

    print("\nANSWER:")
    print(result.answer)

    print("\nCITATIONS:")
    for n, chunk in result.citations.items():
        print(f"  [{n}] paper={chunk.paper_id} section={chunk.section!r} page={chunk.page}")


def main():
    # Genuinely answerable from both papers, and should distinguish them:
    # Transformer is an encoder-decoder trained on translation; BERT is an
    # encoder-only model trained with masked-language-modeling pretraining.
    ask(
        "How do the model architectures and training objectives of these "
        "two papers differ?",
        PAPER_IDS,
    )

    # Neither paper covers this -> should refuse, not hallucinate.
    ask(
        "What dataset and training setup did the GPT-3 paper use?",
        PAPER_IDS,
    )


if __name__ == "__main__":
    main()
