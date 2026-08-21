"""Smoke test for M7 — generation (pipeline/generator.py).

Tests two things:
1. A genuinely answerable question -> grounded, cited answer.
2. A question the paper can't answer -> model should say so, not hallucinate.

Usage:
    python scripts/verify_m7_generation.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from pipeline import generator, retriever

PAPER_ID = "1706.03762v7"  # Attention Is All You Need


def ask(question: str):
    print(f"\n{'=' * 70}\nQ: {question}\n{'=' * 70}")
    chunks = retriever.retrieve(question, paper_ids=[PAPER_ID], top_k=5)
    result = generator.generate_answer(question, chunks)

    print("\nANSWER:")
    print(result.answer)

    print("\nCITATIONS:")
    for n, chunk in result.citations.items():
        print(f"  [{n}] paper={chunk.paper_id} section={chunk.section!r} page={chunk.page}")


def main():
    # Should be well-answerable from the paper's own content.
    ask("What is multi-head attention and why is it used instead of a single attention head?")

    # Should NOT be answerable from this paper — tests the "say not found"
    # instruction rather than falling back to general knowledge.
    ask("What year was the BERT paper published and who are its authors?")


if __name__ == "__main__":
    main()
