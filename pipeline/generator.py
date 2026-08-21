import os
from dataclasses import dataclass

from dotenv import load_dotenv
from groq import Groq

from pipeline import citation_resolver
from pipeline.retriever import RetrievedChunk

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

_SYSTEM_PROMPT = """You are a research assistant answering questions about academic papers.

Rules:
- Answer using ONLY the excerpts provided below. Do not use outside/general knowledge.
- If the excerpts do not contain enough information to answer, say clearly: \
"This is not found in the provided papers." Do not guess or fill gaps.
- Prefer extracting or closely paraphrasing the source text over creative synthesis.
- Cite every factual claim using the excerpt number it came from, in plain \
ASCII square brackets exactly like [1] — not full-width or other bracket \
characters. A claim can cite more than one excerpt, e.g. [1][2]."""

_CROSS_PAPER_SYSTEM_PROMPT = """You are a research assistant answering questions that span multiple academic papers.

The excerpts below are grouped by paper (each group is headed by the paper's title).

Rules:
- Answer using ONLY the excerpts provided below. Do not use outside/general knowledge.
- If the excerpts do not contain enough information to answer, say clearly: \
"This is not found in the provided papers." Do not guess or fill gaps.
- Prefer extracting or closely paraphrasing the source text over creative synthesis.
- Every claim must make clear which paper it comes from — refer to the paper \
by its title (not just a citation number), since the reader is comparing papers.
- When papers agree, say so explicitly. When they differ or use different \
terminology for a similar idea, point out the difference explicitly rather \
than blending them into one claim.
- Cite every factual claim using the excerpt number it came from, in plain \
ASCII square brackets exactly like [1] — not full-width or other bracket \
characters. A claim can cite more than one excerpt, e.g. [1][2]."""

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


@dataclass
class GeneratedAnswer:
    answer: str
    citations: dict[int, RetrievedChunk]


def _format_excerpt(number: int, chunk: RetrievedChunk) -> str:
    title = citation_resolver.load_paper_title(chunk.paper_id) or chunk.paper_id
    header = f"[{number}] (Paper: {title!r}, Section: {chunk.section!r}, p.{chunk.page})"
    lines = [header, chunk.text]
    if chunk.citations:
        cited = "; ".join(f"{c.authors[0] if c.authors else '?'} et al. {c.year} {c.title!r}"
                           for c in chunk.citations[:3])
        lines.append(f"(this excerpt itself cites: {cited})")
    return "\n".join(lines)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    excerpts = "\n\n".join(_format_excerpt(i, c) for i, c in enumerate(chunks, start=1))
    return f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
    if not chunks:
        return GeneratedAnswer(answer="This is not found in the provided papers.", citations={})

    user_prompt = build_prompt(question, chunks)
    response = _get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    answer = response.choices[0].message.content

    citations = {i: chunk for i, chunk in enumerate(chunks, start=1)}
    return GeneratedAnswer(answer=answer, citations=citations)


def _flatten_by_paper(chunks_by_paper: dict[str, list[RetrievedChunk]]) -> list[RetrievedChunk]:
    return [chunk for chunks in chunks_by_paper.values() for chunk in chunks]


def build_cross_paper_prompt(question: str, chunks_by_paper: dict[str, list[RetrievedChunk]]) -> str:
    """Groups excerpts under a paper-title header (so the model can see the
    grouping directly, not just infer it from repeated titles in each
    excerpt header) while keeping one global [N] numbering across all
    papers combined — same citation format generate_answer() already uses,
    so downstream citation parsing doesn't need a separate code path."""
    lines = []
    number = 1
    for paper_id, chunks in chunks_by_paper.items():
        if not chunks:
            continue
        title = citation_resolver.load_paper_title(paper_id) or paper_id
        lines.append(f"=== Paper: {title!r} ===")
        for chunk in chunks:
            lines.append(_format_excerpt(number, chunk))
            number += 1
    excerpts = "\n\n".join(lines)
    return f"Excerpts (grouped by paper):\n\n{excerpts}\n\nQuestion: {question}"


def generate_cross_paper_answer(
    question: str, chunks_by_paper: dict[str, list[RetrievedChunk]]
) -> GeneratedAnswer:
    """Synthesis call for questions spanning multiple papers (M8). Every
    paper_id passed in retriever.retrieve_cross_paper()'s output is included
    here too — no relevance filtering, the caller already decided which
    papers are in scope."""
    all_chunks = _flatten_by_paper(chunks_by_paper)
    if not all_chunks:
        return GeneratedAnswer(answer="This is not found in the provided papers.", citations={})

    user_prompt = build_cross_paper_prompt(question, chunks_by_paper)
    response = _get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _CROSS_PAPER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    answer = response.choices[0].message.content

    citations = {i: chunk for i, chunk in enumerate(all_chunks, start=1)}
    return GeneratedAnswer(answer=answer, citations=citations)
