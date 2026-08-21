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
