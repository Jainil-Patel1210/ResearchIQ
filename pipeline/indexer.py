from pipeline import chunker, citation_resolver, embedder, vector_store


def index_pdf(pdf_path, paper_id: str) -> int:
    chunks = chunker.chunk_pdf(pdf_path, paper_id=paper_id)
    embeddings = embedder.embed([c.text for c in chunks])
    vector_store.add_chunks(chunks, embeddings)
    citation_resolver.fetch_and_save_references(pdf_path, paper_id)
    return len(chunks)
