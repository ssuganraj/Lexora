from ingestion.loader import load_pdf_pages
from ingestion.chunker import chunk_text_with_overlap
from embeddings.embedder import Embedder
from qa.retriever import Retriever
from backend import state

def ingest_pdf(file_path: str):
    # 1. Load PDF
    pages = load_pdf_pages(file_path)

    # 2. Chunk text
    chunks_with_pages = chunk_text_with_overlap(pages)
    # expected: List[(page_number, chunk_text)]

    # 3. Create embedder
    embedder = Embedder()

    # 4. Create retriever + build FAISS index
    retriever = Retriever(embedder)
    retriever.build_index(chunks_with_pages)

    # 5. Store retriever globally
    state.retriever = retriever
