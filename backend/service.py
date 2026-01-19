import faiss
import numpy as np
import os
from embeddings.embedder import Embedder
from ingestion.loader import load_pdf_pages
from ingestion.chunker import chunk_text_with_overlap
from qa.retriever import Retriever
from qa.prompt import build_prompt
from qa.answer import OllamaLLM

FAISS_DIR = "data/faiss_index"
os.makedirs(FAISS_DIR, exist_ok=True)

index_path = os.path.join(FAISS_DIR, "faiss_index.index")
chunks_path = os.path.join(FAISS_DIR, "chunks_meta.npy")
embeddings_path = os.path.join(FAISS_DIR, "embeddings.npy")

embedder = Embedder()
retriever = Retriever(embedder, dim=384)
llm = OllamaLLM(device="cpu")


def build_index(pdf_path: str):
    pages = load_pdf_pages(pdf_path)

    chunks_with_pages = chunk_text_with_overlap(
        pages,
        chunk_size=600,
        overlap=150
    )

    retriever.build_index(chunks_with_pages)

    faiss.write_index(retriever.index, index_path)
    np.save(chunks_path, np.array(chunks_with_pages, dtype=object))
    np.save(embeddings_path, retriever.embeddings)


def load_index():
    retriever.index = faiss.read_index(index_path)
    retriever.chunks_with_pages = list(
        np.load(chunks_path, allow_pickle=True)
    )
    retriever.embeddings = np.load(embeddings_path)


def answer_question(question: str):
    results = retriever.retrieve(question, top_k=3)

    context = ""
    scores = []
    sources = []

    for page, chunk, score in results:
        context += f"[Page {page}]\n{chunk}\n\n"
        scores.append(score)
        sources.append({"page": page, "score": score})

    confidence = min(max(scores), 1.0) * 100
    prompt = build_prompt(context, question)
    answer = llm.generate(prompt)

    return answer, confidence, sources
