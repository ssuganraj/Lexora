import faiss
import numpy as np
import os

from embeddings.embedder import Embedder
from qa.retriever import Retriever
from evaluation.retrieval_eval import evaluate_retriever

# ---------- Paths ----------
FAISS_DIR = "data/faiss_index"
index_path = os.path.join(FAISS_DIR, "faiss_index.index")
chunks_path = os.path.join(FAISS_DIR, "chunks_meta.npy")
embeddings_path = os.path.join(FAISS_DIR, "embeddings.npy")

# ---------- Load components ----------
embedder = Embedder()
retriever = Retriever(embedder, dim=384)

# ---------- LOAD persisted data ----------
retriever.index = faiss.read_index(index_path)
retriever.chunks_with_pages = list(
    np.load(chunks_path, allow_pickle=True)
)
retriever.embeddings = np.load(embeddings_path)

# ---------- Eval dataset ----------
eval_data = [
    {
        "question": "What is CRM?",
        "answer_page": 12
    },
    {
        "question": "Explain customer relationship management",
        "answer_page": 12
    }
]

# ---------- Run evaluation ----------
recall = evaluate_retriever(retriever, eval_data, k=3)
print(f"Recall@3: {recall:.2f}")
