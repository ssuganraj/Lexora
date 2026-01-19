
# qa/retriever.py

import faiss
import numpy as np
from embeddings.embedder import Embedder
from typing import List, Tuple


class Retriever:
    def __init__(self, embedder: Embedder, dim: int = 384):
        self.embedder = embedder
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity
        self.chunks_with_pages: List[Tuple[int, str]] = []
        self.embeddings = None  # store doc embeddings for re-ranking

    def build_index(self, chunks_with_pages):
        self.chunks_with_pages = chunks_with_pages

        texts = [chunk for _, chunk in chunks_with_pages]
        embeddings = self.embedder.embed_texts(texts)

        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings.astype(np.float32))

        self.embeddings = embeddings   # 👈 ADD THIS


    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        candidate_k: int = 10
    ) -> List[Tuple[int, str, float]]:
        """
        Step 1: Retrieve top-N candidates from FAISS
        Step 2: Re-rank using refined cosine similarity
        Step 3: Return top-k results
        """

        # Embed query
        q_emb = self.embedder.embed_texts([query])
        faiss.normalize_L2(q_emb)
        q_emb = q_emb.astype(np.float32)

        # ---- FAISS candidate retrieval ----
        distances, indices = self.index.search(q_emb, candidate_k)
        indices = indices[0]

        # ---- Re-ranking ----
        reranked = []
        for idx in indices:
            page, chunk = self.chunks_with_pages[idx]
            doc_emb = self.embeddings[idx]

            # cosine similarity (since normalized)
            score = float(np.dot(q_emb[0], doc_emb))
            reranked.append((page, chunk, score))

        # Sort by refined score
        reranked.sort(key=lambda x: x[2], reverse=True)

        return reranked[:top_k]
