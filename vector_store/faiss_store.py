import faiss
import os
import pickle
import numpy as np

class FAISSVectorStore:
    def __init__(self, dim: int, index_path="data/faiss_index"):
        self.dim = dim
        self.index_path = index_path
        self.index_file = os.path.join(index_path, "index.faiss")
        self.meta_file = os.path.join(index_path, "chunks.pkl")

        os.makedirs(index_path, exist_ok=True)

        self.index = None
        self.chunks = []

        if self.exists():
            self.load()
        else:
            self.index = faiss.IndexFlatIP(dim)  # cosine similarity

    def exists(self):
        return os.path.exists(self.index_file) and os.path.exists(self.meta_file)

    def add(self, embeddings: np.ndarray, chunks: list):
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def save(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self):
        self.index = faiss.read_index(self.index_file)
        with open(self.meta_file, "rb") as f:
            self.chunks = pickle.load(f)

    def search(self, query_embedding: np.ndarray, top_k=3):
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)
        return [self.chunks[i] for i in indices[0]]
