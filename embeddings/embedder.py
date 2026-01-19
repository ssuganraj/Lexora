from sentence_transformers import SentenceTransformer
from typing import List

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(
            model_name,
            local_files_only=True
        )

    def embed_texts(self, texts):
        return self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )

    def embed_query(self, query: str):
        """
        Embed a single query string.
        """
        return self.embed_texts([query])[0]
