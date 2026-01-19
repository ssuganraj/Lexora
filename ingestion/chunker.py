# ingestion/chunker.py
# ingestion/chunker.py

from typing import List, Tuple

def chunk_text_with_overlap(
    pages,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[Tuple[int, str]]:
    """
    pages: List[dict] with keys like {"page": int, "text": str}
    returns: List[(page_number, chunk_text)]
    """

    chunks = []

    for page_data in pages:
        page_num = page_data["page"]
        page_text = page_data["text"]

        start = 0
        text_length = len(page_text)

        while start < text_length:
            end = start + chunk_size
            chunk = page_text[start:end]

            if chunk.strip():
                chunks.append((page_num, chunk))

            start += chunk_size - overlap

    return chunks
