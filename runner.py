# TZyGewQa6i3pdPovVUa193zQiyUhAbhLwCkO72mI
#data\\raw\\IM.pdf


# runner.py
# from answer.cohere_llm import CohereLLM
from ingestion.loader import load_pdf
from ingestion.cleaner import clean_text
from ingestion.chunker import chunk_text
from embeddings.embedder import Embedder
from vector_store.chroma_store import ChromaVectorStore
from qa.retriever import Retriever
from qa.prompt import build_prompt
# from qa.answer import OllamaLLM
from qa.local_llm import OllamaLLM

# -----------------------------
# 1️⃣ Load and clean PDF
# -----------------------------
pdf_path = "data\\raw\\IM.pdf"  # Replace with your PDF path
raw_text = load_pdf(pdf_path)
cleaned_text = clean_text(raw_text)

print(f"Cleaned text excerpt:\n{cleaned_text[:200]}...\n")

# -----------------------------
# 2️⃣ Chunk text
# -----------------------------
chunks = chunk_text(cleaned_text)
print(f"Total chunks: {len(chunks)}")

# -----------------------------
# 3️⃣ Generate embeddings
# -----------------------------
embedder = Embedder()
embeddings = embedder.embed_texts(chunks)

# -----------------------------
# 4️⃣ Store in vector DB
# -----------------------------
store = ChromaVectorStore()
store.add_documents(chunks, embeddings)

# -----------------------------
# 5️⃣ Initialize retriever and LLM
# -----------------------------
retriever = Retriever()
# llm = CohereLLM(api_key="TZyGewQa6i3pdPovVUa193zQiyUhAbhLwCkO72mI")  # Put your Cohere key here
llm = OllamaLLM(model="mistral")  # Using local Ollama LLM
# -----------------------------
# 6️⃣ Example query
# -----------------------------
query = "What is a office automation system?"
print(f"\nQuery: {query}\n")

# Embed query
query_embedding = embedder.embed_texts([query])[0]

# Retrieve top-k chunks
top_chunks = retriever.retrieve(query_embedding, top_k=3)

print("\n--- Retrieved Chunks ---\n")
for idx, c in enumerate(top_chunks, start=1):
    print(f"[Chunk {idx}] {c[:300]}...\n")

# Build grounded prompt
prompt = build_prompt(top_chunks, query)

# -----------------------------
# 7️⃣ Generate answer
# -----------------------------
answer = llm.generate(prompt)
print("\n--- Answer ---\n")
print(answer)
