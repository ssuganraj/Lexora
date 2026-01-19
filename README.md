# 📄 Doc-AI — Document Question Answering System

Doc-AI is a Retrieval-Augmented Generation (RAG) based system that enables users to ask natural language questions over uploaded PDF documents and receive grounded, context-aware answers.

---

## 🚀 Features
- PDF ingestion and intelligent text chunking
- Semantic search using sentence-transformer embeddings
- Retrieval-Augmented Generation (RAG) pipeline
- Local LLM inference using Ollama (Mistral)
- Confidence scoring and source attribution
- Clean separation of frontend (Streamlit) and backend (FastAPI)

---

## 🧠 Architecture
Streamlit (UI) → FastAPI (Backend) → Vector Store → Ollama LLM

---

## 🛠️ Tech Stack
- Python
- FastAPI
- Streamlit
- Sentence Transformers
- Ollama (Mistral)
- RAG Architecture

---

## ▶️ How to Run

### Backend
```bash
uvicorn backend.main:app --reload
