# 📄 DocAI-RAG

A **production-style Document Question Answering system** built using **Retrieval-Augmented Generation (RAG)**. DocAI-RAG allows users to upload PDFs and ask natural language questions, with answers grounded strictly in the document content.

---

## ✨ Features

* 📤 Upload and index PDF documents
* 🔍 Semantic search using vector embeddings
* 🧠 Context-aware answers using LLMs (RAG)
* 📊 Confidence score for each answer
* 📄 Reference pages with similarity scores
* ⚡ FastAPI backend + Streamlit frontend
* 🧱 Modular, scalable architecture

---

## 🧠 Architecture Overview

```
User (Streamlit UI)
      ↓
FastAPI Backend
      ↓
PDF Ingestion Pipeline
      ↓
Text Chunking & Cleaning
      ↓
Vector Embeddings
      ↓
Vector Store (Chroma / FAISS)
      ↓
Retriever + LLM
      ↓
Answer + Confidence + Sources
```

---

## 🛠️ Tech Stack

### Frontend

* Streamlit
* Custom CSS (Dark UI)

### Backend

* FastAPI
* Pydantic

### AI / ML

* Retrieval-Augmented Generation (RAG)
* Sentence Embeddings
* Similarity Search

### Vector Stores

* FAISS
* ChromaDB

### Utilities

* Python
* PDF parsing

---

## 📁 Project Structure

```
Doc-AI/
│
├── backend/
│   ├── main.py              # FastAPI entry point
│   ├── ingest.py            # PDF ingestion pipeline
│   ├── qa_service.py        # Question answering logic
│   ├── api/
│   ├── ingestion/
│   ├── qa/
│   ├── vector_store/
│   └── data/
│
├── app.py                   # Streamlit frontend
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 How to Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/DocAI-RAG.git
cd DocAI-RAG
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Start backend

```bash
uvicorn backend.main:app --reload
```

### 5️⃣ Start frontend

```bash
streamlit run app.py
```

---

## 🧪 Example Workflow

1. Upload a PDF document
2. Backend indexes and stores embeddings
3. Ask a question in natural language
4. System retrieves relevant chunks
5. LLM generates grounded answer
6. UI displays answer, confidence, and references

---

## 📌 Use Cases

* Academic notes Q&A
* Technical documentation assistant
* Company policy search
* Interview preparation from PDFs
* Knowledge base querying

---

## 🔒 Answer Grounding

* Answers are **strictly generated from uploaded documents**
* Reference pages and similarity scores are shown
* Prevents hallucination by design

---

## 🔮 Future Improvements

* Multi-document support
* Chat history & memory
* User authentication
* Docker deployment
* Cloud vector DB support
* Better confidence calibration

---

## 👨‍💻 Author

**Sugan Raj**
Final Year CSE | AI & Backend Enthusiast

---

## ⭐ If you like this project

Give it a star ⭐ and feel free to fork or contribute!
