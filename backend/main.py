# backend/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import tempfile
import re

from backend.ingest import ingest_pdf
from backend.qa_service import answer_question

app = FastAPI()


# ---------- Utils ----------
def strip_html(text: str) -> str:
    """Remove any HTML tags if accidentally generated"""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", "", text)
    return clean.strip()


# ---------- PDF Upload Endpoint ----------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            ingest_pdf(tmp.name)

        return {"status": "PDF indexed successfully"}

    except Exception as e:
        print("UPLOAD ERROR:", e)
        raise HTTPException(status_code=500, detail="PDF indexing failed")


# ---------- Request body schema ----------
class QuestionRequest(BaseModel):
    question: str


# ---------- Ask Question Endpoint ----------
@app.post("/ask")
def ask(request: QuestionRequest):
    try:
        answer, confidence, sources = answer_question(request.question)

        # ✅ HARD GUARANTEES
        answer = strip_html(str(answer))
        confidence = float(confidence)

        clean_sources = []
        for src in sources:
            clean_sources.append({
                "page": src.get("page"),
                "score": float(src.get("score", 0))
            })

        return {
            "answer": answer,          # ✅ plain text only
            "confidence": confidence,  # ✅ float
            "sources": clean_sources   # ✅ safe JSON
        }

    except Exception as e:
        print("ASK ERROR:", e)
        raise HTTPException(status_code=500, detail="Question answering failed")
