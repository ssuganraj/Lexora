from backend import state
from qa.prompt import build_prompt
from qa.answer import OllamaLLM

llm = OllamaLLM()

MIN_SCORE_THRESHOLD = 0.3  # interview-friendly

def answer_question(question: str):
    if state.retriever is None:
        return "No document uploaded yet.", 0.0, []

    # Retrieve top chunks
    context_chunks = state.retriever.retrieve(question, top_k=3)

    if not context_chunks:
        return "No relevant information found in the document.", 0.0, []

    scores = [score for _, _, score in context_chunks]
    max_score = max(scores)

    # 🚨 Hallucination guard
    if max_score < MIN_SCORE_THRESHOLD:
        return (
            "The document does not contain enough information to answer this question.",
            max_score * 100,
            []
        )

    # Build prompt
    prompt = build_prompt(context_chunks, question)

    # Generate answer
    answer_text = llm.generate(prompt)

    # Sources
    sources = [
        {
            "page": page,
            "score": round(score, 3),
            "snippet": text[:200] + "..."
        }
        for page, text, score in context_chunks
    ]

    confidence = round(max_score * 100, 1)

    return answer_text, confidence, sources
