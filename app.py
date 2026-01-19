import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Doc-AI",
    page_icon="📄",
    layout="wide"
)

# ---------------- Minimal CSS ----------------
st.markdown(
    """
    <style>
        html, body, [class*="css"] {
            font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .block-container {
            padding-top: 2rem;
            max-width: 1100px;
        }

        .subtle {
            color: #6b7280;
            font-size: 14px;
        }

        .card {
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 20px;
            background: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Session State ----------------
st.session_state.setdefault("indexed", False)
st.session_state.setdefault("filename", "")
st.session_state.setdefault("answer", None)

# ---------------- Header ----------------
st.title("📄 Doc-AI")
st.markdown(
    "Ask questions directly from your PDF using AI-powered document understanding.",
    help="Upload a document and query it using semantic search + LLMs."
)

st.divider()

# ===================== UPLOAD SECTION =====================
if not st.session_state["indexed"]:
    st.subheader("1. Upload document")

    uploaded_file = st.file_uploader(
        "PDF file",
        type=["pdf"]
    )

    if uploaded_file:
        if st.button("Index document", type="primary"):
            with st.spinner("Indexing document…"):
                try:
                    r = requests.post(
                        f"{BACKEND_URL}/upload",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "application/pdf"
                            )
                        },
                        timeout=60
                    )
                    r.raise_for_status()
                    st.session_state["indexed"] = True
                    st.session_state["filename"] = uploaded_file.name
                    st.success("Document indexed successfully")
                    st.rerun()
                except Exception as e:
                    st.error("Failed to index document")

# ===================== QA SECTION =====================
else:
    st.subheader("2. Ask questions")

    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"**Active document:** `{st.session_state['filename']}`")
    with col2:
        if st.button("Reset"):
            st.session_state.clear()
            st.rerun()

    question = st.text_input(
        "Your question",
        placeholder="e.g. What is CRM?"
    )

    if question:
        if st.button("Ask", type="primary"):
            with st.spinner("Generating answer…"):
                try:
                    r = requests.post(
                        f"{BACKEND_URL}/ask",
                        json={"question": question},
                        timeout=60
                    )
                    r.raise_for_status()
                    st.session_state["answer"] = r.json()
                except Exception:
                    st.error("Backend error")

    # ===================== ANSWER =====================
    if st.session_state["answer"]:
        data = st.session_state["answer"]

        answer_text = data.get("answer", "")
        confidence = float(data.get("confidence", 0))
        sources = data.get("sources", [])

        st.divider()
        st.subheader("Answer")

        # ---- Answer Card ----
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(answer_text)
        st.markdown('</div>', unsafe_allow_html=True)

        # ---- Confidence ----
        st.markdown("**Confidence**")
        st.progress(min(confidence / 100, 1.0))
        st.markdown(f"<span class='subtle'>{confidence:.1f}% match with document context</span>",
                    unsafe_allow_html=True)

        # ---- Sources ----
        if sources:
            st.subheader("Sources")
            for src in sources:
                with st.container(border=True):
                    st.markdown(
                        f"""
                        **Page:** {src.get("page", "?")}  
                        **Similarity:** `{src.get("score", 0):.3f}`
                        """
                    )
