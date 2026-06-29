import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from retriever import load_and_index_pdf, load_existing_vectorstore
from rag_graph import build_graph

load_dotenv()

st.set_page_config(
    page_title="Self-Healing RAG Assistant",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #0f0f0f; }
    .stApp { background-color: #0f0f0f; color: #ffffff; }
    .verdict-pass { 
        background-color: #1a3a1a; 
        border-left: 4px solid #00ff88; 
        padding: 10px; 
        border-radius: 5px;
        color: #00ff88;
    }
    .verdict-fail { 
        background-color: #3a1a1a; 
        border-left: 4px solid #ff4444; 
        padding: 10px; 
        border-radius: 5px;
        color: #ff4444;
    }
    .trace-box {
        background-color: #1a1a2e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 15px;
        font-family: monospace;
        font-size: 13px;
    }
    .answer-box {
        background-color: #16213e;
        border-left: 4px solid #6c63ff;
        padding: 20px;
        border-radius: 8px;
        font-size: 15px;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Self-Healing RAG Assistant")
st.markdown("*Upload a PDF — Ask anything — The system critiques and self-corrects its own answers*")
st.divider()

with st.sidebar:
    st.header("📄 Document Upload")
    st.markdown("Upload one or more PDF files to build your knowledge base.")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("📥 Index Documents", use_container_width=True):
            with st.spinner("Indexing documents... please wait"):
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    try:
                        load_and_index_pdf(tmp_path)
                    finally:
                        os.unlink(tmp_path)
            st.success(f"✅ {len(uploaded_files)} document(s) indexed successfully!")
            st.session_state["docs_indexed"] = True

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("""
    1. 📥 Upload & index PDF
    2. ❓ Ask a question
    3. 🔍 System retrieves chunks
    4. 🤖 LLM generates answer
    5. 🧐 Critic evaluates answer
    6. 🔄 If hallucinated → retry
    7. ✅ Final grounded answer
    """)

    st.divider()
    st.markdown("**Settings**")
    show_trace = st.toggle("Show reasoning trace", value=True)
    show_sources = st.toggle("Show source chunks", value=False)

chroma_exists = os.path.exists("./chroma_db")
if not chroma_exists and not st.session_state.get("docs_indexed"):
    st.info("👈 Please upload and index a PDF document from the sidebar to get started.")
    st.stop()

st.subheader("💬 Ask a Question")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])
    with st.chat_message("assistant"):
        st.markdown(f'<div class="answer-box">{chat["answer"]}</div>', unsafe_allow_html=True)

question = st.chat_input("Ask anything about your uploaded documents...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Retrieving → 🤖 Generating → 🧐 Critiquing..."):
            try:
                graph = build_graph()
                initial_state = {
                    "question": question,
                    "context": [],
                    "answer": "",
                    "verdict": "",
                    "reason": "",
                    "reformulated_query": question,
                    "retry_count": 0,
                    "final_answer": "",
                    "trace_log": []
                }
                result = graph.invoke(initial_state)

                st.markdown(f'<div class="answer-box">{result["final_answer"]}</div>', unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    if result["verdict"] == "PASS":
                        st.markdown('<div class="verdict-pass">✅ Critic: PASS — Answer is grounded</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="verdict-fail">⚠️ Max retries reached — Best available answer shown</div>', unsafe_allow_html=True)
                with col2:
                    retries = result.get("retry_count", 0)
                    st.metric("Retries used", f"{retries} / 3")

                if show_trace and result.get("trace_log"):
                    st.markdown("**🔍 Reasoning Trace:**")
                    trace_html = "<br>".join([f"→ {log}" for log in result["trace_log"]])
                    st.markdown(f'<div class="trace-box">{trace_html}</div>', unsafe_allow_html=True)

                if show_sources and result.get("context"):
                    with st.expander("📄 Source Chunks Used"):
                        for i, doc in enumerate(result["context"]):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.text(doc.page_content)
                            st.divider()

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": result["final_answer"]
                })

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Make sure your PDF is indexed and GROQ_API_KEY is set correctly.")