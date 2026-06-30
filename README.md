# Self-Healing RAG Pipeline

A Retrieval-Augmented Generation (RAG) system that doesn't just retrieve-and-generate — it **critiques its own output and retries** when the answer isn't properly grounded in the source documents.

** Live Demo:** [self-healing-rag-pipeline.streamlit.app](https://self-healing-rag-pipeline-prnkgfemfs4xml3tyktnxh.streamlit.app)

---

##  The Problem

Most RAG systems follow a simple linear flow: retrieve chunks → generate an answer → return it to the user. There's no verification step, which means hallucinated or poorly-grounded answers slip through silently.

This project adds a **self-correction loop**: every generated answer is evaluated by a critic agent before being shown to the user. If the critic detects hallucination or insufficient grounding, the system automatically reformulates the query and retries — up to 3 times — before gracefully admitting it doesn't have enough information.

---

## How It Works

```
User Question
      │
      ▼
┌─────────────┐
│  Retrieve   │  ── FAISS vector search over document chunks
└─────────────┘
      │
      ▼
┌─────────────┐
│  Generate   │  ── LLM (Llama 3.3 70B via Groq) answers using retrieved context
└─────────────┘
      │
      ▼
┌─────────────┐
│   Critic    │  ── Second LLM call evaluates: Is this grounded? Does it answer the question?
└─────────────┘
      │
      ├── PASS ──────────────► Return answer to user
      │
      └── FAIL ──► Reformulate query ──► Retry (max 3 times) ──► If still failing, return honest "I don't know"
```

1. **Retrieve** – Relevant chunks are pulled from a FAISS vector index built from the uploaded PDF(s).
2. **Generate** – An LLM answers the question using only the retrieved context.
3. **Critique** – A second LLM call acts as a strict evaluator, checking whether the answer is actually grounded in the context or hallucinated.
4. **Self-heal** – If the critic rejects the answer, the system reformulates the search query and retries — instead of blindly returning a possibly wrong answer.
5. **Honest fallback** – After 3 failed attempts, the system explicitly tells the user it doesn't have enough information, rather than making something up.

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Llama 3.3 70B Versatile (via [Groq](https://groq.com) API) |
| Embeddings | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Vector Store | FAISS |
| PDF Processing | PyPDF |
| Frontend | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## Features

- Upload any PDF and chat with it
- Automatic self-correction when answers aren't grounded
- Transparent reasoning trace — see exactly what the system retrieved, generated, and critiqued at each step
- Retry counter showing how many self-healing attempts were used
- Refuses to hallucinate — explicitly says "I don't know" instead of guessing
- Fully deployed with a shareable public link

---

## Running Locally

```bash
# Clone the repository
git clone https://github.com/Vicky-Vrishni/Self-Healing-RAG-Pipeline.git
cd Self-Healing-RAG-Pipeline

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key
echo GROQ_API_KEY=your_key_here > .env

# Run the app
streamlit run app.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

---

## Project Structure

```
Self-Healing-RAG-Pipeline/
├── app.py            # Streamlit UI
├── retriever.py       # PDF chunking, embedding, and FAISS indexing
├── critic.py           # LLM-based answer evaluation
├── rag_graph.py         # Self-healing retrieval-generation-critique loop
├── requirements.txt
└── .env                  # GROQ_API_KEY (not committed)
```

---

## Why This Project

Production RAG systems need more than a happy-path demo — they need to handle the case where retrieval fails or the LLM hallucinates. This project demonstrates an agentic, stateful approach to that problem: treating answer generation as a loop with verification and retry logic, rather than a single one-shot call.

---

## Author

**Vicky Kumar**
[GitHub](https://github.com/Vicky-Vrishni) · [LinkedIn](https://linkedin.com/in/vicky-kumar-167189323)