# Self-Healing RAG Pipeline

A Retrieval-Augmented Generation (RAG) system that retrieves relevant information from documents, generates answers using an LLM, and evaluates its own output before returning it to the user. It also includes custom error handling and logging for retrieval failures.

**Live Demo:** [self-healing-rag-pipeline](https://self-healing-rag-pipeline-1.onrender.com)

---

## The Problem

Most basic RAG systems follow a linear workflow:

**Retrieve → Generate → Return**

Without a verification step, these systems may return hallucinated answers or responses that are not sufficiently grounded in the source documents. They can also fail unexpectedly when document retrieval or vector search encounters an error.

This project addresses these challenges through a self-correction loop and controlled retrieval error handling. A critic evaluates generated answers, while custom exceptions and logging help identify and manage retrieval failures.

---

## How It Works

```text
                 User Question
                       |
                       v
              +----------------+
              |    Retrieve    |
              |  FAISS Search  |
              +----------------+
                       |
                       v
              +----------------+
              |    Generate    |
              |   LLM / Groq   |
              +----------------+
                       |
                       v
              +----------------+
              |     Critic     |
              | Grounding Check|
              +----------------+
                       |
                +------+------+
                |             |
               PASS          FAIL
                |             |
                v             v
          Return Answer   Reformulate Query
                              |
                              v
                         Retry Retrieval
                         (up to 3 times)
                              |
                              v
                         Honest Fallback
```

### Pipeline Stages

1. **Document Ingestion** – PDF files are loaded and divided into smaller text chunks using `RecursiveCharacterTextSplitter`.
2. **Embedding Generation** – Text chunks are converted into vector embeddings using the `all-MiniLM-L6-v2` model.
3. **Vector Indexing** – FAISS stores the embeddings and supports similarity-based retrieval.
4. **Answer Generation** – Llama 3.3 70B Versatile, accessed through the Groq API, generates an answer using retrieved context.
5. **Answer Critique** – A critic evaluates whether the generated answer is relevant to the question and grounded in the retrieved context.
6. **Self-Healing** – If the answer fails evaluation, the query is reformulated and the retrieval-generation-critique process is retried, up to the configured limit.
7. **Honest Fallback** – If the answer still fails evaluation after the retries, the system can return an insufficient-information response rather than presenting an unsupported answer.

---

## Error Handling and Logging

The project includes a dedicated `retrieval_error_handling.py` module to define custom exceptions and centralize error logging for retrieval operations.

### Custom Exceptions

| Exception          | Purpose                                       |
| ------------------ | --------------------------------------------- |
| `RAGPipelineError` | Base exception for controlled pipeline errors |
| `RetrievalError`   | Represents errors during document retrieval   |
| `GenerationError`  | Represents errors during answer generation    |
| `CriticError`      | Represents errors during answer evaluation    |

### Retrieval Error Handling

The retrieval function in `retriever.py` uses `try-except` to handle errors during retrieval.

* **Input validation:** Checks that the query is not empty and that `k` is a positive integer.
* **Error logging:** Uses Python's logging module to record retrieval failures for debugging.
* **Custom exceptions:** Wraps unexpected retrieval errors in a `RetrievalError` while preserving the original exception as the cause.
* **Controlled error messages:** Raises a clear custom exception instead of exposing internal error details directly to the user.

This approach separates internal debugging information from user-facing errors and provides a consistent way for other pipeline components to identify retrieval failures.

> The retrieval module handles errors within its retrieval function. Other pipeline stages, such as generation and critique, can use the same exception-handling approach as their error handling is expanded.

---

## Tech Stack

| Component      | Technology                        |
| -------------- | --------------------------------- |
| LLM            | Llama 3.3 70B Versatile           |
| LLM API        | Groq                              |
| Embeddings     | `all-MiniLM-L6-v2`                |
| Vector Store   | FAISS                             |
| PDF Processing | PyPDF                             |
| Text Splitting | Recursive Character Text Splitter |
| Frontend       | Streamlit                         |
| Error Handling | Python custom exceptions          |
| Logging        | Python `logging` module           |
| Deployment     | Render                            |

---

## Features

* Upload PDF documents and ask questions about their content.
* Retrieve relevant document chunks using FAISS similarity search.
* Generate context-based answers using the Groq API.
* Evaluate generated answers through a critic step.
* Automatically reformulate queries and retry when an answer is not sufficiently grounded.
* Provide an honest fallback when the system cannot produce a grounded answer.
* Validate retrieval inputs before running similarity search.
* Use custom exceptions to represent retrieval and other pipeline errors.
* Log internal retrieval failures to support debugging.
* View the retrieval, generation, and critique process through the application's reasoning trace.
* Monitor retry attempts through the retry counter.
* Access the deployed application through a public demo link.

---

## Running Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Vicky-Vrishni/Self-Healing-RAG-Pipeline.git
cd Self-Healing-RAG-Pipeline
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API Key

Create a `.env` file in the project's root directory and add:

```env
GROQ_API_KEY=your_groq_api_key
```

Get an API key from [Groq Console](https://console.groq.com/).

**Security:** Never commit your `.env` file, API keys, or other secrets to GitHub. Make sure `.env` is listed in `.gitignore`.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will usually be available at:

```text
http://localhost:8501
```

---

## Project Structure

```text
Self-Healing-RAG-Pipeline/
│
├── app.py                       # Streamlit user interface
├── retriever.py                 # PDF loading, chunking, embeddings and retrieval
├── retrieval_error_handling.py  # Custom exceptions and error logging
├── critic.py                    # LLM-based answer evaluation
├── rag_graph.py                 # Self-healing RAG workflow
├── requirements.txt             # Project dependencies
├── .env                         # API key configuration (not committed)
├── .gitignore                   # Files excluded from Git
└── README.md                    # Project documentation
```

---

## Security Considerations

* API keys and sensitive configuration should be stored in environment variables and excluded from version control.
* Internal exceptions and tracebacks should be logged for debugging rather than exposed to end users.
* FAISS index files loaded with `allow_dangerous_deserialization=True` must be treated as trusted input. Only load indexes from sources you control and protect.

---

## Why This Project?

A reliable RAG application needs to do more than retrieve documents and generate text. It should also evaluate answer grounding, recover from unsuccessful retrieval attempts, and provide useful error information for debugging.

This project explores an agentic, stateful RAG workflow that combines retrieval, generation, critique, query reformulation, and custom error handling. It demonstrates how verification and controlled failure handling can be incorporated into a document-question-answering system.

---

## Future Improvements

* Extend consistent custom error handling to generation and critic stages.
* Add automated unit and integration tests for retrieval and error scenarios.
* Improve recovery strategies for missing or corrupted vector indexes.
* Track retrieval accuracy, answer grounding, and response quality.
* Add monitoring and structured logging for pipeline failures.
* Improve evaluation using a representative set of questions and source documents.
* Try to improve API errors.

---

## Author

**Vicky Kumar**

* [GitHub](https://github.com/Vicky-Vrishni)
* [LinkedIn](https://linkedin.com/in/vicky-kumar-167189323)
