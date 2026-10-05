
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from retrieval_error_handling import (
    RetrievalError,
    log_pipeline_error
)

FAISS_INDEX_DIR = "./faiss_index"
_embeddings = None


def get_embedder():
    global _embeddings

    #HuggingFace sentence Transformer
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

    return _embeddings


# Upload user private data
def load_and_index_pdf(pdf_path: str):
    """Load a PDF, split it, embed it and add it to FAISS."""

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)
    embeddings = get_embedder()

    if os.path.exists(FAISS_INDEX_DIR):
        vectorstore = FAISS.load_local(
            FAISS_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

    vectorstore.save_local(FAISS_INDEX_DIR)
    return vectorstore


def load_existing_index():
    embeddings = get_embedder()

    return FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )


def retrieve_chunks(query: str, k: int = 5) -> list:
    """Return the top-k most relevant chunk texts."""

    try:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty") #This Execptional handling code try to handle the error if user not provide relevenat pdf.

        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError("k must be a positive integer")

        vectorstore = load_existing_index()

        results = vectorstore.similarity_search(
            query,
            k=k
        )

        return [doc.page_content for doc in results]

    except RetrievalError:
        raise

    except Exception as error:
        log_pipeline_error("retrieval", error)

        raise RetrievalError(
            "Document retrieval failed"
        ) from error


