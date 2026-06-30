import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

FAISS_INDEX_PATH = "./faiss_index.bin"
FAISS_META_PATH = "./faiss_meta.pkl"

_model = None

def get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]

def load_and_index_pdf(pdf_path: str):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    chunks = chunk_text(full_text)
    model = get_embedder()
    embeddings = model.encode(chunks, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_META_PATH, "rb") as f:
            all_chunks = pickle.load(f)
        index.add(embeddings)
        all_chunks.extend(chunks)
    else:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        all_chunks = chunks

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    return index, all_chunks

def load_existing_index():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "rb") as f:
        all_chunks = pickle.load(f)
    return index, all_chunks

def retrieve_chunks(query: str, k: int = 5) -> list:
    index, all_chunks = load_existing_index()
    model = get_embedder()
    query_embedding = model.encode([query], normalize_embeddings=True).astype("float32")
    distances, indices = index.search(query_embedding, k)
    results = [all_chunks[i] for i in indices[0] if i < len(all_chunks)]
    return results