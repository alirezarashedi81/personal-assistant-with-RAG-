import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


# -----------------------------
# Paths
# -----------------------------
DATA_DIR = "data"
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.json")
FAISS_PATH = os.path.join(DATA_DIR, "faiss_index")
OPENROUTER_API_KEY = "api_key"


# -----------------------------
# Ensure /data directory exists
# -----------------------------
os.makedirs(DATA_DIR, exist_ok=True)


# -----------------------------
# Load bio file
# -----------------------------
def load_bio(path="bio.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# -----------------------------
# Chunk text using LangChain
# -----------------------------
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(text)


# -----------------------------
# Save chunks into /data/chunks.json
# -----------------------------
def save_chunks_json(chunks, save_path=CHUNKS_PATH):
    json_data = [{"id": i, "text": chunk} for i, chunk in enumerate(chunks)]

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"✔ chunks.json saved to: {save_path}")


# -----------------------------
# Build and save FAISS index into /data/faiss_index/
# -----------------------------
def build_faiss_index(text, save_path=FAISS_PATH):
    chunks = chunk_text(text)

    # Save chunk file
    save_chunks_json(chunks)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )

    vector_db = FAISS.from_texts(chunks, embeddings)
    vector_db.save_local(save_path)

    print(f"✔ FAISS index saved to: {save_path}")


# -----------------------------
# Run once
# -----------------------------
if __name__ == "__main__":
    bio_text = load_bio()
    build_faiss_index(bio_text)
