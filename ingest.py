import os
import pickle
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
CHROMA_PATH = "chroma_db"
BM25_DOCS_PATH = "bm25_docs.pkl"

def ingest_documents():
    os.makedirs(DATA_DIR, exist_ok=True)
    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()

    if not documents:
        print(f"No PDFs found in {DATA_DIR}. Please add some PDFs.")
        return

    print(f"Loaded {len(documents)} pages from PDFs.")

    # Apply chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    # Modify metadata for citation purposes if needed
    for i, chunk in enumerate(chunks):
        if "source" in chunk.metadata:
            chunk.metadata["source_doc"] = os.path.basename(chunk.metadata["source"])
            chunk.metadata["chunk_id"] = i

    # Persist to Chroma
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Saving to ChromaDB...")
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"ChromaDB persistence complete. Path: {CHROMA_PATH}")

    # Save chunks to a pickle for BM25 Retriver later
    print("Saving document chunks for BM25...")
    with open(BM25_DOCS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_documents()
