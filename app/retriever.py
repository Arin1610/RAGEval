import os
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.ingest import load_papers, chunk_documents

VECTORSTORE_PATH = "data/vectorstore"

def get_embeddings():
    """Load HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

def build_vectorstore() -> FAISS:
    """Build FAISS vectorstore from papers."""
    print("Loading and chunking papers...")
    docs = load_papers()
    chunks = chunk_documents(docs)
    
    print("\nBuilding FAISS vectorstore...")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save locally so we don't rebuild every time
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"✅ Vectorstore saved to {VECTORSTORE_PATH}")
    return vectorstore

def load_vectorstore() -> FAISS:
    """Load existing vectorstore or build if not exists."""
    embeddings = get_embeddings()
    if os.path.exists(VECTORSTORE_PATH):
        print("Loading existing vectorstore...")
        return FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("No vectorstore found — building from scratch...")
        return build_vectorstore()

if __name__ == "__main__":
    vectorstore = build_vectorstore()
    
    # Quick test
    results = vectorstore.similarity_search("What is attention mechanism?", k=3)
    print(f"\nTest query: 'What is attention mechanism?'")
    print(f"Top result:\n{results[0].page_content[:300]}")