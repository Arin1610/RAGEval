import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_papers(papers_dir: str = "data/papers") -> list:
    """Load all PDFs from the papers directory."""
    documents = []
    pdf_files = [f for f in os.listdir(papers_dir) if f.endswith(".pdf")]
    
    print(f"Found {len(pdf_files)} papers to load...")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(papers_dir, pdf_file)
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)
            print(f"✅ Loaded: {pdf_file} ({len(docs)} pages)")
        except Exception as e:
            print(f"❌ Failed: {pdf_file} — {e}")
    
    print(f"\nTotal pages loaded: {len(documents)}")
    return documents


def chunk_documents(documents: list) -> list:
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks


if __name__ == "__main__":
    docs = load_papers()
    chunks = chunk_documents(docs)
    print(f"\nSample chunk:\n{chunks[0].page_content[:300]}")