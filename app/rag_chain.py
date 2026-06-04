from app.retriever import load_vectorstore
from app.llm_client import call_llm

PROMPT_TEMPLATE = """You are a helpful AI research assistant. Answer the question based ONLY on the provided context. 
If the answer cannot be found in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""

class RAGChain:
    def __init__(self, k: int = 4):
        self.vectorstore = load_vectorstore()
        self.k = k

    def retrieve(self, question: str) -> list:
        """Retrieve top k relevant chunks."""
        docs = self.vectorstore.similarity_search(question, k=self.k)
        return docs

    def generate(self, question: str, context_docs: list) -> str:
        """Generate answer from retrieved context."""
        context = "\n\n".join([doc.page_content for doc in context_docs])
        prompt = PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )
        return call_llm(prompt)

    def query(self, question: str) -> dict:
        """Full RAG pipeline — retrieve + generate."""
        context_docs = self.retrieve(question)
        answer = self.generate(question, context_docs)
        return {
            "question": question,
            "answer": answer,
            "context_docs": context_docs,
            "context": "\n\n".join([doc.page_content for doc in context_docs])
        }

if __name__ == "__main__":
    rag = RAGChain()
    
    # Test query
    result = rag.query("What is the attention mechanism in transformers?")
    print(f"Question: {result['question']}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources: {len(result['context_docs'])} chunks retrieved")