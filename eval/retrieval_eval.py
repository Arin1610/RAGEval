import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def score_retrieval(question: str, retrieved_chunks: list, ground_truth_answer: str) -> dict:
    """Score retrieval quality by measuring semantic similarity between
    retrieved context and ground truth answer."""
    
    if not retrieved_chunks:
        return {"retrieval_score": 0.0, "best_chunk_score": 0.0}
    
    # Embed question and ground truth
    question_embedding = model.encode([question])
    answer_embedding = model.encode([ground_truth_answer])
    
    # Embed all retrieved chunks
    chunk_texts = [chunk.page_content for chunk in retrieved_chunks]
    chunk_embeddings = model.encode(chunk_texts)
    
    # Score 1: How similar are retrieved chunks to the ground truth answer?
    answer_chunk_similarities = cosine_similarity(answer_embedding, chunk_embeddings)[0]
    best_chunk_score = float(np.max(answer_chunk_similarities))
    avg_chunk_score = float(np.mean(answer_chunk_similarities))
    
    # Score 2: How relevant are chunks to the question?
    question_chunk_similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
    question_relevance = float(np.mean(question_chunk_similarities))
    
    return {
        "retrieval_score": avg_chunk_score,
        "best_chunk_score": best_chunk_score,
        "question_relevance": question_relevance
    }

if __name__ == "__main__":
    from app.rag_chain import RAGChain
    
    rag = RAGChain()
    
    test_question = "What is multi-head attention?"
    test_answer = "Multi-head attention runs attention in parallel across multiple representation subspaces."
    
    retrieved = rag.retrieve(test_question)
    scores = score_retrieval(test_question, retrieved, test_answer)
    
    print(f"Retrieval Score: {scores['retrieval_score']:.3f}")
    print(f"Best Chunk Score: {scores['best_chunk_score']:.3f}")
    print(f"Question Relevance: {scores['question_relevance']:.3f}")