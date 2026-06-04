import json
import time
import pandas as pd
from app.rag_chain import RAGChain
from eval.hallucination import score_hallucination
from eval.retrieval_eval import score_retrieval

def run_evaluation(qa_path: str = "results/qa_pairs.json", sample_size: int = 50):
    """Run full evaluation pipeline on QA pairs."""
    
    # Load QA pairs
    with open(qa_path) as f:
        qa_pairs = json.load(f)
    
    # Sample for evaluation (50 is enough for meaningful results)
    import random
    random.seed(42)
    sampled = random.sample(qa_pairs, min(sample_size, len(qa_pairs)))
    
    print(f"Running evaluation on {len(sampled)} QA pairs...")
    
    rag = RAGChain()
    results = []
    
    for i, qa in enumerate(sampled):
        question = qa["question"]
        ground_truth = qa["answer"]
        
        try:
            # RAG pipeline
            rag_result = rag.query(question)
            generated_answer = rag_result["answer"]
            context_docs = rag_result["context_docs"]
            context = rag_result["context"]
            
            # Hallucination score
            hall_score = score_hallucination(question, generated_answer, context)
            
            # Retrieval score
            ret_score = score_retrieval(question, context_docs, ground_truth)
            
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "generated_answer": generated_answer,
                "groundedness_score": hall_score["groundedness_score"],
                "reasoning": hall_score["reasoning"],
                "retrieval_score": ret_score["retrieval_score"],
                "best_chunk_score": ret_score["best_chunk_score"],
                "question_relevance": ret_score["question_relevance"]
            })
            
            print(f"[{i+1}/{len(sampled)}] ✅ Groundedness: {hall_score['groundedness_score']:.2f} | Retrieval: {ret_score['retrieval_score']:.3f}")
        
        except Exception as e:
            print(f"[{i+1}/{len(sampled)}] ❌ Failed — {e}")
        
        time.sleep(3)
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv("results/eval_results.csv", index=False)
    
    # Print summary
    print(f"\n--- EVALUATION SUMMARY ---")
    print(f"Total evaluated: {len(df)}")
    print(f"Avg Groundedness Score: {df['groundedness_score'].mean():.3f}")
    print(f"Hallucination Rate: {(df['groundedness_score'] < 0.5).mean()*100:.1f}%")
    print(f"Avg Retrieval Score: {df['retrieval_score'].mean():.3f}")
    print(f"Avg Best Chunk Score: {df['best_chunk_score'].mean():.3f}")
    print(f"Avg Question Relevance: {df['question_relevance'].mean():.3f}")
    
    return df

if __name__ == "__main__":
    df = run_evaluation(sample_size=50)