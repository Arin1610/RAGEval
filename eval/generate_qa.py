import json
import time
import random
from app.ingest import load_papers, chunk_documents
from app.llm_client import call_llm

QA_PROMPT = """You are an expert AI researcher. Given the following text from a research paper, generate a question and answer pair.

Rules:
- The question must be answerable ONLY from the provided text
- The answer must be factual and grounded in the text
- Keep the answer concise (1-3 sentences)
- Return ONLY valid JSON in this exact format, nothing else:
{{"question": "your question here", "answer": "your answer here"}}

Text:
{chunk}

JSON:"""

def generate_qa_pairs(num_pairs: int = 200) -> list:
    """Generate QA pairs from paper chunks."""
    docs = load_papers()
    chunks = chunk_documents(docs)
    
    # Filter chunks that are long enough to generate good questions
    valid_chunks = [c for c in chunks if len(c.page_content) > 200]
    print(f"Valid chunks for QA generation: {len(valid_chunks)}")
    
    # Sample random chunks
    selected_chunks = random.sample(valid_chunks, min(num_pairs, len(valid_chunks)))
    
    qa_pairs = []
    failed = 0
    
    for i, chunk in enumerate(selected_chunks):
        try:
            prompt = QA_PROMPT.format(chunk=chunk.page_content[:800])
            response = call_llm(prompt)
            
            # Clean response and parse JSON
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            qa = json.loads(response)
            qa["source"] = chunk.metadata.get("source", "unknown")
            qa["chunk_text"] = chunk.page_content[:500]
            qa_pairs.append(qa)
            
            print(f"[{i+1}/{len(selected_chunks)}] ✅ Generated QA pair")
        
        except Exception as e:
            failed += 1
            print(f"[{i+1}/{len(selected_chunks)}] ❌ Failed — {e}")
        
        time.sleep(0.3)  # Rate limiting
    
    print(f"\nGenerated {len(qa_pairs)} QA pairs ({failed} failed)")
    return qa_pairs

def save_qa_pairs(qa_pairs: list, path: str = "results/qa_pairs.json"):
    """Save QA pairs to JSON file."""
    with open(path, "w") as f:
        json.dump(qa_pairs, f, indent=2)
    print(f"Saved {len(qa_pairs)} QA pairs to {path}")

if __name__ == "__main__":
    print("Generating QA pairs from research papers...")
    qa_pairs = generate_qa_pairs(num_pairs=200)
    save_qa_pairs(qa_pairs)
    
    # Preview first 3
    print("\n--- Sample QA Pairs ---")
    for qa in qa_pairs[:3]:
        print(f"\nQ: {qa['question']}")
        print(f"A: {qa['answer']}")