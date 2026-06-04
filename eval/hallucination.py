import json
from app.llm_client import call_llm

HALLUCINATION_PROMPT = """You are evaluating if an AI answer is grounded in context.

Context: {context}

Question: {question}

Answer: {answer}

Step 1: List the key claims in the answer.
Step 2: For each claim, check if it's supported by the context.
Step 3: Calculate the fraction of claims supported.

Rules:
- Paraphrasing counts as supported
- Reasonable inference counts as supported  
- Only completely unsupported facts count against

Return ONLY this JSON:
{{"groundedness_score": 0.7, "reasoning": "X out of Y claims supported"}}

JSON:"""

def score_hallucination(question: str, answer: str, context: str) -> dict:
    """Score whether an answer is grounded in the context."""
    prompt = HALLUCINATION_PROMPT.format(
        context=context[:2000],
        question=question,
        answer=answer
    )
    
    try:
        response = call_llm(prompt, model="llama-3.3-70b-versatile")
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        
        result = json.loads(response)
        return {
            "groundedness_score": float(result["groundedness_score"]),
            "reasoning": result.get("reasoning", "")
        }
    except Exception as e:
        return {
            "groundedness_score": 0.0,
            "reasoning": f"Evaluation failed: {e}"
        }

if __name__ == "__main__":
    test_question = "What is the attention mechanism?"
    test_answer = "The attention mechanism allows the model to focus on relevant parts of the input sequence."
    test_context = """The attention mechanism is a key component of the Transformer architecture. 
    It allows the model to weigh the importance of different input tokens when generating each output token.
    Multi-head attention runs the attention function in parallel across multiple representation subspaces."""
    
    result = score_hallucination(test_question, test_answer, test_context)
    print(f"Groundedness Score: {result['groundedness_score']}")
    print(f"Reasoning: {result['reasoning']}")