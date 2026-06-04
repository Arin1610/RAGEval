import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def call_llm(prompt: str, model: str = "llama-3.1-8b-instant") -> str:
    """Call Groq LLM with a prompt."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0  # deterministic for evaluation
    )
    return response.choices[0].message.content