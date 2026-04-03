import ollama
from prompts import NO_RAG_PROMPT

def ask_llm(question):
    prompt = NO_RAG_PROMPT.format(question=question)

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]