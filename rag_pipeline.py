from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
from prompts import RAG_PROMPT
import re

INJECTION_PATTERNS = [
    r"ignore (all |previous |above )?instructions",
    r"you are now",
    r"system prompt",
    r"diagnostic mode",
    r"repeat (all |the )?context",
    r"disregard",
    r"new instruction",
]

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def is_poisoned(chunk: str) -> bool:
    text = chunk.lower()
    return any(re.search(p, text) for p in INJECTION_PATTERNS)
    
def load_chunks():
    with open("data/resume.txt") as f:
        text = f.read()
    return [line.strip() for line in text.split("\n") if line.strip()]

def build_index(chunks):
    embeddings = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, chunks

def retrieve(query, index, chunks, k=2):
    q_emb = embed_model.encode([query])
    D, I = index.search(np.array(q_emb), k)
    retrieved = [chunks[i] for i in I[0]]
    clean = [c for c in retrieved if not is_poisoned(c)] #retreival guardrail
    return clean

def ask_rag(question, index, chunks):
    context_chunks = retrieve(question, index, chunks)
    context = "\n".join(context_chunks)

    prompt = RAG_PROMPT.format(context=context, question=question)

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"], context_chunks
