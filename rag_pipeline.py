from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
from prompts import RAG_PROMPT

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

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
    return [chunks[i] for i in I[0]]

def ask_rag(question, index, chunks):
    context_chunks = retrieve(question, index, chunks)
    context = "\n".join(context_chunks)

    prompt = RAG_PROMPT.format(context=context, question=question)

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"], context_chunks