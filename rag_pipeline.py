from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
from prompts import RAG_PROMPT_BASELINE, RAG_PROMPT_STRICT
import re
from shared_models import embed_model
INJECTION_PATTERNS = [
    r"ignore (all |previous |above )?instructions",
    r"you are now",
    r"system prompt",
    r"diagnostic mode",
    r"repeat (all |the )?context",
    r"disregard",
    r"new instruction",
]

#embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def is_poisoned(chunk: str) -> bool:
    text = chunk.lower()
    return any(re.search(p, text) for p in INJECTION_PATTERNS)
    
# def load_chunks():
#     with open("/home/safrin/Projects/01/rag-vs-llm/data/resume.txt") as f:
#         text = f.read()
#     return [line.strip() for line in text.split("\n") if line.strip()]

def chunk_document(text):
    section_headers = [
        "PROFESSIONAL SUMMARY", "WORK EXPERIENCE", 
        "EDUCATION", "SKILLS", "CERTIFICATIONS", "REFERENCES"
    ]
    
    chunks = []
    current_section = []
    current_header = "HEADER"
    
    for line in text.split("\n"):
        if any(header in line.upper() for header in section_headers):
            # save previous section as one chunk
            if current_section:
                chunks.append("\n".join(current_section))
            current_section = [line]
            current_header = line.strip()
        else:
            current_section.append(line)
    
    # save last section
    if current_section:
        chunks.append("\n".join(current_section))
    
    return chunks
def build_index(sources: list[tuple[str, str]]):
    all_chunks = []
    metadata = []

    for filepath, label in sources:
        with open(filepath) as f:
            text = f.read()
        chunks = chunk_document(text)  # replaces the line-by-line split
        all_chunks.extend(chunks)
        metadata.extend([{"text": c, "source": label} for c in chunks])

    embeddings = embed_model.encode(all_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, metadata
    
def retrieve(query, index, metadata, k=8):
    q_emb = embed_model.encode([query])
    D, I = index.search(np.array(q_emb), k)
    retrieved = [metadata[i] for i in I[0]]
    clean = [m for m in retrieved if not is_poisoned(m["text"])]
    return clean
    
def looks_suspicious(response: str, context_chunks: list) -> bool:
    for chunk in context_chunks:
        # extract text from dict
        text = chunk["text"] if isinstance(chunk, dict) else chunk
        if text.lower() in response.lower():
            return True
    
    exfil_patterns = [r"here is the context", r"verbatim", r"as instructed", r"EXFIL", r"context_dump_complete"]
    return any(re.search(p, response.lower()) for p in exfil_patterns)
    
def ask_rag(question, index, metadata, prompt_version="baseline"):
    context_chunks = retrieve(question, index, metadata)
    context = "\n".join([m["text"] for m in context_chunks])
    
    if prompt_version == "strict":
        prompt = RAG_PROMPT_STRICT.format(context=context, question=question)
    else:
        prompt = RAG_PROMPT_BASELINE.format(context=context, question=question)
    
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response["message"]["content"]
    suspicious = looks_suspicious(answer, context_chunks)
    
    return answer, context_chunks, suspicious
