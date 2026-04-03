# rag-vs-llm
This project is a simple demonstration of LLM with RAG and No-RAG. This depicts how grounding and information retrieval contribute to the accuracy of the model output. It compares a standard LLM with a Retrieval-Augmented Generation (RAG) system using a fully local setup.

## Key Idea
- No-RAG: Generic answers (hallucination-prone)
- RAG: Context-grounded answers (more accurate)

## Tech Stack
- Ollama (mistral)
- FAISS (vector DB)
- Sentence Transformers (embeddings)

## Run
pip install -r requirements.txt

python compare.py
