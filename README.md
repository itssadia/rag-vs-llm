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

# How it looks after running the compare.py (I kept the resume.txt file very simple, will further experiment with changes)
<img width="1254" height="1254" alt="57f8ac4e-9aca-4d32-b47e-786d9b07d27a" src="https://github.com/user-attachments/assets/08d1dc2c-48a9-46fb-b6f8-cddb36005dc6" />
