# rag-vs-llm
This project is a simple demonstration of LLM with RAG and No-RAG. This depicts how grounding and information retrieval contribute to the accuracy of the model output. It compares a standard LLM with a Retrieval-Augmented Generation (RAG) system using a fully local setup. Here, the model will answer few questions regarding a candidates eligibility for ML engineer role and the external resource for RAG here is the resume uploaded by the candidate. 

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

# Updates:
I further investigated by giving the resume.txt file as a context prompt to the no_rag LLM and it was able to successfully answer the questions meeting the ground truths. Even in case of using a 1823 characters prompt, similar behavior was noticed. 

It is because Mistral is pretty capable of handling this. However, with larger context prompt the latency was higher. However, I opted out from experimenting with highly noised large prompts due to limited computational resource.

Another notable point is, the accuracy metric here is totally dependent on finding the ground truth as an exact match. So, in case of using direct contextual prompt without RAG, results like "million" is considered wrong if the ground truth is "M", which is not ideal.

To conclude, this experiment can be seen just as an idea of how information is retrieved using RAG and how LLM can benefit from it. This very model or dataset do not claim production level example/benchmark. It is more about encouraging to start thinking about the backend technology of RAG.

