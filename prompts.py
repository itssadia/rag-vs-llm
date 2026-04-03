RAG_PROMPT = """
You are a helpful assistant. Answer ONLY using the provided context.

If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

NO_RAG_PROMPT = """
Answer the following question as accurately as possible.

Question:
{question}

Answer:
"""