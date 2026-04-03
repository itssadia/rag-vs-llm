from rag_pipeline import load_chunks, build_index, ask_rag
from no_rag_pipeline import ask_llm
from evaluation import evaluate_answer, compute_accuracy
import json
import os

questions = [
    {"q": "How many records did the candidate process daily?", "ground_truth": "3M"},
    {"q": "What industry experience does the candidate have?", "ground_truth": "telecom"},
    {"q": "What is a key achievement?", "ground_truth": "12%"},
    {"q": "Is the candidate suitable for an ML role?", "ground_truth": "machine learning"}
]

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

# Build RAG index
chunks = load_chunks()
index, chunks = build_index(chunks)

results = []

for item in questions:
    q = item["q"]
    gt = item["ground_truth"]

    print(f"\nQuestion: {q}")

    no_rag_ans = ask_llm(q)
    rag_ans, context = ask_rag(q, index, chunks)

    results.append({
        "question": q,
        "ground_truth": gt,
        "no_rag": {
            "answer": no_rag_ans,
            "correct": evaluate_answer(no_rag_ans, gt)
        },
        "rag": {
            "answer": rag_ans,
            "correct": evaluate_answer(rag_ans, gt),
            "retrieved_context": context
        }
    })

summary = compute_accuracy(results)

output = {
    "results": results,
    "summary": summary
}

with open("results/evaluation.json", "w") as f:
    json.dump(output, f, indent=4)

print("\n✅ Done! Check results/evaluation.json")
print(summary)