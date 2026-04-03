def evaluate_answer(answer, ground_truth):
    return ground_truth.lower() in answer.lower()

def compute_accuracy(results):
    total = len(results)

    rag_correct = sum(r["rag"]["correct"] for r in results)
    no_rag_correct = sum(r["no_rag"]["correct"] for r in results)

    return {
        "total_questions": total,
        "rag_accuracy": rag_correct / total,
        "no_rag_accuracy": no_rag_correct / total
    }