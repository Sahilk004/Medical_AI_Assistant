import json
import os
import sys

SCORE_FILE = "ragas_score.json"
FAITHFULNESS_THRESHOLD = 0.8
ANSWER_RELEVANCY_THRESHOLD = 0.8
CONTEXT_PRECISION_THRESHOLD = 0.8


def avg(score):
    """Handle both list and float cases safely"""
    if isinstance(score, list):
        return sum(score) / len(score) if score else 0.0
    return score


def check_deployment_gate():
    if not os.path.exists(SCORE_FILE):
        print(f"Error: {SCORE_FILE} not found. Run evaluate.py first.")
        sys.exit(1)

    with open(SCORE_FILE, "r") as f:
        scores = json.load(f)

    # ✅ Convert to averages
    faithfulness = avg(scores.get("faithfulness", 0.0))
    answer_relevancy = avg(scores.get("answer_relevancy", 0.0))
    context_precision = avg(scores.get("context_precision", 0.0))

    print("--- CI/CD RAG Quality Gate ---")
    print(f"Faithfulness: {faithfulness:.2f} (Threshold: {FAITHFULNESS_THRESHOLD})")
    print(f"Answer Relevancy: {answer_relevancy:.2f} (Threshold: {ANSWER_RELEVANCY_THRESHOLD})")
    print(f"Context Precision: {context_precision:.2f} (Threshold: {CONTEXT_PRECISION_THRESHOLD})")

    # ✅ Gate check
    if (
        faithfulness >= FAITHFULNESS_THRESHOLD and
        answer_relevancy >= ANSWER_RELEVANCY_THRESHOLD and
        context_precision >= CONTEXT_PRECISION_THRESHOLD
    ):
        print("\n✅ Quality Gate Passed. Deployment authorized.")
        print("Starting mock deployment...")
        sys.exit(0)
    else:
        print("\n❌ Quality Gate Failed. Deployment rejected due to low RAG scores.")
        sys.exit(1)


if __name__ == "__main__":
    check_deployment_gate()