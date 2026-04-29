import json
import os
from datasets import Dataset
from ragas import evaluate

# ✅ Correct metric imports (latest Ragas)
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision
)

from rag_chain import get_rag_chain
from dotenv import load_dotenv
import pandas as pd

# ✅ NEW: HuggingFace embeddings (fix for your error)
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ✅ Redirect OpenAI → GitHub (for Ragas LLM)
os.environ["OPENAI_API_KEY"] = os.getenv("GITHUB_TOKEN")
os.environ["OPENAI_BASE_URL"] = "https://models.github.ai/inference"

# ✅ Use HF embeddings instead of OpenAI
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Example Golden Dataset
GOLDEN_DATASET = [
    {
        "question": "What are Antiulcer drugs",
        "ground_truth": "Antiulcer drugs are a class of drugs, exclusive of the antibacterial agents, used to treat ulcers in the stomach and the upper part of the small intestine."
    },
    {
        "question": "What is Cytomegalovirus (CMV)?",
        "ground_truth": "A type of virus that attacks and enlarges certain cells in the body. The virus also causes a disease in infants."
    }
]


def run_evaluation():
    print("Initializing RAG chain for evaluation...")
    
    try:
        rag_chain = get_rag_chain()
    except Exception as e:
        print(f"Error initializing RAG chain. Ensure ChromaDB exists: {e}")
        return

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("Generating responses for Golden Dataset...")

    for item in GOLDEN_DATASET:
        question = item["question"]

        try:
            response = rag_chain.invoke({"input": question})

            answer = response.get("answer", "")
            context_docs = response.get("context", [])

            # Convert docs → strings
            context_strings = [doc.page_content for doc in context_docs]

            questions.append(question)
            answers.append(answer)
            contexts.append(context_strings)
            ground_truths.append(item["ground_truth"])

        except Exception as e:
            print(f"Error for question: {question}\n{e}")
            continue

    if not questions:
        print("No questions processed. Exiting evaluation.")
        return

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }

    dataset = Dataset.from_dict(data)

    print("Running ragas evaluation...")

    # ✅ FIXED metrics (class-based)
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision()
    ]

    # ✅ FIXED: pass embeddings (CRITICAL FIX)
    result = evaluate(
        dataset,
        metrics=metrics,
        embeddings=embeddings
    )

    # Save results
    result_df = result.to_pandas()
    result_df.to_csv("evaluation_results.csv", index=False)

    # ✅ FIXED: correct access
    scores = {
        "faithfulness": result["faithfulness"],
        "answer_relevancy": result["answer_relevancy"],
        "context_precision": result["context_precision"]
    }

    with open("ragas_score.json", "w") as f:
        json.dump(scores, f, indent=4)

    print("\n✅ Evaluation completed!")
    print("📄 Saved to: evaluation_results.csv & ragas_score.json\n")
    print("📊 Scores:")
    print(scores)


if __name__ == "__main__":
    run_evaluation()