import os
import pickle
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# ✅ ADD THESE (same as your first code)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
ENDPOINT = "https://models.github.ai/inference"
MODEL_NAME = "openai/gpt-4.1"

CHROMA_PATH = "chroma_db"
BM25_DOCS_PATH = "bm25_docs.pkl"

def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    if not os.path.exists(BM25_DOCS_PATH):
        raise FileNotFoundError("BM25 documents not found. Run ingest.py first.")
    
    with open(BM25_DOCS_PATH, "rb") as f:
        chunks = pickle.load(f)
    
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY not found in environment variables.")

    compressor = CohereRerank(
    cohere_api_key=cohere_api_key,
    model="rerank-english-v3.0",
    top_n=3
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

    return compression_retriever


def get_rag_chain():
    # ✅ FIXED LLM (GitHub instead of OpenAI)
    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=ENDPOINT,
        api_key=GITHUB_TOKEN,
        temperature=0
    )

    system_prompt = (
        "You are an expert AI assistant for medical professionals. "
        "Your task is to answer questions strictly based on the provided context. "
        "If the answer cannot be found in the context, you must answer exactly "
        "'I don't know'. You must not hallucinate or use any knowledge outside of the provided context. "
        "When providing your answer, you MUST use inline citations indicating the exact source. "
        "Format your inline citations using brackets mapping to the source document and chunk format, e.g., [Source: <source_doc>].\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    compression_retriever = get_retriever()

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(compression_retriever, question_answer_chain)

    return rag_chain


if __name__ == "__main__":
    retriever = get_retriever()
    print("Retriever configured successfully.")