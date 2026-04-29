# Ask My Research - Medical AI Assistant ⚕️

**Ask My Research** is a high-fidelity Retrieval-Augmented Generation (RAG) system built for medical professionals. It allows doctors and researchers to query an offline dataset of medical PDFs and receive highly accurate answers. Crucially, the system is designed to prevent hallucinations by strictly citing sources and defaulting to "I don't know" when the answer cannot be found in the provided texts. 

---

## 🏗️ Architecture & Workflow

This project utilizes a modern RAG stack:

1. **Document Ingestion (`ingest.py`)**: 
   - Uses `PyPDFDirectoryLoader` to read all medical PDFs from the `data/` directory.
   - Text is split into chunks using `RecursiveCharacterTextSplitter` (with overlap for context retention).
   - Chunks are embedded using `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) and stored in a persistent **ChromaDB** vector database.
   - The raw document chunks are also persisted locally for BM25 (keyword-based) retrieval.

2. **Hybrid Search & Reranking (`rag_chain.py`)**:
   - The system utilizes an **EnsembleRetriever** combining the strengths of semantic search (ChromaDB) and exact keyword matching (BM25).
   - A **Cohere Reranker** (`ContextualCompressionRetriever`) is applied on top of the retrieved results to ensure the most relevant chunks are sent to the LLM.

3. **Strict Generation**:
   - Uses OpenAI (`gpt-4`) with a rigorous System Prompt.
   - The LLM is instructed to only answer based on the provided context, include inline citations `[Source: document.pdf]`, and explicitly reject out-of-context queries.

4. **User Interface (`app.py`)**:
   - A modern chat interface built with **Streamlit**.
   - Displays real-time typing, inline citations, and expandable "References" sections displaying the exact chunk of text used for the answer.

5. **Evaluation & CI/CD Gating (`evaluate.py` & `ci_cd_gate.py`)**:
   - Integrated with the **Ragas** framework to evaluate a mock "Golden Dataset" on three core metrics: *Faithfulness*, *Answer Relevance*, and *Context Precision*.
   - A CI/CD gate script ensures that a pseudo-deployment is only allowed if all RAG quality metrics score above a defined threshold (e.g., `0.80`).

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+
- OpenAI API Key
- Cohere API Key

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/your-username/medical-rag.git
cd medical-rag
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the root of the project with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
```

### 4. Ingesting Data
Place your medical PDFs inside a folder named `data/` at the project root. Then run the ingestion pipeline:
```bash
python ingest.py
```
*This will create the `chroma_db/` folder and `bm25_docs.pkl`.*

### 5. Running the App
Start the Streamlit interface:
```bash
streamlit run app.py
```

---

## 📊 Evaluation & Testing

To evaluate the quality of the RAG system against your predefined dataset:

1. Run the evaluation script:
   ```bash
   python evaluate.py
   ```
   *This will generate a `ragas_score.json` and a detailed CSV report.*

2. Run the CI/CD Quality Gate to verify thresholds:
   ```bash
   python ci_cd_gate.py
   ```

---

## 🛠️ Tech Stack
- **Frameworks**: LangChain, Streamlit
- **LLM**: OpenAI (GPT-4)
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **Vector Store**: ChromaDB
- **Retrievers**: BM25, Chroma Vector Store
- **Reranker**: Cohere
- **Evaluation**: Ragas
