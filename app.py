import streamlit as st
import os
from rag_chain import get_rag_chain

st.set_page_config(page_title="Ask My Research", page_icon="⚕️", layout="wide")

st.title("⚕️ Ask My Research - Medical AI Assistant")

st.markdown("""
Welcome to the Medical AI Assistant. This tool searches through your ingested medical PDFs
and provides answers strictly based on the provided documents with inline citations.
""")

try:
    with st.spinner("Initializing RAG chain..."):
        if "rag_chain" not in st.session_state:
            st.session_state.rag_chain = get_rag_chain()
except Exception as e:
    st.error(f"Error initializing RAG chain: {e}")
    st.info("Make sure you have ingested documents by running `python ingest.py` and set your API keys.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a medical question based on the documents..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Searching and synthesizing answer..."):
            try:
                response = st.session_state.rag_chain.invoke({"input": prompt})
                answer = response.get("answer", "I don't know")
                context_docs = response.get("context", [])

                st.markdown(answer)

                if context_docs:
                    st.markdown("### References")
                    for doc in context_docs:
                        source = doc.metadata.get("source_doc", "Unknown Source")
                        chunk = doc.metadata.get("chunk_id", "Unknown Chunk")
                        
                        with st.expander(f"Reference: [Source: {source}] - Chunk {chunk}"):
                            st.write(doc.page_content)
                
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Error generating response: {e}")
