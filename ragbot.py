import os, time, streamlit as st
from pathlib import Path
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(layout="wide",page_title="DOBID EXTRACTOR" ,page_icon="🍌")
st.title("Student ID-> NAME &DOB")
st.caption("Tiny RAG with LangChain + FAISS + Ollama")

DATA_DIR = Path("data2")
FILES = [DATA_DIR/"iphone_history.txt", DATA_DIR/"iphone_specs.txt",DATA_DIR/"iphone_care.txt"]
EMBED_MODEL = "nomic-embed-text"
HEAVY =st.text_input("Heavy model","qwen2.5vl:3b")
LIGHT = st.text_input("light model","gemma3:1b")
top_k = st.slider("Top_k",1,8,5)
version = st.radio("Prompt version",
                   ["v1(hallucinate)","v2(loose RAG)","v3(Strict RAG)"],
                   horizontal=True)
model_pick =st.radio("Which model to use now?",["Heavy first","Light"], horizontal=True)


@st.cache_resource(show_spinner=False)
def build_store():
    texts = []
    for p in Files:
        texts.append(p.read_text(encoding="utf-8"))
        splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=40)
        docs = []
        for p, t in zip(FILES,texts):
            for c in splitter.spilt_text(t):
                docs.append({"page_content": c, "metadata": {"source": p.name}})
                emb = OllamaEmbeddings(model=EMBED_MODEL)
                store = FAISS.from_texts([d["page_content"] for d in docs], emb, metadatas=[d["metadata"] for d in docs])
    return store 