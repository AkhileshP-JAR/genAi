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


# @st.cache_resource(show_spinner=False)
# def build_store():
#     texts = []
#     for p in FILES:
#         texts.append(p.read_text(encoding="utf-8"))
#         splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=40)
#         docs = []
#         for p, t in zip(FILES,texts):
#             for c in splitter.split_text(t):
#                 docs.append({"page_content": c, "metadata": {"source": p.name}})
#                 emb = OllamaEmbeddings(model=EMBED_MODEL)
#                 store = FAISS.from_texts([d["page_content"] for d in docs], emb, metadatas=[d["metadata"] for d in docs])
#     return store 
@st.cache_resource(show_spinner=False)
def build_store():
    texts = []
    for p in FILES:
        texts.append(p.read_text(encoding="utf-8"))

    splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=40)
    docs = []
    for p, t in zip(FILES, texts):
        for c in splitter.split_text(t):
            docs.append({"page_content": c, "metadata": {"source": p.name}})

    emb = OllamaEmbeddings(model=EMBED_MODEL)
    store = FAISS.from_texts(
        [d["page_content"] for d in docs], 
        emb, 
        metadatas=[d["metadata"] for d in docs]
    )
    return store


col = st.columns(2,gap="large")
with col[0]:
    if st.button("Run Indexing (vector store) "):
        t0 = time.time()
        st.session_state["store"] = build_store()
        st.success(f"Indexing done in {time.time()-t0:.1f} seconds")
with col[1]:
    st.write("Ollama should be running with models downloaded")

q = st.text_input("Your question about iphone")
go = st.button("ask")

def make_prompt(version, context, question):
    if version.startswith("v1"):
        return f"""Your are an imaginative storyteller.
        ignore accuracy and answer creatively.
        Question: {question}
        Answer:"""
    if version.startswith("v2"):
        return f"""you are a helpful Iphone expert.
        Use the context if relevent; if missing, make resonable inferencs.
        Context: {context}
        Question: {question}
        Answer:"""
    return f"""Your are precsie.
        Use only the context. If not in context, say "I don't know based on the doscuments"
        Context: {context}
        Question: {question}
        Answer with brief citations like [source]:
        """
if go:
    if "store" not in st.session_state:
        st.warning("Runf indexing frist")
    elif not q.strip:
        st.warning("Enter a question")
    else:
        store = st.session_state["store"]
        docs = store.similarity_search(q, k=top_k)
        context = "\n\n".join([f"{d.page_content} [{d.metadata['source']}]" for d in docs])
        prompt = make_prompt(version, context, q)
        model = HEAVY if model_pick == "Heavy first" else LIGHT
        llm = Ollama(model=model)
        with st.spinner(f"Generating with {model} ..."):
            out = llm.invoke(prompt)
        st.markdown(f"**Model**'{model} | **Prompt**: {version}  ")
        st.write(out)
