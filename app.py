import os
import shutil
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── Configuration ─────────────────────────────────────────────────────────────
CHROMA_DIR = "./chroma_db_nomic"
UPLOAD_DIR = "./data"

MODEL_PATH = r"C:\Users\MY PC\Desktop\document-rag\models\Qwen3-4B-Q4_K_M.gguf"
# MODEL_PATH = r"C:\Users\MY PC\Desktop\document-rag\models\DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
# MODEL_PATH = r"C:\Users\MY PC\rag-chatbot\models\Qwen3.5-0.8B-Q5_K_M.gguf"
# MODEL_PATH = r"C:\Users\MY PC\Nadi\models\nadi-tesseract-v11\Nadi_Tesseract_V11.gguf"

N_CTX        = 4096
N_GPU_LAYERS = 0
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Document RAG", page_icon="📄", layout="wide")
st.title("📄 Multi-Document RAG")


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5",
                                  model_kwargs={"trust_remote_code": True})


@st.cache_resource(show_spinner="Loading LLM...")
def get_llm():
    return LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        temperature=0.1,
        max_tokens=512,
        repeat_penalty=1.3,
        stop=["Question:", "\nQuestion", "\nContext", "<|im_end|>", "<|end|>", "<|EOT|>"],
        verbose=False,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_vector_store(pdf_paths: list[str]) -> Chroma:
    embeddings = get_embeddings()
    docs = []
    for path in pdf_paths:
        try:
            docs.extend(PyPDFLoader(path).load())
        except Exception as e:
            st.warning(f"Could not load {os.path.basename(path)}: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Close existing connection before deleting files
    if "vector_store" in st.session_state:
        try:
            st.session_state.vector_store._client.reset()
        except Exception:
            pass
        del st.session_state["vector_store"]

    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="rag_docs",
    )


def load_vector_store() -> Chroma:
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
        collection_name="rag_docs",
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def answer(query: str, vector_store: Chroma) -> str:
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    prompt = ChatPromptTemplate.from_template(
        """<|im_start|>system
You are a helpful assistant. Answer using ONLY the context provided. Be concise. If the answer is not in the context, say "I don't know."<|im_end|>
<|im_start|>user
Context:
{context}

Question: {question}<|im_end|>
<|im_start|>assistant
<think>

</think>
"""
    )
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    import re
    result = chain.invoke(query)
    # Strip any Qwen3 thinking block
    result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
    return result


# ── Sidebar: upload & index ───────────────────────────────────────────────────

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if uploaded:
        saved_paths = []
        for f in uploaded:
            dest = os.path.join(UPLOAD_DIR, f.name)
            with open(dest, "wb") as out:
                out.write(f.read())
            saved_paths.append(dest)

        if st.button("Index documents", type="primary"):
            with st.spinner(f"Indexing {len(saved_paths)} file(s)..."):
                vs = build_vector_store(saved_paths)
                st.session_state.vector_store = vs
            st.success(f"Indexed {len(saved_paths)} document(s).")

    elif os.path.exists(CHROMA_DIR):
        st.info("Using existing vector DB.")
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = load_vector_store()

    if "vector_store" in st.session_state:
        if st.button("Clear index"):
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)
            del st.session_state.vector_store
            st.session_state.messages = []
            st.rerun()


# ── Chat ──────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if "vector_store" not in st.session_state:
    st.info("Upload and index PDFs using the sidebar to get started.")
else:
    if query := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = answer(query, st.session_state.vector_store)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
