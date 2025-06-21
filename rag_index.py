# rag_index.py
import os
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

from load_documents import load_local_documents
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)

load_dotenv()


# rag_index.py
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL   = os.getenv("REPO_ID", "HuggingFaceH4/zephyr-7b-beta")
HF_TOKEN    = os.getenv("HUGGINGFACE_API_TOKEN")

# ---------- builders ---------- #
def _chunks_from_docs(docs: List[Document]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def _embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def _llm():
    return HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.3,
        max_new_tokens=512,
    )

def build_rag_chain_from_documents(docs: List[Document]) -> RetrievalQA:
    chunks     = _chunks_from_docs(docs)
    vectorstore = FAISS.from_documents(chunks, embedding=_embeddings())
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})

    return RetrievalQA.from_chain_type(
        llm=_llm(),
        retriever=retriever,
        return_source_documents=True,
    )

# ✅ NEW: load a *single* file safely
def load_single_file(file_path: str) -> List[Document]:
    ext = Path(file_path).suffix.lower()
    if ext == ".txt":
        loader = TextLoader(file_path)
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return loader.load()

# ✅ FIXED: Use correct loader depending on input
def build_rag_chain_from_file(file_path: str) -> RetrievalQA:
    docs = load_single_file(file_path)
    return build_rag_chain_from_documents(docs)

# ✅ LLM-only fallback
def get_llm_only_chain():
    return RetrievalQA.from_chain_type(llm=_llm(), retriever=None, return_source_documents=False)
