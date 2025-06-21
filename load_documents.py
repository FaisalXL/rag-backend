from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from pathlib import Path
from typing import List

def load_local_documents(folder_path: str) -> List[Document]:
    """
    Load all .txt, .pdf, and .docx files from a folder and return as list of LangChain Documents.
    """
    docs = []

    folder = Path(folder_path)
    for file in folder.iterdir():
        if file.suffix == ".txt":
            loader = TextLoader(str(file))
        elif file.suffix == ".pdf":
            loader = UnstructuredPDFLoader(str(file))
        elif file.suffix == ".docx":
            loader = UnstructuredWordDocumentLoader(str(file))
        else:
            print(f"Skipping unsupported file: {file}")
            continue

        try:
            file_docs = loader.load()
            docs.extend(file_docs)
            print(f"Loaded {len(file_docs)} documents from {file.name}")
        except Exception as e:
            print(f"Failed to load {file.name}: {e}")

    print(f"\n✅ Total documents loaded: {len(docs)}")
    return docs
